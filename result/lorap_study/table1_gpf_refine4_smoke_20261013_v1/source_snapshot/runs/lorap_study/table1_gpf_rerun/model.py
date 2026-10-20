"""GPF-plus inputs and low-rank aggregation prompts on the paired backbone.

The aggregation prompt is inserted immediately before native update(): after
QAT/DQ aggregation quantization, including BOTH DQ precision populations.
Equal-width layers share their scoring network, with independent prompt factors.
"""
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from prompt.GPF import GPF_plus, GPF_plus_LoRA_Layer
from ..table1_frozen.backends import load_state, optimizer, predict
from ..table1_frozen.common import cpu_tree, rng_state, restore_rng
from ..table1_frozen.data import labels_and_output
from ..table1_frozen.train import task_loss
from ..table1_graphlora.model import sites


def _nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite nonnegative number')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a finite nonnegative number') from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return value


def _scaled_add(prompt, value, scale):
    # Preserve the original floating-point arithmetic for historical defaults.
    prompted = prompt.add(value)
    return prompted if scale == 1. else value + scale * (prompted - value)


class GPFLoRAP(nn.Module):
    def __init__(self, backbone, config, seed, example, rank=2, num_prompts=20,
                 input_scale=1., aggregation_scale=1., zero_init=False):
        super().__init__()
        if rank < 1 or num_prompts < 1:
            raise ValueError('Positive rank and prompt count required')
        self.input_scale = _nonnegative(input_scale, 'input_scale')
        self.aggregation_scale = _nonnegative(aggregation_scale, 'aggregation_scale')
        if not isinstance(zero_init, bool):
            raise ValueError('zero_init must be a boolean')
        self.zero_init = zero_init
        self.backbone, self.config, self.frozen = backbone, config, False
        self.prompts = nn.ModuleDict()
        self.dimensions, self.handles, self.originals = [], [], []
        self.enabled = False
        for i, (name, layer) in enumerate(sites(backbone)):
            self.dimensions.append(dict(name=name, insertion='before_native_update'))
            native = layer.update
            self.originals.append((layer, native))

            def update(module, value, *args, i=i, native=native, **kwargs):
                if self.enabled:
                    shape = value.shape
                    value = _scaled_add(self.prompts['aggregation'][i], value.flatten(1),
                                        self.aggregation_scale).view(shape)
                else:
                    self.dimensions[i]['width'] = value.flatten(1).size(1)
                return native(value, *args, **kwargs)

            layer.update = MethodType(update, layer)
        if len(self.dimensions) != config['layers']:
            raise ValueError('Unexpected message-passing layer count')
        state, rng = cpu_tree(backbone.state_dict()), rng_state(example.x.device)
        training = backbone.training
        try:
            with torch.no_grad():
                backbone.train()
                backbone(example)
        finally:
            load_state(backbone, state)
            restore_rng(rng, example.x.device)
            backbone.train(training)
        if config['node']:
            input_module = sites(backbone)[0][1]
            input_width, post = example.x.size(1), False
        else:
            input_module = getattr(backbone, 'embeding', None)
            if input_module is None:
                input_module = backbone.embedding
            post = isinstance(input_module, nn.Embedding)
            input_width = input_module.embedding_dim if post else input_module.in_features
        self.input_site = dict(width=input_width, insertion='after_embedding' if post else 'before_input_projection')
        # CPU-only private stream: construction consumes no training/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 915000)
            self.prompts['input'] = GPF_plus(input_width, num_prompts)
            scores = {}
            layers = nn.ModuleList()
            for d in self.dimensions:
                width = d['width']
                if width not in scores:
                    scores[width] = nn.Linear(width, num_prompts)
                layers.append(GPF_plus_LoRA_Layer(width, num_prompts, rank, scores[width]))
            self.prompts['aggregation'] = layers
            if zero_init:
                with torch.no_grad():
                    self.prompts['input'].p_list.zero_()
                    for prompt in layers:
                        prompt.P_left.zero_()
        self.prompts.to(example.x.device)
        if post:
            def after(module, args, value):
                return _scaled_add(self.prompts['input'], value, self.input_scale) if self.enabled else value
            self.handles.append(input_module.register_forward_hook(after))
        else:
            def before(module, args):
                return (_scaled_add(self.prompts['input'], args[0], self.input_scale), *args[1:]) if self.enabled else args
            self.handles.append(input_module.register_forward_pre_hook(before))
        self.enabled = True
        self.train(training)

    def forward(self, data):
        return self.backbone(data)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, original in reversed(self.originals):
            module.update = original
        self.handles.clear()
        self.originals.clear()


def make_optimizer(model, c, prompt_lr=None, prompt_weight_decay=0.):
    prompt_lr = _nonnegative(c['lr'] if prompt_lr is None else prompt_lr, 'prompt_lr')
    prompt_weight_decay = _nonnegative(prompt_weight_decay, 'prompt_weight_decay')
    opt = optimizer(model.backbone, c)
    opt.add_param_group(dict(params=list(model.prompts.parameters()), name='gpf_lorap',
                            lr=prompt_lr, weight_decay=prompt_weight_decay))
    ids = [id(p) for group in opt.param_groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer has omitted or duplicated parameters')
    return opt


def counts(model):
    number = sum(p.numel() for p in model.prompts.parameters())
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                adapter_parameters=number, prompt_parameters=number,
                input_prompt_parameters=sum(p.numel() for p in model.prompts['input'].parameters()),
                aggregation_prompt_parameters=sum(p.numel() for p in model.prompts['aggregation'].parameters()),
                adapter_fp32_bytes=sum(p.numel() * p.element_size() for p in model.prompts.parameters()),
                adapter_buffer_bytes=0, sites=model.dimensions, input_site=model.input_site,
                input_scale=model.input_scale, aggregation_scale=model.aggregation_scale,
                zero_init=model.zero_init)


def objective(model, c, batch, hp):
    out, budget = predict(model, c, batch)
    logits, y = labels_and_output(c, batch, out, 'train')
    task = task_loss(c, logits, y)
    storage = c['a_loss'] * F.relu(budget - c['a_storage']).square()
    zero = task.new_zeros(())
    return task + storage, logits, y, dict(task=task, storage=storage, contrast=zero, structure=zero)

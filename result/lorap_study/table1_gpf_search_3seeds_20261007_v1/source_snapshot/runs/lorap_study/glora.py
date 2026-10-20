"""GLoRA's five support tensors and 432-path supernet on existing QLinear layers.

Adapted from Arnav Chavan's MIT-licensed official GLoRA/layer.py (2022).
Pinned source and license: glora_official/. The GNN/QAT extension substitutes
W_q and x_q in the official formula; adapters stay FP32 and are never requantized.
"""
import copy
import hashlib
import itertools
import random

import torch
from torch import nn
from torch.nn import functional as F

from .lora import FixedGraphLinear, GraphLinear, NodeLinear, build_baseline
from .models import optimizer_for

PATHS = ('A', 'B', 'C', 'D', 'E')
CHOICES = (('lora', 'vector', 'constant', 'none'),
           ('lora', 'vector', 'constant', 'none'),
           ('lora', 'vector', 'none'), ('constant', 'none', 'vector'), ('constant', 'none', 'vector'))
ARCHITECTURES = tuple(itertools.product(*CHOICES))
REFERENCE_ARCH = ARCHITECTURES.index(('lora', 'lora', 'lora', 'vector', 'vector'))
NONE_ARCH = ARCHITECTURES.index(('none',)*5)
ADAPTER_TENSORS = ('Ad', 'Au', 'Bd', 'Bu', 'Cd', 'Cu', 'D', 'E')


def adapter_name(name):
    return any(part.startswith('glora_') for part in name.split('.'))


def _copy_base(target, base):
    if base.weight.dtype != torch.float32:
        raise ValueError('GLoRA joint-QAT requires FP32 master weights')
    target.in_features, target.out_features = base.in_features, base.out_features
    target.weight, target.bias = base.weight, base.bias
    target.weight_quant, target.fea_quant, target.bit = base.weight_quant, base.fea_quant, base.bit
    target.graph_level = isinstance(base, (GraphLinear, FixedGraphLinear)) or getattr(base, 'graph_level', False)
    target.training = base.training


class GLoRAQLinear(nn.Module):
    def __init__(self, base, rank=2, rng=None):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError('GLoRA rank must be a positive integer')
        _copy_base(self, base)
        self.rank, self.sample_paths = rank, True
        self._path_rng = rng if rng is not None else random.Random(0)
        self.register_buffer('glora_architecture', torch.tensor(REFERENCE_ARCH, dtype=torch.long))
        for path, rows, columns in (('A', self.out_features, self.in_features),
                                    ('B', self.out_features, self.in_features), ('C', self.in_features, 1)):
            down = nn.Parameter(self.weight.new_zeros(rows, rank))
            up = nn.Parameter(self.weight.new_empty(rank, columns))
            nn.init.xavier_uniform_(up)
            setattr(self, f'glora_{path}d', down)
            setattr(self, f'glora_{path}u', up)
        self.glora_D = nn.Parameter(self.weight.new_zeros(self.out_features))
        self.glora_E = nn.Parameter(self.weight.new_zeros(self.out_features))

    def set_architecture(self, architecture):
        if isinstance(architecture, bool) or not isinstance(architecture, int) or not 0 <= architecture < len(ARCHITECTURES):
            raise ValueError('Invalid GLoRA architecture index')
        self.glora_architecture.fill_(architecture)

    def support(self, path, mode):
        if path in 'ABC':
            down, up = getattr(self, f'glora_{path}d'), getattr(self, f'glora_{path}u')
            if mode == 'lora':
                return down @ up
            if mode == 'vector':
                return down[:, :1]
            if mode == 'constant':
                return down[0, 0]
            return down.new_zeros(down.shape[0], up.shape[1])
        value = getattr(self, f'glora_{path}')
        if mode == 'vector':
            return value
        if mode == 'constant':
            return value[0]
        return value.new_zeros(1)

    def adapted_projection(self, features, weight):
        index = self._path_rng.randrange(len(ARCHITECTURES)) if self.training and self.sample_paths else int(self.glora_architecture)
        A, B, C, D, E = [self.support(path, mode) for path, mode in zip(PATHS, ARCHITECTURES[index])]
        adapted_weight = weight + weight * A + B  # Hadamard scaling, not W @ A.
        adapted_bias = E if self.bias is None else self.bias + self.bias * D + E
        activation_prompt = (weight @ C).reshape(-1)
        return F.linear(features, adapted_weight, adapted_bias + activation_prompt)

    def forward(self, x, edge_index=None, bit_sum=None):
        weight = self.weight_quant(self.weight)
        if self.graph_level:
            if bit_sum is None:
                raise ValueError('Graph QLinear requires a feature-budget accumulator')
            if isinstance(self.fea_quant, nn.Identity):
                features, layer_budget = x, 0
            else:
                features, layer_budget = self.fea_quant(x, edge_index)
            bit_sum += layer_budget
            return self.adapted_projection(features, weight), edge_index, bit_sum
        return self.adapted_projection(self.fea_quant(x), weight)

    def active_parameters(self, architecture=None):
        index = int(self.glora_architecture) if architecture is None else architecture
        count = 0
        for path, mode in zip(PATHS, ARCHITECTURES[index]):
            rows = self.in_features if path == 'C' else self.out_features
            columns = 1 if path == 'C' else self.in_features
            if mode == 'lora':
                count += self.rank*(rows+columns)
            elif mode == 'vector':
                count += rows
            elif mode == 'constant':
                count += 1
        return count


class CompactGLoRAQLinear(GLoRAQLinear):
    """Physically discard unused supernet tensors without merging into W_q."""
    def __init__(self, source):
        nn.Module.__init__(self)
        _copy_base(self, source)
        self.rank, self.sample_paths = source.rank, False
        self.register_buffer('glora_architecture', source.glora_architecture.clone())
        self.glora_compact = nn.ParameterDict()
        for path, mode in zip(PATHS, ARCHITECTURES[int(self.glora_architecture)]):
            if mode == 'none':
                continue
            if mode == 'lora':
                for suffix in ('d', 'u'):
                    self.glora_compact[path+suffix] = nn.Parameter(getattr(source, f'glora_{path}{suffix}').detach().clone())
            else:
                self.glora_compact[path] = nn.Parameter(source.support(path, mode).detach().clone())

    def support(self, path, mode):
        if mode == 'none':
            shape = ((self.in_features, 1) if path == 'C' else
                     (self.out_features, self.in_features) if path in 'AB' else (1,))
            return self.weight.new_zeros(shape)
        if mode == 'lora':
            return self.glora_compact[path+'d'] @ self.glora_compact[path+'u']
        return self.glora_compact[path]

    def set_architecture(self, architecture):
        if architecture != int(self.glora_architecture):
            raise ValueError('A compact export cannot switch to an architecture whose tensors were discarded')


def layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, GLoRAQLinear)]


def install_glora(model, seed, rank=2):
    if layers(model):
        raise ValueError('GLoRA already installed')
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, (NodeLinear, GraphLinear, FixedGraphLinear))]
    if not targets or any(layer.weight.device.type != 'cpu' for _, layer in targets):
        raise ValueError('Install into supported QLinear layers on CPU before model.to(device)')
    rng = random.Random(seed + 910_000)
    model._glora_rng = rng
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 900_000)
        for name, module in targets:
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, GLoRAQLinear(module, rank, rng))
    return [name for name, _ in targets]


def build_glora(config, seed, data):
    model = build_baseline(config, 'no_prompt', seed, data)
    install_glora(model, seed, config.get('glora_rank', 2))
    return model


def backbone_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith('prompt_') or adapter_name(name):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_subnet(model, candidate):
    adapted = layers(model)
    if len(adapted) != len(candidate):
        raise ValueError('Subnet must specify every adapted linear layer')
    for (_, layer), architecture in zip(adapted, candidate):
        layer.set_architecture(architecture)


def subnet_parameters(model, candidate):
    adapted = layers(model)
    if len(adapted) != len(candidate):
        raise ValueError('Incomplete subnet')
    return sum(layer.active_parameters(index) for (_, layer), index in zip(adapted, candidate))


def compact_model(model):
    result = copy.deepcopy(model)
    for name, source in layers(result):
        if isinstance(source, CompactGLoRAQLinear):
            continue
        parent, _, child = name.rpartition('.')
        setattr(result.get_submodule(parent) if parent else result, child, CompactGLoRAQLinear(source))
    return result


def glora_optimizer(model, config):
    optimizer = optimizer_for(model, config)
    adapters = [p for name, p in model.named_parameters() if adapter_name(name) and p.requires_grad]
    identifiers = {id(p) for p in adapters}
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params'] if id(p) not in identifiers]
    optimizer.param_groups[:] = [group for group in optimizer.param_groups if group['params']]
    optimizer.add_param_group(dict(params=adapters, lr=config['lr'], weight_decay=0., name='glora'))
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters() if p.requires_grad}
    return optimizer


def model_counts(model):
    named=list(model.named_parameters())
    adapted=[p for name,p in named if adapter_name(name)]
    return dict(total_parameters=sum(p.numel() for _,p in named),
                trainable_parameters=sum(p.numel() for _,p in named if p.requires_grad),
                adapter_parameters=sum(p.numel() for p in adapted),
                adapter_fp32_bytes=sum(p.numel()*p.element_size() for p in adapted),
                external_prompt_parameters=sum(p.numel() for n,p in named if n.startswith('prompt_')),
                targets=[name for name,_ in layers(model)])

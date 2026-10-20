"""Locked zero-start + standardized conditions + uniform per-layer rank family.

The main variant matches StablePromptBank(spectral=False) exactly. Ablations
change one switch relative to it; historical experiment code remains unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops

from .improved import LayerView, StablePromptBank, calibrate
from .models import state_digest, sum_messages
from .prompts import make_prompt, quantization_context

VARIANTS = ("none", "original", "main", "nonzero_start", "raw_conditions",
            "no_conditions", "spectral_basis")
LABELS = {"none": "无聚合提示", "original": "原共享 LoRAP",
          "main": "主版本：零初始化＋标准化条件＋均匀 rank",
          "nonzero_start": "取消零初始化", "raw_conditions": "取消条件标准化",
          "no_conditions": "关闭量化条件", "spectral_basis": "改用误差谱基"}


class UniformLayerView(LayerView):
    def __init__(self, bank, index):
        super().__init__(bank, index)
        self.needs_context = bank.needs_context


class UniformPromptBank(StablePromptBank):
    def __init__(self, width, layers, calibration, seed, variant="main", rank=2, num_prompts=20):
        if variant not in VARIANTS[2:]:
            raise ValueError(variant)
        super().__init__(width, num_prompts, [rank]*layers, calibration, seed,
                         spectral=variant == "spectral_basis", residual_scale=1.)
        self.variant = variant
        self.needs_context = variant != "no_conditions"
        if variant == "no_conditions":
            # Remove the unused parameters; never pad the control's capacity.
            del self.condition
            del self.gate
            for index in range(layers):
                delattr(self, f"context_mean_{index}")
                delattr(self, f"context_scale_{index}")
        elif variant == "nonzero_start":
            for index, factors in enumerate(self.factors):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + 700_000 + index)
                    nn.init.xavier_uniform_(factors.P_left)

    def for_layer(self, index):
        return UniformLayerView(self, index)

    def add_at(self, index, features, context):
        if self.needs_context:
            if context is None or context.shape != (features.size(0), 4):
                raise ValueError("Expected quantization context [nodes, 4]")
            if self.variant == "raw_conditions":
                z = context.detach()
            else:
                z = ((context.detach() - getattr(self, f"context_mean_{index}")) /
                     getattr(self, f"context_scale_{index}")).clamp(-5, 5)
        rms = features.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        score = self.a(features / rms)
        if self.needs_context:
            score = score + self.condition(z)
        factors = self.factors[index]
        delta = F.softmax(score, dim=-1) @ factors.P_left @ factors.P_right
        amplitude = rms.detach().clamp_min(getattr(self, f"rms_floor_{index}"))
        if self.needs_context:
            return features + self.residual_scale * amplitude * 2 * torch.sigmoid(self.gate(z)) * delta
        return features + self.residual_scale * amplitude * delta


def install_uniform(model, dataset, seed, calibration, variant="main"):
    width, layers = (16, 1) if dataset == "CiteSeer" else (110, 4)
    if variant == "none":
        from .mechanism import NoPrompt
        model.prompt_aggr = NoPrompt()
    elif variant == "original":
        model.prompt_aggr = make_prompt("original", width, 20, 2, seed)
    else:
        model.prompt_aggr = UniformPromptBank(width, layers, calibration, seed, variant)
    return [2]*layers if variant != "none" else []


@torch.no_grad()
def calibrate_uniform(model, dataset, data, seed, include_spectral=False,
                      max_nodes=4096, max_graphs=128):
    if include_spectral:
        return calibrate(model, dataset, data, seed, max_nodes, max_graphs)
    # The main method needs only context and aggregate RMS. Avoid all FP
    # reference aggregation and SVD when the spectral-basis control is absent.
    if max_nodes < 1 or max_graphs < 1:
        raise ValueError("Calibration limits must be positive")
    layers = [model.conv2] if dataset == "CiteSeer" else list(model.convs)
    samples, handles = [{} for _ in layers], []
    generators = [torch.Generator().manual_seed(seed + 600_000 + i) for i in range(len(layers))]
    digest = state_digest(model, "all")
    modes = {module: module.training for module in model.modules()}

    def capture(index):
        def hook(layer, args):
            x, edges = args[:2]
            edges, _ = remove_self_loops(edges)
            q = layer.q_xw(x.T).T
            aggregate = sum_messages(q, edges)
            context = quantization_context(q, edges, layer.q_xw)
            mask = data.train_mask if dataset == "CiteSeer" else slice(None)
            values = dict(context=context[mask].cpu(), aggregate=aggregate[mask].cpu())
            keys = torch.rand(len(values["context"]), generator=generators[index])
            state = samples[index]
            state["seen"] = state.get("seen", 0) + len(keys)
            if "keys" in state:
                keys = torch.cat((state["keys"], keys))
                values = {name: torch.cat((state[name], value)) for name, value in values.items()}
            select = keys.topk(min(max_nodes, len(keys))).indices
            state.update({name: value[select] for name, value in values.items()})
            state["keys"] = keys[select]
        return hook

    try:
        model.eval()
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(capture(index)))
        batches = ([data] if dataset == "CiteSeer" else
                   DataLoader(data["train"][:min(max_graphs, len(data["train"]))], batch_size=128,
                              shuffle=False, generator=torch.Generator().manual_seed(seed + 610_000)))
        with torch.random.fork_rng(devices=[]):
            for batch in batches:
                model(batch)
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in modes.items():
            module.training = mode
    assert state_digest(model, "all") == digest
    output = []
    for index, sample in enumerate(samples):
        context = sample["context"].double()
        output.append(dict(layer=f"layer_{index + 2 if dataset == 'CiteSeer' else index + 1}",
            seen_nodes=sample["seen"], sampled_nodes=len(context),
            context_mean=context.mean(0),
            context_scale=torch.maximum(context.std(0, unbiased=False), context.new_tensor([.5, 1., .125, .1])),
            rms_floor=sample["aggregate"].square().mean(-1).sqrt().mean().mul(.1).clamp_min(1e-4),
            split="train", calibration_seed=seed, fp_reference_used=False))
    return output

"""Weight LoRA on the existing QAT inputs; the backbone remains trainable.

This is a joint-QAT control, not frozen-backbone PEFT or GraphLoRA/GLoRA.
The two branches share dequantized features and never merge their weights.
"""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .full_config import full_config
from .full_models import build_full_backbone
from .mechanism import NoPrompt
from .models import GraphLinear, NodeLinear, optimizer_for
from .uniform import install_uniform

DATASETS = ("Cora", "CiteSeer", "MNIST")
VARIANTS = ("no_prompt", "main", "lora")


def validate_lora(rank, alpha):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")


class LoRAQLinear(nn.Module):
    """Take ownership of a QLinear's parameters without reinitializing them."""

    def __init__(self, base, rank=2, alpha=2.):
        super().__init__()
        validate_lora(rank, alpha)
        if base.weight.dtype != torch.float32:
            raise ValueError("The joint-QAT baseline requires an FP32 backbone")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.weight_quant, self.fea_quant = base.weight_quant, base.fea_quant
        self.bit = base.bit
        # QuantizationObserver identifies quantizers by .bit and .alpha/.gama.
        # Keep adapter metadata distinct from quantizer scale attributes.
        self.rank, self.lora_alpha, self.scaling = rank, float(alpha), float(alpha) / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.training = base.training

    def project(self, features, weight):
        delta = F.linear(F.linear(features, self.lora_A), self.lora_B)
        return F.linear(features, weight, self.bias) + self.scaling * delta


class NodeLoRAQLinear(LoRAQLinear):
    def forward(self, x):
        weight = self.weight_quant(self.weight)
        features = self.fea_quant(x)
        return self.project(features, weight)


class GraphLoRAQLinear(LoRAQLinear):
    def forward(self, x, edge_index, bit_sum):
        weight = self.weight_quant(self.weight)
        if isinstance(self.fea_quant, nn.Identity):
            features, layer_budget = self.fea_quant(x), 0
        else:
            features, layer_budget = self.fea_quant(x, edge_index)
        bit_sum += layer_budget
        return self.project(features, weight), edge_index, bit_sum


def install_lora(model, seed, rank=2, alpha=2.):
    validate_lora(rank, alpha)
    if any(isinstance(module, LoRAQLinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, (NodeLinear, GraphLinear))]
    if not targets:
        raise ValueError("No supported quantized linear layers found")
    # Installation happens on CPU before model.to(device); no global RNG drift.
    if any(layer.weight.device.type != "cpu" for _, layer in targets):
        raise ValueError("Install LoRA before moving the model to CUDA")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 800_000)
        for name, layer in targets:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            cls = GraphLoRAQLinear if isinstance(layer, GraphLinear) else NodeLoRAQLinear
            setattr(parent, child, cls(layer, rank, alpha))
    return [name for name, _ in targets]


def common_backbone_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("prompt_") or name.rsplit(".", 1)[-1] in ("lora_A", "lora_B"):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def baseline_config(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    config = full_config(dataset)
    config.update(profile="lora_joint_qat", variants=list(VARIANTS), seeds=[41, 42, 43],
                  lora_rank=2, lora_alpha=2., lora_dropout=0., device="cuda:0",
                  diagnostics=False, epochs=30 if dataset == "MNIST" else 200,
                  subset_sizes=dict(train=512, val=256, test=256) if dataset == "MNIST" else None,
                  method_note="Joint QAT; lora/no_prompt remove all prompts; main keeps input GPF")
    return config


def build_baseline(config, variant, seed, data, calibration=None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    model = build_full_backbone(config, seed, data)
    if variant == "main":
        if calibration is None:
            raise ValueError("main requires training-only calibration")
        install_uniform(model, config["family"], seed, calibration, "main")
    else:
        model.prompt_aggr = NoPrompt()
        if hasattr(model, "prompt_node"):
            model.prompt_node = NoPrompt()
        if variant == "lora":
            install_lora(model, seed, config["lora_rank"], config["lora_alpha"])
    return model


def baseline_counts(model):
    named = list(model.named_parameters())
    lora = [p for n, p in named if n.rsplit(".", 1)[-1] in ("lora_A", "lora_B")]
    prompts = [p for n, p in named if n.startswith("prompt_")]
    return dict(total_parameters=sum(p.numel() for _, p in named),
                trainable_parameters=sum(p.numel() for _, p in named if p.requires_grad),
                lora_parameters=sum(p.numel() for p in lora),
                prompt_parameters=sum(p.numel() for p in prompts),
                lora_fp32_bytes=sum(p.numel() * p.element_size() for p in lora),
                prompt_fp32_bytes=sum(p.numel() * p.element_size() for p in prompts),
                prompt_buffer_bytes=sum(b.numel() * b.element_size() for n, b in model.named_buffers()
                                        if n.startswith("prompt_")),
                lora_targets=[name for name, m in model.named_modules() if isinstance(m, LoRAQLinear)])


def baseline_optimizer(model, config):
    optimizer = optimizer_for(model, config)
    lora = [p for m in model.modules() if isinstance(m, LoRAQLinear) for p in (m.lora_A, m.lora_B)]
    lora_ids = {id(p) for p in lora}
    if lora:
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in lora_ids]
        optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
        optimizer.add_param_group(dict(params=lora, lr=config["lr"], weight_decay=0., name="lora"))
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError("Optimizer parameters must cover every trainable tensor exactly once")
    return optimizer

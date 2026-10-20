"""Aggregation prompts and inference-available quantization conditions."""

import torch
from torch import nn
from torch.nn import functional as F
from prompt.GPF import GPF_plus_LoRA


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


class LoRAP(GPF_plus_LoRA):
    needs_context = False

    def add(self, features, context=None):
        return super().add(features)


class ConditionedLoRAP(LoRAP):
    needs_context = True

    def __init__(self, in_channels, p_num, rank):
        super().__init__(in_channels, p_num, rank)
        self.condition = nn.Linear(4, p_num, bias=False)
        self.gate = nn.Linear(4, 1)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def add(self, features, context=None):
        if context is None or context.shape != (features.shape[0], 4):
            raise ValueError("ConditionedLoRAP requires context with shape [nodes, 4]")
        context = context.detach()
        weights = F.softmax(self.a(features) + self.condition(context), dim=1)
        delta = weights @ (self.P_left @ self.P_right)
        return features + 2 * torch.sigmoid(self.gate(context)) * delta


class ResidualMLP(nn.Module):
    needs_context = False

    def __init__(self, in_channels, target_parameters):
        super().__init__()
        # Two biased linear layers: h(2d + 1) + d parameters.
        estimate = (target_parameters - in_channels) / (2 * in_channels + 1)
        candidates = {max(1, int(estimate)), max(1, int(estimate) + 1)}
        self.hidden_width = min(candidates, key=lambda h: (
            abs(h * (2 * in_channels + 1) + in_channels - target_parameters), h))
        self.net = nn.Sequential(nn.Linear(in_channels, self.hidden_width), nn.ReLU(),
                                 nn.Linear(self.hidden_width, in_channels))
        self.target_parameters = target_parameters
        self.relative_parameter_difference = abs(parameter_count(self) - target_parameters) / target_parameters
        if self.relative_parameter_difference > 0.05:
            raise ValueError("No residual MLP width matches the prompt budget within 5%")

    def add(self, features, context=None):
        return features + self.net(features)


@torch.no_grad()
def quantization_context(quantized_features, edge_index, quantizer):
    """Use the quantizer directly before sum aggregation, never FP references.

    q_xw has one scale/bit parameter per channel, shared across source nodes.
    Saturation is measured from dequantized integer codes at their endpoints.
    All edge statistics use the exact self-loop-free message edge list.
    """
    x = quantized_features.detach()
    source, target = edge_index
    n = x.shape[0]
    degree = x.new_zeros(n).index_add_(0, target, x.new_ones(target.numel()))
    scale = quantizer.alpha.detach().abs().flatten().to(x)
    bits = quantizer.bit.detach().abs().round().flatten().to(x)
    qmax = 2 ** (bits - 1) - 1
    threshold = qmax * scale
    saturation = torch.isclose(x.abs(), threshold[None, :], rtol=1e-5, atol=1e-8).float().mean(1)
    neighbor_saturation = x.new_zeros(n).index_add_(0, target, saturation[source]) / degree.clamp_min(1)
    return torch.stack((degree.log1p(),
                        scale.clamp_min(1e-12).log().mean().expand(n),
                        (bits.mean() / 8).expand(n), neighbor_saturation), dim=1).detach()


def make_prompt(variant, width, num_prompts, rank, seed):
    # A separate RNG stream gives original/conditioned exactly the same bases
    # without perturbing backbone initialization or training/dropout RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 100_000)
        if variant == "original":
            return LoRAP(width, num_prompts, rank)
        if variant == "conditioned":
            return ConditionedLoRAP(width, num_prompts, rank)
        if variant == "mlp":
            target = parameter_count(ConditionedLoRAP(width, num_prompts, rank))
            return ResidualMLP(width, target)
    raise ValueError(f"Unknown variant: {variant}")


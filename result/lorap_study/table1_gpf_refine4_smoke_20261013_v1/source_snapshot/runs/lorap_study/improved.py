"""Zero-start, normalized quantization conditioning and layerwise spectral rank.

Calibration uses training nodes/graphs only. FP message references are used once
for calibration and in optional diagnostics, never in deployed prompt inference.
Historical models, prompts and quantizers are unchanged.
"""

import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_self_loops

from .mechanism import NoPrompt
from .models import build_model, sum_messages, state_digest
from .prompts import ConditionedLoRAP, make_prompt, quantization_context


class LayerView:
    """Unregistered view; the owning bank registers each parameter exactly once."""
    needs_context = True

    def __init__(self, bank, index):
        self.bank, self.index = bank, index

    def add(self, features, context=None):
        return self.bank.add_at(self.index, features, context)


class LowRankFactors(nn.Module):
    def __init__(self, width, num_prompts, rank, basis):
        super().__init__()
        self.P_left = nn.Parameter(torch.zeros(num_prompts, rank))
        self.P_right = nn.Parameter(basis[:rank].float().clone())


class StablePromptBank(nn.Module):
    needs_context = True

    def __init__(self, width, num_prompts, ranks, calibration, seed, spectral=True,
                 residual_scale=1.0):
        super().__init__()
        self.ranks = list(ranks)
        self.residual_scale = residual_scale
        # Share score, condition projection and gate. Capacity differences between
        # uniform/adaptive banks come only from sum(ranks), fixed by construction.
        template = make_prompt("conditioned", width, num_prompts, 2, seed)
        self.a, self.condition, self.gate = template.a, template.condition, template.gate
        self.factors = nn.ModuleList()
        for index, rank in enumerate(ranks):
            info = calibration[index]
            if spectral:
                basis = info["basis"]
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + 500_000 + index)
                    basis = torch.linalg.qr(torch.randn(width, max(ranks)), mode="reduced").Q.T
            self.factors.append(LowRankFactors(width, num_prompts, rank, basis))
            self.register_buffer(f"context_mean_{index}", info["context_mean"].float().clone())
            self.register_buffer(f"context_scale_{index}", info["context_scale"].float().clone())
            self.register_buffer(f"rms_floor_{index}", info["rms_floor"].float().clone())

    def for_layer(self, index):
        return LayerView(self, index)

    def add(self, features, context=None):
        if len(self.ranks) != 1:
            raise ValueError("Multi-layer bank requires for_layer(index)")
        return self.add_at(0, features, context)

    def add_at(self, index, features, context):
        if context is None or context.shape != (features.size(0), 4):
            raise ValueError("Expected quantization context [nodes, 4]")
        mean = getattr(self, f"context_mean_{index}")
        scale = getattr(self, f"context_scale_{index}")
        # Finite, frozen training calibration; no labels or FP references here.
        z = ((context.detach() - mean) / scale).clamp(-5, 5)
        rms = features.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = features / rms
        weights = F.softmax(self.a(normalized) + self.condition(z), dim=-1)
        factors = self.factors[index]
        delta = weights @ factors.P_left @ factors.P_right
        amplitude = rms.detach().clamp_min(getattr(self, f"rms_floor_{index}"))
        return features + self.residual_scale * amplitude * 2 * torch.sigmoid(self.gate(z)) * delta


class LayerwiseMNIST(nn.Module):
    def __init__(self, base):
        super().__init__()
        for name in ("embedding", "convs", "lin1", "lin2", "prompt_node", "prompt_aggr"):
            setattr(self, name, getattr(base, name))

    def forward(self, data, diagnostics=None, node_mask=None):
        x = self.prompt_node.add(torch.cat((data.x, data.pos), dim=1))
        x, _, budget = self.embedding(x, data.edge_index, x.new_zeros(1))
        for index, conv in enumerate(self.convs):
            prompt = (self.prompt_aggr.for_layer(index) if isinstance(self.prompt_aggr, StablePromptBank)
                      else self.prompt_aggr)
            x, budget = conv(x, data.edge_index, budget, prompt,
                             diagnostics, f"layer_{index + 1}", node_mask)
        x = global_mean_pool(x, data.batch)
        x, _, budget = self.lin1(x, data.edge_index, budget)
        x, _, budget = self.lin2(F.relu(x), data.edge_index, budget)
        return x, budget.squeeze()


def bare_model(dataset, seed, data=None):
    base = build_model(dataset, "original", seed, data=data)
    base.prompt_aggr = NoPrompt()
    return LayerwiseMNIST(base) if dataset == "MNIST" else base


def canonical_basis(error):
    _, singular, vh = torch.linalg.svd(error.double(), full_matrices=False)
    # Fix the arbitrary sign so snapshots are easier to compare.
    largest = vh.abs().argmax(dim=1)
    sign = vh[torch.arange(len(vh)), largest].sign()
    vh = vh * torch.where(sign == 0, torch.ones_like(sign), sign)[:, None]
    return singular, vh


@torch.no_grad()
def calibrate(model, dataset, data, seed=41, max_nodes=4096, max_graphs=128):
    """Collect the full local GIN input residual; retain at most max_nodes/layer."""
    if max_nodes < 1 or max_graphs < 1:
        raise ValueError("Calibration limits must be positive")
    layers = [model.conv2] if dataset == "CiteSeer" else list(model.convs)
    samples, handles = [{} for _ in layers], []
    generators = [torch.Generator().manual_seed(seed + 600_000 + i) for i in range(len(layers))]
    original_state = state_digest(model, "all")
    was_training = model.training

    def capture(index):
        def hook(layer, args):
            x, edge_index = args[:2]
            edges, _ = remove_self_loops(edge_index)
            q = layer.q_xw(x.T).T
            sq, sf = sum_messages(q, edges), sum_messages(x, edges)
            reference = (1 + layer.eps) * x + sf
            error = reference - ((1 + layer.eps) * q + sq)
            context = quantization_context(q, edges, layer.q_xw)
            mask = data.train_mask if dataset == "CiteSeer" else slice(None)
            values = {"error": error[mask].cpu(), "reference": reference[mask].cpu(),
                      "context": context[mask].cpu(), "aggregate": sq[mask].cpu()}
            keys = torch.rand(values["error"].size(0), generator=generators[index])
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
                   DataLoader(data["train"][:min(max_graphs, len(data["train"]))],
                              batch_size=128, shuffle=False,
                              generator=torch.Generator().manual_seed(seed + 610_000)))
        with torch.random.fork_rng(devices=[]):
            for batch in batches:
                model(batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    assert state_digest(model, "all") == original_state, "Calibration mutated model parameters/buffers"
    result = []
    for index, state in enumerate(samples):
        error = state["error"].double()
        singular, basis = canonical_basis(error)
        reference_energy = float(state["reference"].double().square().sum())
        context = state["context"].double()
        floors = context.new_tensor([.5, 1., .125, .1])
        normalized_energy = singular.square() / max(reference_energy, 1e-20)
        centered = error - error.mean(0, keepdim=True)
        centered_singular = torch.linalg.svdvals(centered)
        result.append(dict(layer=f"layer_{index + 2 if dataset == 'CiteSeer' else index + 1}",
                           seen_nodes=state["seen"], sampled_nodes=len(error),
                           context_mean=context.mean(0),
                           context_scale=torch.maximum(context.std(0, unbiased=False), floors),
                           rms_floor=state["aggregate"].square().mean(-1).sqrt().mean().mul(.1).clamp_min(1e-4),
                           singular_values=singular, centered_singular_values=centered_singular,
                           normalized_energy=normalized_energy, basis=basis,
                           reference_energy=reference_energy,
                           reference="same input/weights; message quantizer bypassed, including GIN self term",
                           split="train", calibration_seed=seed))
    return result


def allocate_ranks(calibration, total_rank, max_rank=8):
    """Greedy normalized error-energy allocation; same channel width per layer.

    Keep one direction/layer; each extra direction has identical parameter cost.
    Equal gains prefer the lower current rank, then the earlier layer.
    """
    count = len(calibration)
    caps = [min(max_rank, len(info["normalized_energy"])) for info in calibration]
    if any(cap < 1 for cap in caps) or not count <= total_rank <= sum(caps):
        raise ValueError("Infeasible rank budget")
    ranks = [1] * count
    while sum(ranks) < total_rank:
        candidates = [i for i in range(count) if ranks[i] < caps[i]]
        selected = max(candidates, key=lambda i: (float(calibration[i]["normalized_energy"][ranks[i]]),
                                                   -ranks[i], -i))
        ranks[selected] += 1
    return ranks


def install_prompt(model, dataset, variant, seed, calibration):
    width, layers = (16, 1) if dataset == "CiteSeer" else (110, 4)
    if variant in ("original", "conditioned"):
        model.prompt_aggr = make_prompt(variant, width, 20, 2, seed)
        return [2] * layers
    if variant == "zero_init":
        model.prompt_aggr = make_prompt("conditioned", width, 20, 2, seed)
        with torch.no_grad():
            model.prompt_aggr.P_left.zero_()
        return [2] * layers
    if variant not in ("stable_random", "stable_spectral", "adaptive_spectral"):
        raise ValueError(variant)
    ranks = allocate_ranks(calibration, 2 * layers) if variant == "adaptive_spectral" else [2] * layers
    model.prompt_aggr = StablePromptBank(width, 20, ranks, calibration, seed,
                                        spectral=variant != "stable_random")
    return ranks


def calibration_json(calibration):
    return [{name: value.tolist() if torch.is_tensor(value) else value
             for name, value in info.items() if name != "basis"} for info in calibration]


class FullUpdateDiagnostics:
    """Optional validation diagnostics for both neighbor and full GIN errors."""
    def __init__(self, model, mask=None):
        self.model, self.mask, self.handles, self.rows = model, mask, [], {}

    def __enter__(self):
        from .mechanism import vector_metrics
        layers = [self.model.conv2] if hasattr(self.model, "conv2") else list(self.model.convs)

        def hook(name, index):
            @torch.no_grad()
            def capture(layer, args):
                x, edge_index = args[:2]
                edges, _ = remove_self_loops(edge_index)
                q = layer.q_xw(x.T).T
                sq, sf = sum_messages(q, edges), sum_messages(x, edges)
                bank = self.model.prompt_aggr
                prompt = bank.for_layer(index) if isinstance(bank, StablePromptBank) else bank
                context = quantization_context(q, edges, layer.q_xw) if prompt.needs_context else None
                delta = prompt.add(sq, context) - sq
                mask = self.mask if self.mask is not None else slice(None)
                self.rows.setdefault(name, []).append(dict(
                    neighbor=vector_metrics((sf - sq)[mask], delta[mask]),
                    full_update=vector_metrics(((1 + layer.eps)*(x - q) + sf - sq)[mask], delta[mask])))
            return capture
        for index, layer in enumerate(layers):
            name = f"layer_{index + 2 if hasattr(self.model, 'conv2') else index + 1}"
            self.handles.append(layer.register_forward_pre_hook(hook(name, index)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

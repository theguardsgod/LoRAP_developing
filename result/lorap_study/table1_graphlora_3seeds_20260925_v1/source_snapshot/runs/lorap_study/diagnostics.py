"""Validation-only local aggregation error and logical bit-use accounting."""

import hashlib
import math
import torch
from quantize_function.get_scale_index import get_scale_index_uniform


class AggregationDiagnostics:
    def __init__(self, seed=41, max_nodes=4096):
        self.seed, self.max_nodes = seed, max_nodes
        self.layers = {}

    @torch.no_grad()
    def add(self, name, reference, quantized, corrected, mask=None):
        if mask is not None:
            reference, quantized, corrected = reference[mask], quantized[mask], corrected[mask]
        error = (reference - quantized).detach().cpu().double()
        delta = (corrected - quantized).detach().cpu().double()
        residual = error - delta
        if not all(torch.isfinite(t).all() for t in (error, delta, residual)):
            raise FloatingPointError(f"Non-finite diagnostic tensor in {name}")
        if name not in self.layers:
            generator = torch.Generator().manual_seed(
                self.seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
            self.layers[name] = dict(nodes=0, values=0, before=0., after=0., delta_sq=0.,
                                     dot=0., cosine_sum=0., cosine_count=0, generator=generator,
                                     sample=error[:0], keys=torch.empty(0, dtype=torch.float64))
        state = self.layers[name]
        state["nodes"] += error.shape[0]
        state["values"] += error.numel()
        state["before"] += error.square().sum().item()
        state["after"] += residual.square().sum().item()
        state["delta_sq"] += delta.square().sum().item()
        state["dot"] += (error * delta).sum().item()
        denominator = error.norm(dim=1) * delta.norm(dim=1)
        valid = denominator > 1e-12
        state["cosine_sum"] += ((error * delta).sum(1)[valid] / denominator[valid]).sum().item()
        state["cosine_count"] += int(valid.sum())
        # Random-priority reservoir: uniform sampling across all validation
        # batches, bounded memory, identical selected node positions per variant.
        keys = torch.rand(error.shape[0], generator=state["generator"], dtype=torch.float64)
        keys = torch.cat((state["keys"], keys))
        rows = torch.cat((state["sample"], error))
        indices = keys.topk(min(self.max_nodes, keys.numel())).indices
        state["keys"], state["sample"] = keys[indices], rows[indices]

    def result(self):
        result = {}
        for name, state in self.layers.items():
            singular = torch.linalg.svdvals(state["sample"])
            energy = singular.square()
            total_energy = float(energy.sum())
            cumulative = energy.cumsum(0) / total_energy if total_energy > 0 else torch.zeros_like(energy)
            before, after = math.sqrt(state["before"]), math.sqrt(state["after"])
            denom = math.sqrt(state["before"] * state["delta_sq"])
            result[name] = dict(
                nodes=state["nodes"], sampled_nodes=state["sample"].shape[0],
                error_before_frobenius=before, error_after_frobenius=after,
                mse_before=state["before"] / max(state["values"], 1),
                mse_after=state["after"] / max(state["values"], 1),
                relative_error_reduction=1 - after / before if before > 1e-12 else None,
                global_cosine=state["dot"] / denom if denom > 1e-12 else None,
                mean_node_cosine=state["cosine_sum"] / state["cosine_count"] if state["cosine_count"] else None,
                valid_cosine_nodes=state["cosine_count"],
                singular_values=singular.tolist(), cumulative_energy=cumulative.tolist(),
                rank_90=int(torch.searchsorted(cumulative, 0.90)) + 1 if total_energy > 0 else 0,
                rank_95=int(torch.searchsorted(cumulative, 0.95)) + 1 if total_energy > 0 else 0,
                energy_at_rank_2=float(cumulative[min(1, len(cumulative) - 1)]) if len(cumulative) else 0.)
        return dict(reference="same layer input, message quantizer bypassed only; not an FP32 teacher",
                    split="validation", seed=self.seed, max_nodes=self.max_nodes, layers=result)


class QuantizationObserver:
    """Actual rounded bits at visited quantizers, not physical QAT memory.

    The graph-level legacy budget sums unique parameter slots, not node storage.
    Record both that training proxy (in the runner) and real tensor assignments.
    A signed surrogate used for nonnegative activations has one additional bit;
    report its raw quantizer bits and effective nonnegative bits separately.
    """
    def __init__(self, model):
        self.model, self.handles, self.stats = model, [], {}

    def __enter__(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "bit") and (hasattr(module, "gama") or hasattr(module, "alpha")):
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def _hook(self, name):
        @torch.no_grad()
        def observe(module, inputs):
            tensor = inputs[0]
            bits = module.bit.detach().abs().round()
            if hasattr(module, "uniform"):
                if not module.uniform:
                    raise ValueError("Study graph models must use the configured uniform assignment")
                index = get_scale_index_uniform(tensor, inputs[1], module.gama, module.bit)
                bits = bits[index]
            if bits.shape[0] != tensor.shape[0]:
                raise ValueError(f"Bit assignment shape mismatch in {name}")
            unsigned = int(getattr(module, "study_unsigned", False))
            values, counts = torch.unique(bits, return_counts=True)
            elements_per_row = tensor.numel() // tensor.shape[0]
            state = self.stats.setdefault(name, dict(kind="weight" if "weight_quant" in name else "activation",
                                                     elements=0, signed_bit_sum=0., effective_bit_sum=0., histogram={}))
            state["elements"] += tensor.numel()
            state["signed_bit_sum"] += float(bits.sum()) * elements_per_row
            state["effective_bit_sum"] += float((bits - unsigned).sum()) * elements_per_row
            for bit, count in zip(values.tolist(), counts.tolist()):
                key = str(int(bit) - unsigned)
                state["histogram"][key] = state["histogram"].get(key, 0) + count * elements_per_row
        return observe

    def result(self):
        activations = [s for s in self.stats.values() if s["kind"] == "activation"]
        total_values = sum(s["elements"] for s in activations)
        total_bits = sum(s["effective_bit_sum"] for s in activations)
        layers = {}
        for name, state in self.stats.items():
            layers[name] = dict(**state,
                               mean_quantizer_bits=state["signed_bit_sum"] / state["elements"],
                               mean_effective_bits=state["effective_bit_sum"] / state["elements"])
        return dict(layers=layers,
                    mean_activation_effective_bits=total_bits / total_values if total_values else None,
                    logical_activation_bytes_over_validation=total_bits / 8,
                    note="Logical tensor traffic across observed quantizers; excludes indices/scales, is not peak memory or integer execution. CiteSeer propagation visits the full public graph.")

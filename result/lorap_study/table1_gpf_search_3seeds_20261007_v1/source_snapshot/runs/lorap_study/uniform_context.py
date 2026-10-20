"""Validation-only condition drift audit; observes the existing prompt inputs."""

from unittest.mock import patch

import torch

from .models import state_digest
from .run import evaluate
from .uniform import UniformPromptBank


@torch.no_grad()
def inspect_conditions(model, validation, config):
    bank = model.prompt_aggr
    if not isinstance(bank, UniformPromptBank) or not bank.needs_context:
        return None
    original = bank.add_at
    states = {}
    before = state_digest(model, "all")

    def capture(index, features, context):
        output = original(index, features, context)
        # These are the actual inputs already computed by the normal forward.
        mask = validation.val_mask if config["dataset"] == "CiteSeer" else slice(None)
        raw = context.detach()[mask]
        observed, features = output[mask], features[mask]
        standardized = ((raw-getattr(bank, f"context_mean_{index}")) /
                        getattr(bank, f"context_scale_{index}"))
        actual = raw if bank.variant == "raw_conditions" else standardized.clamp(-5, 5)
        gate = 2*torch.sigmoid(bank.gate(actual.to(features.dtype)))
        name = f"layer_{index+2 if config['dataset']=='CiteSeer' else index+1}"
        state = states.setdefault(name, dict(nodes=0, raw_sum=torch.zeros(4, dtype=torch.double),
            standardized_abs_sum=torch.zeros(4, dtype=torch.double), clipped=torch.zeros(4, dtype=torch.double),
            gate_sum=0., gate_min=float("inf"), gate_max=-float("inf"), prompt_sq=0., aggregate_sq=0.))
        state["nodes"] += len(raw)
        state["raw_sum"] += raw.double().sum(0).cpu()
        state["standardized_abs_sum"] += standardized.double().abs().sum(0).cpu()
        state["clipped"] += (standardized.abs()>5).sum(0).cpu()
        state["gate_sum"] += float(gate.double().sum())
        state["gate_min"] = min(state["gate_min"], float(gate.min()))
        state["gate_max"] = max(state["gate_max"], float(gate.max()))
        state["prompt_sq"] += float((observed-features).double().square().sum())
        state["aggregate_sq"] += float(features.double().square().sum())
        return output

    with patch.object(bank, "add_at", side_effect=capture):
        metrics = evaluate(model, validation, config, torch.device("cpu"), "val")
    assert state_digest(model, "all") == before
    layers = {}
    for name, state in states.items():
        count = state["nodes"]
        layers[name] = dict(nodes=count, raw_mean=(state["raw_sum"]/count).tolist(),
            standardized_mean_absolute=(state["standardized_abs_sum"]/count).tolist(),
            potential_clipped_fraction=(state["clipped"]/count).tolist(),
            gate_mean=state["gate_sum"]/count, gate_min=state["gate_min"], gate_max=state["gate_max"],
            prompt_to_aggregate_norm=(state["prompt_sq"]/state["aggregate_sq"])**.5 if state["aggregate_sq"] else None)
    return dict(split="validation graphs or validation node mask",
                fields=["log1p_degree", "mean_log_step", "effective_bits_over_8", "neighbor_saturation"],
                note="Observed existing prompt inputs; no reference branch or extra quantization. Potential clipping is hypothetical for raw_conditions.",
                validation=metrics, layers=layers)

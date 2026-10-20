"""Paired CiteSeer capacity/quantization controls and validation interventions.

python -m runs.lorap_study.mechanism --output-dir result/lorap_study/mechanism_20260909_v1
"""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import MethodType

import torch
from torch import nn
from torch_geometric.utils import remove_self_loops

from .config import dataset_config
from .data import load_data
from .diagnostics import QuantizationObserver
from .models import build_model, model_counts, optimizer_for, state_digest, sum_messages
from .prompts import ResidualMLP, parameter_count
from .run import (ROOT, BestCheckpoint, evaluate, provenance, seed_training,
                  train_epoch, write_json)

PRECISIONS = ("qat", "fp32")
CONTROLS = ("none", "original", "mlp_exact")
INTERVENTIONS = ("learned", "off", "signflip", "shuffle", "mean",
                 "parallel", "orthogonal")


class NoPrompt(nn.Module):
    needs_context = False

    def add(self, features, context=None):
        return features


def quantizer_leaves(model):
    # CiteSeer leaf quantizers all return tensors. QLinear wrappers have .bit
    # but neither .alpha nor .gama and must not be replaced.
    return [(name, module) for name, module in model.named_modules()
            if hasattr(module, "bit") and
            (hasattr(module, "alpha") or hasattr(module, "gama"))]


@contextmanager
def replace_forward(module, function):
    had_local = "forward" in module.__dict__
    previous = module.__dict__.get("forward")
    module.forward = MethodType(function, module)
    try:
        yield
    finally:
        if had_local:
            module.forward = previous
        else:
            del module.forward


@contextmanager
def bypass_quantizers(model):
    """Identity leaves: no quantizer forward/backward computation is executed."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        for _, module in quantizer_leaves(model):
            stack.enter_context(replace_forward(module, lambda self, x: x))
        yield


def make_control(data, seed, precision, control):
    if precision not in PRECISIONS or control not in CONTROLS:
        raise ValueError((precision, control))
    model = build_model("CiteSeer", "original", seed, data=data)
    if control == "none":
        model.prompt_aggr = NoPrompt()
    elif control == "mlp_exact":
        target = parameter_count(model.prompt_aggr)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100_000)
            model.prompt_aggr = ResidualMLP(16, target)
        assert parameter_count(model.prompt_aggr) == target == 412
    if precision == "fp32":
        for _, module in quantizer_leaves(model):
            module.requires_grad_(False)
    return model


def precision_context(model, precision):
    return bypass_quantizers(model) if precision == "fp32" else nullcontext()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_control(config, data, seed, precision, control, directory, reuse_pilot=None):
    directory.mkdir(parents=True, exist_ok=False)
    effective = dict(config, precision=precision, control=control, seed=seed, device="cpu")
    if precision == "fp32":
        effective["a_loss"] = 0.0
    model = make_control(data, seed, precision, control)
    initial = dict(backbone_sha256=state_digest(model),
                   prompt_base_sha256=state_digest(model, "prompt_base"),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   **model_counts(model))
    write_json(directory / "config.json", effective)
    write_json(directory / "initialization.json", initial)
    device = torch.device("cpu")
    if reuse_pilot is not None and precision == "qat" and control == "original":
        source = reuse_pilot / "CiteSeer" / "original" / f"seed_{seed}"
        old_config = json.loads((source / "config.json").read_text())
        for key in config:
            if key not in ("profile", "diagnostics") and config[key] != old_config[key]:
                raise ValueError(f"Cannot reuse pilot: config mismatch {key}")
        old = json.loads((source / "result.json").read_text())
        for key in ("backbone_sha256", "prompt_base_sha256"):
            assert old[key] == initial[key], key
        for filename in ("best.pt", "epochs.jsonl", "quantization.json"):
            shutil.copy2(source / filename, directory / filename)
        selected = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
        model.load_state_dict(selected["model"])
        validation = evaluate(model, data, effective, device, "val")
        assert abs(validation["loss"] - old["validation"]["loss"]) < 1e-6
        result = dict(old, **{k: initial[k] for k in initial}, precision=precision, control=control,
                      reused_from=str(source), source_checkpoint_sha256=file_digest(source / "best.pt"),
                      new_test_evaluations=0)
    else:
        optimizer = optimizer_for(model, effective)
        checkpoint = BestCheckpoint(directory / "best.pt")
        seed_training(seed + 300_000)
        start = time.perf_counter()
        with precision_context(model, precision), (directory / "epochs.jsonl").open("w") as log:
            for epoch in range(1, config["epochs"] + 1):
                training = train_epoch(model, data, optimizer, effective, device)
                validation = evaluate(model, data, effective, device, "val")
                selected = checkpoint.consider(validation["loss"], epoch, model, optimizer)
                row = dict(epoch=epoch, train=training, validation=validation, selected=selected)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if epoch == 1 or epoch % 25 == 0:
                    print(f"{precision}/{control} seed={seed} epoch={epoch} "
                          f"val_loss={validation['loss']:.4f} val_acc={validation['accuracy']:.3f}", flush=True)
            selected = torch.load(checkpoint.path, weights_only=False, map_location="cpu")
            model.load_state_dict(selected["model"])
            test = evaluate(model, data, effective, device, "test")
            validation = evaluate(model, data, effective, device, "val")
        if precision == "qat":
            with QuantizationObserver(model) as observer:
                evaluate(model, data, effective, device, "val")
            quantization = observer.result()
            write_json(directory / "quantization.json", quantization)
        else:
            quantization = dict(mean_activation_effective_bits=32.0,
                                note="All fake quantizer leaves bypassed; their parameters are frozen.")
            write_json(directory / "quantization.json", quantization)
        result = dict(dataset="CiteSeer", precision=precision, control=control, seed=seed,
                      epochs=config["epochs"], best_epoch=checkpoint.epoch,
                      validation=validation, test=test, test_evaluations=1, new_test_evaluations=1,
                      training_seconds=time.perf_counter() - start,
                      mean_activation_effective_bits=quantization["mean_activation_effective_bits"],
                      **initial)
    result["checkpoint_sha256"] = file_digest(directory / "best.pt")
    write_json(directory / "result.json", result)
    print(f"DONE {precision}/{control} seed={seed} test={result['test']['accuracy']:.3f} "
          f"best={result['best_epoch']}", flush=True)
    return result, model


def vector_metrics(error, delta):
    error, delta = error.double(), delta.double()
    e2, p2 = float(error.square().sum()), float(delta.square().sum())
    dot = float((error * delta).sum())
    after2 = float((error - delta).square().sum())
    cosine = dot / (e2 * p2) ** 0.5 if e2 > 0 and p2 > 0 else None
    norm = e2 ** 0.5
    return dict(before_frobenius=norm, after_frobenius=after2 ** 0.5,
                norm_reduction=1 - (after2 / e2) ** 0.5 if e2 else None,
                squared_error_reduction=1 - after2 / e2 if e2 else None,
                cosine=cosine, prompt_norm=p2 ** 0.5,
                prompt_to_error_norm=(p2 / e2) ** 0.5 if e2 else None,
                projection_scalar=dot / p2 if p2 else None)


def decompose(delta, error):
    # Per-node projection onto the full GIN input correction, including self.
    denominator = error.square().sum(1, keepdim=True)
    coefficient = (delta * error).sum(1, keepdim=True) / denominator.clamp_min(1e-30)
    parallel = coefficient * error
    return parallel, delta - parallel


@contextmanager
def intervene(model, mode="learned", strength=1.0, seed=41, records=None):
    """Validation-only oracle intervention at conv2; leaves upstream states fixed.

    strength=0 bypasses only conv2's message quantizer; 1 is the trained model.
    Intermediate values interpolate its error, 2 extrapolates (not a bitwidth).
    """
    if mode not in INTERVENTIONS:
        raise ValueError(mode)

    def forward(layer, x, edge_index, budget, prompt=None, diagnostics=None,
                layer_name="", node_mask=None):
        edges, _ = remove_self_loops(edge_index)
        raw_q = layer.q_xw(x.T).T
        q = raw_q if strength == 1 else x if strength == 0 else x + strength * (raw_q - x)
        aggregate = sum_messages(q, edges)
        reference = sum_messages(x, edges)
        delta = prompt.add(aggregate) - aggregate
        neighbor_error = reference - aggregate
        update_error = (1 + layer.eps) * (x - q) + neighbor_error
        parallel, orthogonal = decompose(delta, update_error)
        if mode == "learned":
            chosen = delta
        elif mode == "off":
            chosen = torch.zeros_like(delta)
        elif mode == "signflip":
            chosen = -delta
        elif mode == "shuffle":
            order = torch.randperm(len(delta), generator=torch.Generator().manual_seed(seed + 400_000))
            chosen = delta[order.to(delta.device)]
        elif mode == "mean":
            chosen = delta.mean(0, keepdim=True).expand_as(delta)
        elif mode == "parallel":
            chosen = parallel
        elif mode == "orthogonal":
            chosen = orthogonal
        if records is not None:
            mask = node_mask if node_mask is not None else slice(None)
            records.append(dict(neighbor=vector_metrics(neighbor_error[mask], chosen[mask]),
                                full_update=vector_metrics(update_error[mask], chosen[mask]),
                                parallel_prompt_norm=float(parallel[mask].double().norm()),
                                orthogonal_prompt_norm=float(orthogonal[mask].double().norm())))
        # Preserve original add() arithmetic exactly in the baseline intervention.
        corrected = prompt.add(aggregate) if mode == "learned" else aggregate + chosen
        return layer.mlp((1 + layer.eps) * q + corrected), budget

    with replace_forward(model.conv2, forward):
        yield


@torch.no_grad()
def validation_probes(model, data, config, seed, trained_precision):
    model.eval()
    device = torch.device("cpu")
    original_digest = state_digest(model, "all")
    rows = []
    with precision_context(model, trained_precision):
        baseline_logits = model(data)[0]
        for mode in INTERVENTIONS:
            records = []
            with intervene(model, mode, seed=seed, records=records):
                value = evaluate(model, data, config, device, "val")
                if mode == "learned":
                    actual = model(data)[0]
                    torch.testing.assert_close(actual, baseline_logits, rtol=0, atol=0)
            rows.append(dict(precision=trained_precision, seed=seed, intervention=mode,
                             strength=1.0, validation=value, geometry=records[0]))
        if trained_precision == "qat":
            for strength in (0., 0.25, 0.5, 2.0):
                for mode in ("learned", "off"):
                    records = []
                    with intervene(model, mode, strength, seed, records):
                        value = evaluate(model, data, config, device, "val")
                    rows.append(dict(precision=trained_precision, seed=seed,
                                     intervention=mode, strength=strength,
                                     validation=value, geometry=records[0]))
            for mode in ("learned", "off"):
                with bypass_quantizers(model), intervene(model, mode, seed=seed):
                    value = evaluate(model, data, config, device, "val")
                rows.append(dict(precision=trained_precision, seed=seed,
                                 intervention="all_quantizers_bypassed_" + mode,
                                 strength=None, validation=value))
    assert state_digest(model, "all") == original_digest
    return rows


def verify_reuse_sources(pilot):
    old = json.loads((pilot / "provenance.json").read_text())["source_sha256"]
    # report.py was updated after the original pilot; it cannot affect training.
    dependencies = ("models.py", "prompts.py", "config.py", "data.py", "run.py", "diagnostics.py")
    names = [name for name in old if not name.startswith("runs/lorap_study/")]
    names += ["runs/lorap_study/" + name for name in dependencies]
    for name in names:
        if file_digest(ROOT / name) != old[name]:
            raise ValueError(f"Pilot training dependency changed: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-pilot", type=Path, default=ROOT / "result/lorap_study/pilot_20260909_v1")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dataset_config("CiteSeer", "pilot")
    data, manifest = load_data(config)
    pilot = None if args.no_reuse else args.reuse_pilot.resolve()
    if pilot is not None:
        verify_reuse_sources(pilot)
        assert manifest == json.loads((pilot / "CiteSeer/split_manifest.json").read_text())
    provenance(output)
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "protocol.json", dict(
        config=config, precisions=PRECISIONS, controls=CONTROLS,
        paired_seeds=config["seeds"], prompt_parameters=412, mlp_hidden_width=12,
        fp32="All quantizer leaves bypassed, quantizer parameters frozen, storage penalty zero",
        qat="Original A2Q quantizers, optimizer rates and storage penalty unchanged",
        selection="Minimum validation CE, earlier epoch on ties; one final test per new fit",
        interventions_split="validation only", interventions=list(INTERVENTIONS),
        error_strengths=[0, .25, .5, 1, 2],
        primary_estimand="Paired interaction: (LoRAP-none) QAT minus (LoRAP-none) FP32",
        warning="3 seeds; same hyperparameters, no independent hyperparameter search; QAT/FP32 interaction also includes training-regime effects"))
    results, probes = [], []
    for seed in config["seeds"]:
        backbone = None
        prompt_base = None
        for precision in PRECISIONS:
            for control in CONTROLS:
                directory = output / precision / control / f"seed_{seed}"
                result, model = run_control(config, data, seed, precision, control, directory, pilot)
                if backbone is not None:
                    assert result["backbone_sha256"] == backbone
                backbone = result["backbone_sha256"]
                if control == "original":
                    if prompt_base is not None:
                        assert result["prompt_base_sha256"] == prompt_base
                    prompt_base = result["prompt_base_sha256"]
                    rows = validation_probes(model, data, config, seed, precision)
                    write_json(directory / "interventions.json", rows)
                    probes.extend(rows)
                results.append(result)
                write_json(output / "results.json", results)
                write_json(output / "interventions.json", probes)
    from .mechanism_report import generate_report
    generate_report(output)


if __name__ == "__main__":
    main()

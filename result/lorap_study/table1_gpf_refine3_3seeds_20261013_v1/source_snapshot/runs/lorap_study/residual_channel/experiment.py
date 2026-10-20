"""GPU-only frozen-checkpoint calibration and validation; never opens test data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_self_loops

from ..data import stratified_indices
from ..full_models import build_full_backbone
from ..full_run import atomic_save, require_cuda
from ..full_validation import validation_reproduction
from ..models import state_digest
from ..run import write_json
from .core import (canonical_basis, encode, error_terms, graph_operator,
                   higher_precision, residual_correction, select_channels, selected_precision)

MODES = ("none", "random", "pca", "graph", "higher_8", "higher_16", "higher_32",
         "higher_64", "higher_110", "float_graph", "shuffle_graph", "bypass", "bypass_side")
SIDE_MODES = ("random", "pca", "graph", "float_graph", "shuffle_graph")
DEFAULT_SOURCE = Path("result/lorap_study/full_gpu_20260909_v1").resolve()
DEFAULT_ROOT = Path("result/lorap_study/residual_channel_20260912_v2").resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loader(dataset, size=128):
    return DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0)


def choice(mode, info, x, q, edges, eps, batch, generator):
    if mode.startswith("higher_"):
        indices = info["channels"][mode]
        high = higher_precision(x, info["quantizer"])
        upgraded = q.clone()
        upgraded[:, indices] = high[:, indices]
        delta = graph_operator(upgraded - q, edges, eps)
        return delta, None, upgraded
    if mode in SIDE_MODES or mode == "bypass_side":
        basis_name = mode if mode in ("random", "pca") else "graph"
        basis, scale = info[basis_name]["basis"], info[basis_name]["scale"]
        delta, details = residual_correction(x - q, edges, eps, basis, scale,
            batch=batch, shuffle=mode == "shuffle_graph", generator=generator,
            floating=mode == "float_graph")
        details.update(basis=basis, scale=scale)
        return delta, details, None
    return torch.zeros_like(x), None, None


@torch.no_grad()
def forward(model, data, mode="none", calibration=None, observer=None, generator=None):
    if data.x.device.type != "cuda":
        raise RuntimeError("Model forward requires CUDA")
    x = model.prompt_node.add(torch.cat((data.x, data.pos), 1))
    x, _, budget = model.embedding(x, data.edge_index, x.new_zeros(1))
    edges, _ = remove_self_loops(data.edge_index)
    for index, conv in enumerate(model.convs):
        if mode.startswith("higher_") and observer is None:
            q = selected_precision(x, conv.q_xw, calibration[index]["channels"][mode])
            x, budget = conv.mlp(graph_operator(q, edges, conv.eps), edges, budget)
            continue
        q = x if mode.startswith("bypass") else conv.q_xw(x.T).T
        info = dict(calibration[index], quantizer=conv.q_xw) if calibration else None
        delta, details, upgraded = choice(mode, info, x, q, edges, conv.eps, data.batch, generator)
        out = graph_operator(upgraded if upgraded is not None else q, edges, conv.eps)
        if upgraded is None and mode not in ("none", "bypass"):
            out = out + delta
        if observer:
            observer(index, x, q, edges, conv.eps, data.batch, delta, details, conv.q_xw)
        x, budget = conv.mlp(out, edges, budget)
    x = global_mean_pool(x, data.batch)
    x, _, budget = model.lin1(x, data.edge_index, budget)
    x, _, budget = model.lin2(F.relu(x), data.edge_index, budget)
    return x


class Reservoir:
    """Same node priorities across methods; records stream positions for replay."""
    def __init__(self, seed, device, maximum=4096):
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.maximum, self.seen = maximum, 0
        self.values, self.keys, self.ids = {}, None, None

    def add(self, **values):
        n = len(next(iter(values.values())))
        keys = torch.rand(n, generator=self.generator, device=next(iter(values.values())).device)
        ids = torch.arange(self.seen, self.seen+n, device=keys.device)
        self.seen += n
        if self.keys is not None:
            keys, ids = torch.cat((self.keys, keys)), torch.cat((self.ids, ids))
            values = {k: torch.cat((self.values[k], v)) for k, v in values.items()}
        select = keys.topk(min(self.maximum, len(keys))).indices
        self.keys, self.ids = keys[select], ids[select]
        self.values = {k: v[select].detach() for k, v in values.items()}


@torch.no_grad()
def calibrate(model, training, device):
    samples = [Reservoir(912_041+i, device) for i in range(4)]
    all_r = [[] for _ in range(4)]

    def capture(i, x, q, edges, eps, batch, delta, details, quantizer):
        r = x - q
        e = graph_operator(r, edges, eps)
        e5 = graph_operator(x - higher_precision(x, quantizer), edges, eps)
        samples[i].add(residual=r, propagated=e, high_error=e5)
        all_r[i].append(r)

    for data in loader(training):
        forward(model, data.to(device), observer=capture)
    calibration, manifest = [], []
    for i, sample in enumerate(samples):
        r, e = sample.values["residual"], sample.values["propagated"]
        graph, spectrum = canonical_basis(e)
        pca, raw_spectrum = canonical_basis(r)
        generator = torch.Generator(device=device).manual_seed(913_041+i)
        random = torch.linalg.qr(torch.randn(r.size(1), 2, device=device, generator=generator)).Q
        info = {}
        full_r = torch.cat(all_r[i])
        for name, basis in (("graph", graph), ("pca", pca), ("random", random)):
            scale = (full_r @ basis).abs().amax(0).div(127).clamp_min(1e-12)
            info[name] = dict(basis=basis, scale=scale)
        gains = e.double().square().sum(0) - sample.values["high_error"].double().square().sum(0)
        info["channels"] = {mode: select_channels(gains, int(mode.split("_")[1]))
                            for mode in MODES if mode.startswith("higher_")}
        calibration.append(info)
        manifest.append(dict(layer=i+1, seen_nodes=sample.seen, sampled_nodes=len(sample.ids),
            node_stream_indices=sample.ids.cpu().tolist(), propagated_eigenvalues=spectrum.cpu().tolist(),
            raw_residual_eigenvalues=raw_spectrum.cpu().tolist(), channel_gains=gains.cpu().tolist(),
            selected_channels={k: v.cpu().tolist() for k, v in info["channels"].items()},
            orthogonality_max_error={k: float((v["basis"].T@v["basis"]-torch.eye(2,device=device)).abs().max())
                                     for k,v in info.items() if k != "channels"}))
    return calibration, manifest


class Diagnostics:
    def __init__(self, calibration, device, all_controls=False, spectrum=False):
        self.calibration, self.all_controls = calibration, all_controls
        self.totals = [{} for _ in range(4)]
        self.samples = [Reservoir(914_041+i, device) for i in range(4)] if spectrum else None
        self.generator = torch.Generator(device=device).manual_seed(915_041)
        self.nodes = [0]*4
        self.edges = [0]*4

    def __call__(self, i, x, q, edges, eps, batch, delta, details, quantizer):
        self.nodes[i] += len(x)
        self.edges[i] += edges.size(1)
        if self.samples:
            self.samples[i].add(error=graph_operator(x-q,edges,eps))
        modes = MODES if self.all_controls else ("active",)
        for mode in modes:
            if mode.startswith("bypass"):
                # Local intervention: bypass only this quantizer at fixed input.
                correction, _, _ = choice("bypass_side", self.calibration[i], x, x, edges, eps, batch, self.generator)
                if bool(correction.count_nonzero()):
                    raise AssertionError("Bypassed quantization produced a nonzero residual channel")
                continue
            if self.all_controls:
                info = dict(self.calibration[i], quantizer=quantizer)
                delta, details, _ = choice(mode, info, x, q, edges, eps, batch, self.generator)
            terms = error_terms(x-q, delta, edges, eps,
                basis=details["basis"] if details else None,
                decoded=details["decoded"] if details else None)
            identity_gap = terms.pop("identity_relative_gap", None)
            if details:
                terms["saturated_values"] = (details["values"].abs() > 127*details["scale"]).sum()
                terms["code_values"] = details["values"].numel()
            state = self.totals[i].setdefault(mode, {})
            for k,v in terms.items():
                state[k] = state.get(k, 0) + v
            if identity_gap is not None:
                state["max_batch_identity_relative_gap"] = torch.maximum(
                    state.get("max_batch_identity_relative_gap", torch.zeros_like(identity_gap)), identity_gap)

    def result(self):
        result = []
        for i, states in enumerate(self.totals):
            row = dict(layer=i+1, nodes=self.nodes[i], edges=self.edges[i], modes={})
            for mode, state in states.items():
                s = {k: float(v) for k,v in state.items()}
                den = max(s["before_sq"], 1e-30)
                s["mse_reduction"] = 1-s["after_sq"]/den
                s["norm_ratio"] = (s["after_sq"]/den)**.5
                s["cosine"] = s["dot"]/max((s["before_sq"]*s["correction_sq"])**.5,1e-30)
                if "captured_sq" in s:
                    s["compensation_condition"] = s["side_quantization_sq"] < s["captured_sq"]
                    s["saturation_fraction"] = s["saturated_values"]/s["code_values"]
                    if s["max_batch_identity_relative_gap"] > 2e-5:
                        raise AssertionError(f"Orthogonal error identity failed: {s}")
                if not all(np.isfinite(v) for v in s.values()):
                    raise FloatingPointError("Nonfinite diagnostics")
                row["modes"][mode] = s
            if self.samples:
                sample = self.samples[i]
                _, energy = canonical_basis(sample.values["error"])
                row["spectrum"] = dict(sampled_nodes=len(sample.ids), node_stream_indices=sample.ids.cpu().tolist(),
                    squared_singular_values=energy.cpu().tolist(),
                    cumulative_energy=(energy.cumsum(0)/energy.sum().clamp_min(1e-30)).cpu().tolist())
            result.append(row)
        return result


@torch.no_grad()
def evaluate_modes(model, dataset, calibration, device, diagnostics, progress=None):
    totals = {mode: torch.zeros(3, device=device, dtype=torch.float64) for mode in MODES}
    seconds = dict.fromkeys(MODES, 0.)
    collectors = {mode: Diagnostics(calibration, device, all_controls=mode=="none", spectrum=mode=="none")
                  for mode in MODES if diagnostics and not mode.startswith("bypass")}
    generator = torch.Generator(device=device).manual_seed(916_041)
    max_bypass_diff = 0.
    batches = loader(dataset)
    for j, data in enumerate(batches):
        data = data.to(device)
        bypass = None
        for mode in MODES:
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = forward(model, data, mode, calibration, collectors.get(mode), generator)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(f"Nonfinite outputs: {mode}")
            y = data.y.reshape(-1)
            totals[mode] += torch.stack((F.cross_entropy(logits, y, reduction="sum").double(),
                                       (logits.argmax(1)==y).sum(), y.new_tensor(len(y)))).double()
            if mode == "bypass":
                bypass = logits
            elif mode == "bypass_side":
                max_bypass_diff = max(max_bypass_diff, float((logits-bypass).abs().max()))
            torch.cuda.synchronize()
            seconds[mode] += time.perf_counter()-started
        if progress and (j%10==0 or j+1==len(batches)):
            write_json(progress, dict(phase="validation", batches=j+1, total_batches=len(batches), updated_unix=time.time()))
            print(f"validation batch {j+1}/{len(batches)}", flush=True)
    metrics = {}
    for mode, tensor in totals.items():
        loss, correct, n = tensor.cpu().tolist()
        metrics[mode] = dict(loss=loss/n, accuracy=correct/n, examples=int(n),
                             seconds_including_diagnostics=seconds[mode])
    return dict(metrics=metrics, diagnostics={k:v.result() for k,v in collectors.items()},
                bypass_side_max_logit_difference=max_bypass_diff)


@torch.no_grad()
def benchmark(model, validation, calibration, device):
    rows = []
    for batch_size in (32, 128):
        batch = next(iter(loader(validation, batch_size))).to(device)
        for mode in ("none", "graph", "higher_16"):
            for _ in range(3):
                forward(model, batch, mode, calibration)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                forward(model, batch, mode, calibration)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            rows.append(dict(batch_size=batch_size, mode=mode, seconds_median=float(np.median(times)),
                seconds_samples=times, resident_cuda_bytes=baseline,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                incremental_peak_cuda_bytes=torch.cuda.max_memory_allocated(device)-baseline))
    return rows


def cpu_tree(value):
    if isinstance(value, dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value.detach().cpu() if torch.is_tensor(value) else value


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = require_cuda()
    directory = args.root/args.dataset/f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    source = args.source/args.dataset/"none"/f"seed_{args.seed}"
    tracked = ("best.pt", "last.pt", "result.json", "test_started.json", "config.json")
    hashes = {name:sha(source/name) for name in tracked}
    if (directory/"result.json").exists():
        result = json.loads((directory/"result.json").read_text())
        if result["source_hashes_before"] != hashes or result["pilot"] != args.pilot:
            raise ValueError("Existing result has different inputs")
        return
    started = time.perf_counter()
    config = json.loads((source/"config.json").read_text())
    # Only train and validation are instantiated; test split is never loaded.
    datasets = {s:GNNBenchmarkDataset(config["data_root"], args.dataset, split=s) for s in ("train","val")}
    labels = datasets["train"]._data.y.cpu().numpy().reshape(-1)
    selected = stratified_indices(labels, 512, 912041)
    fit_local = stratified_indices(labels[selected], 256, 912042)
    fit_indices = [selected[i] for i in fit_local]
    heldout_indices = sorted(set(selected)-set(fit_indices))
    write_json(directory/"config.json", dict(dataset=args.dataset, seed=args.seed, source=str(source),
        rank=2, side_bits=8, accumulation="int32 neighbors; FP32 weighted self term and decode",
        scale_rule="training-fit max absolute projected residual / 127; floor 1e-12",
        fit_graphs=256, heldout_graphs=256, max_basis_nodes=4096, modes=list(MODES),
        graph_selection_seed=912041, split_seed=912042, basis_sampling_seed=912041,
        random_basis_seed=913041, validation_sampling_seed=914041,
        calibration_inputs="frozen baseline layer inputs, all layers calibrated independently",
        selection="no optimizer, no validation selection, no additional test evaluation",
        pilot=args.pilot, prior_training_config=config))
    write_json(directory/"indices.json", dict(fit=fit_indices, heldout=heldout_indices,
        validation=list(range(min(256,len(datasets["val"])) if args.pilot else len(datasets["val"])))) )
    model = build_full_backbone(config, args.seed, datasets)
    checkpoint = torch.load(source/"best.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    if any(not bool(torch.all(conv.q_xw.bit.abs().round()==4)) for conv in model.convs):
        raise AssertionError("Expected four-bit message quantizers")
    before = state_digest(model, "all")
    first = next(iter(loader(datasets["train"][fit_indices]))).to(device)
    with torch.no_grad():
        original, _ = model(first)
        plain = forward(model, first)
    replay = float((original-plain).abs().max())
    if not torch.allclose(original, plain, atol=2e-5, rtol=1e-5):
        raise AssertionError(f"Baseline forward changed: {replay}")
    calibration, manifest = calibrate(model, datasets["train"][fit_indices], device)
    after_calibration = state_digest(model, "all")
    diagnostic_replays = {}
    for mode in ("none", "graph", "higher_16"):
        plain = forward(model, first, mode, calibration)
        observed = forward(model, first, mode, calibration, Diagnostics(calibration, device))
        diagnostic_replays[mode] = float((plain-observed).abs().max())
        if not torch.allclose(plain, observed, atol=2e-4, rtol=1e-4):
            raise AssertionError(f"Diagnostic switch changed {mode} outputs")
    atomic_save(cpu_tree(calibration), directory/"calibration.pt")
    write_json(directory/"calibration.json", manifest)
    calibration_seconds = time.perf_counter()-started
    print(f"{args.dataset} seed={args.seed} calibration {calibration_seconds:.1f}s",flush=True)
    heldout = evaluate_modes(model, datasets["train"][heldout_indices], calibration, device, True)
    write_json(directory/"heldout.json", heldout)
    validation_data = datasets["val"][:256] if args.pilot else datasets["val"]
    validation = evaluate_modes(model, validation_data, calibration, device, True, directory/"progress.json")
    validation_replay = (validation_reproduction(checkpoint["validation_loss"], validation["metrics"]["none"]["loss"])
                         if not args.pilot else None)
    write_json(directory/"validation.json", validation)
    timings = benchmark(model, datasets["val"], calibration, device)
    write_json(directory/"benchmark.json", timings)
    after = state_digest(model, "all")
    hashes_after = {name:sha(source/name) for name in tracked}
    if before != after_calibration or before != after or hashes != hashes_after:
        raise AssertionError("Frozen model, BatchNorm, or source artifacts changed")
    # CUDA floating scatter can vary in summation order between repeated calls.
    if validation["bypass_side_max_logit_difference"] != 0:
        raise AssertionError("Deterministic bypass side and bypass baseline must agree exactly")
    result = dict(dataset=args.dataset, seed=args.seed, pilot=args.pilot, device=str(device),
        gpu_name=torch.cuda.get_device_name(), physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        checkpoint_epoch=checkpoint["epoch"], source_hashes_before=hashes, source_hashes_after=hashes_after,
        model_sha256_before=before, model_sha256_after_calibration=after_calibration, model_sha256_after=after,
        baseline_max_logit_difference=replay, calibration_seconds=calibration_seconds,
        validation_reproduction=validation_replay, deterministic_algorithms=True,
        diagnostic_max_logit_differences=diagnostic_replays,
        total_seconds=time.perf_counter()-started, optimizer_updates=0, test_evaluations=0,
        validation=validation["metrics"], heldout=heldout["metrics"],
        bypass_side_max_logit_difference=validation["bypass_side_max_logit_difference"],
        base_state_tensor_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
        budgets={mode:dict(base_message_bits_per_node_per_layer=440,
            extra_bits_per_node_per_layer=([int(len(c["channels"][mode])) for c in calibration]
                if mode.startswith("higher_") else [64 if mode=="float_graph" else 16 if mode in SIDE_MODES else 0]*4),
            extra_static_tensor_bytes=3552 if mode in SIDE_MODES else 0)
            for mode in MODES if not mode.startswith("bypass")})
    write_json(directory/"result.json", result)
    write_json(directory/"progress.json", dict(phase="complete", total_seconds=result["total_seconds"]))
    print(json.dumps(dict(dataset=args.dataset, seed=args.seed, seconds=result["total_seconds"],
                         validation=result["validation"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MNIST", "CIFAR10"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())

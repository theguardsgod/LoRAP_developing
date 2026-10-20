"""Bounded validation-led improvement study; test results never choose variants."""

import argparse
import json
from pathlib import Path
import time

import torch

from .config import dataset_config
from .data import load_data, loaders_for
from .diagnostics import AggregationDiagnostics, QuantizationObserver
from .improved import (FullUpdateDiagnostics, bare_model, calibrate, calibration_json,
                       install_prompt)
from .models import model_counts, optimizer_for, state_digest
from .run import (BestCheckpoint, evaluate, provenance, seed_training, train_epoch,
                  write_json)

CANDIDATES = ("zero_init", "stable_random", "stable_spectral")
MNIST_CONTROLS = ("original", "stable_spectral", "adaptive_spectral")


def choose_candidate(results):
    # Stable min keeps the predeclared candidate order on exact ties.
    return min(results, key=lambda r: r["validation"]["loss"])["variant"]


def fit(config, variant, seed, data, calibration, directory):
    directory.mkdir(parents=True, exist_ok=False)
    model = bare_model(config["dataset"], seed, data if config["dataset"] == "CiteSeer" else None)
    ranks = install_prompt(model, config["dataset"], variant, seed, calibration)
    effective = dict(config, variant=variant, seed=seed, ranks=ranks, device="cpu")
    write_json(directory / "config.json", effective)
    initial = dict(backbone_sha256=state_digest(model), initial_state_sha256=state_digest(model, "all"),
                   ranks=ranks, **model_counts(model))
    write_json(directory / "initialization.json", initial)
    optimizer = optimizer_for(model, effective)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                          patience=5, threshold=.0001)
                 if config["dataset"] == "MNIST" else None)
    if config["dataset"] == "CiteSeer":
        train, validation = data, data
    else:
        loaders = loaders_for(data, config, seed)
        train, validation = loaders["train"], loaders["val"]
    checkpoint = BestCheckpoint(directory / "best.pt")
    seed_training(seed + 300_000)
    device = torch.device("cpu")
    start = time.perf_counter()
    with (directory / "epochs.jsonl").open("w") as stream:
        for epoch in range(1, config["epochs"] + 1):
            training = train_epoch(model, train, optimizer, effective, device)
            val = evaluate(model, validation, effective, device, "val")
            selected = checkpoint.consider(val["loss"], epoch, model, optimizer)
            if scheduler is not None:
                scheduler.step(val["loss"])
            stream.write(json.dumps(dict(epoch=epoch, train=training, validation=val, selected=selected),
                                    allow_nan=False) + "\n")
            stream.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
                print(f"{config['dataset']} {variant} seed={seed} epoch={epoch}/{config['epochs']} "
                      f"val_CE={val['loss']:.4f} val_acc={val['accuracy']:.3f}", flush=True)
    state = torch.load(checkpoint.path, weights_only=False, map_location="cpu")
    model.load_state_dict(state["model"])
    collector = AggregationDiagnostics(config["split_seed"], config["diagnostic_max_nodes"])
    with QuantizationObserver(model) as observer:
        val = evaluate(model, validation, effective, device, "val", collector)
    write_json(directory / "diagnostics.json", collector.result())
    write_json(directory / "quantization.json", observer.result())
    mask = data.val_mask if config["dataset"] == "CiteSeer" else None
    with FullUpdateDiagnostics(model, mask) as full:
        evaluate(model, validation, effective, device, "val")
    write_json(directory / "full_update_diagnostics.json", full.rows)
    result = dict(dataset=config["dataset"], variant=variant, seed=seed,
                  epochs=config["epochs"], best_epoch=checkpoint.epoch,
                  validation=val, test=None, test_evaluations=0,
                  training_seconds=time.perf_counter() - start,
                  mean_activation_effective_bits=observer.result()["mean_activation_effective_bits"],
                  **initial)
    write_json(directory / "result.json", result)
    print(f"FIT DONE {config['dataset']} {variant} seed={seed} best={checkpoint.epoch}, "
          f"val_CE={val['loss']:.4f}; test withheld", flush=True)
    return model, result


def final_test(model, result, config, data, directory):
    if result["test_evaluations"]:
        raise ValueError("Test already evaluated")
    test = data if config["dataset"] == "CiteSeer" else loaders_for(data, config, result["seed"])["test"]
    result["test"] = evaluate(model, test, config, torch.device("cpu"), "test")
    result["test_evaluations"] = 1
    write_json(directory / "result.json", result)
    print(f"TEST {result['dataset']} {result['variant']} seed={result['seed']}: "
          f"{result['test']['accuracy']:.4f}", flush=True)


def run_dataset(config, args, output):
    dataset_name = config["dataset"]
    data, manifest = load_data(config)
    write_json(output / dataset_name / "split_manifest.json", manifest)
    results, pending, calibrations = [], [], {}
    is_screen = args.variant == "screen" and dataset_name == "CiteSeer"
    selected_variant = None
    for seed_index, seed in enumerate(config["seeds"]):
        base = bare_model(dataset_name, seed, data if dataset_name == "CiteSeer" else None)
        calibration = calibrate(base, dataset_name, data, seed=seed)
        calibrations[seed] = calibration
        directory = output / dataset_name / f"calibration_seed_{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(calibration, directory / "calibration.pt")
        write_json(directory / "calibration.json", calibration_json(calibration))
        if is_screen:
            variants = CANDIDATES if seed_index == 0 else (selected_variant,)
        else:
            variants = MNIST_CONTROLS if args.variant == "screen" else (args.variant,)
        seed_results, seed_pending = [], []
        for variant in variants:
            path = output / dataset_name / variant / f"seed_{seed}"
            model, result = fit(config, variant, seed, data, calibration, path)
            results.append(result)
            seed_results.append(result)
            seed_pending.append((model, result, path))
            write_json(output / dataset_name / "results.json", results)
        if is_screen and seed_index == 0:
            selected_variant = choose_candidate(seed_results)
            write_json(output / dataset_name / "selection.json", dict(
                discovery_seed=seed, candidates=list(CANDIDATES), selected_variant=selected_variant,
                criterion="minimum validation CE at its best epoch; candidate-order tie break",
                test_results_seen=0,
                validation_scores={r["variant"]: r["validation"]["loss"] for r in seed_results}))
            pending.extend(item for item in seed_pending if item[1]["variant"] == selected_variant)
            print(f"LOCKED CiteSeer variant={selected_variant}; selection uses validation only", flush=True)
        else:
            pending.extend(seed_pending)
    # Every architecture/hyperparameter choice is locked before new test metrics.
    for model, result, path in pending:
        final_test(model, result, config, data, path)
    write_json(output / dataset_name / "results.json", results)
    full = dataset_config(dataset_name, "full")
    full.update(variant=selected_variant or args.variant, calibration_max_nodes=4096,
                calibration_max_graphs=128, total_rank=2 if dataset_name == "CiteSeer" else 8,
                full_data=True)
    write_json(output / "full_configs" / f"{dataset_name}.json", full)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("CiteSeer", "MNIST", "both"), default="both")
    parser.add_argument("--variant", choices=("screen", "original", "conditioned", "zero_init",
                                              "stable_random", "stable_spectral", "adaptive_spectral"), default="screen")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.epochs is not None and args.epochs < 1) or not args.seeds:
        parser.error("Positive threads/epochs and at least one seed required")
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance(output)
    configs = []
    for name in ("CiteSeer", "MNIST"):
        if args.dataset in ("both", name):
            config = dataset_config(name, "pilot")
            config.update(epochs=args.epochs or (100 if name == "CiteSeer" else 30), seeds=args.seeds)
            if args.full_data:
                config["subset_sizes"] = None
            configs.append(config)
    write_json(output / "protocol.json", dict(configs=configs,
        discovery_candidates=CANDIDATES, mnist_controls=MNIST_CONTROLS,
        cite_selection="First seed validation CE, lock variant before remaining seeds and before all new tests",
        calibration="training nodes/graphs, maximum 4096 nodes/layer, initial zero-prompt backbone in eval mode",
        rank_rule="At least one/layer, greedy next normalized error eigenvalue, total rank 8 for MNIST",
        initialization="P_left zero, P_right nonzero; random orthogonal or training-error right singular vectors",
        inference="quantized messages and detached quantizer conditions only; no FP reference branch",
        residual_scale=1.0))
    all_results = []
    for config in configs:
        all_results.extend(run_dataset(config, args, output))
        write_json(output / "results.json", all_results)
    from .improve_report import generate_report
    generate_report(output)


if __name__ == "__main__":
    main()

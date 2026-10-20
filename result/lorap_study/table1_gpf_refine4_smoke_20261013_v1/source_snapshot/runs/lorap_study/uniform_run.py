"""Uniform LoRAP: a locked main configuration and paired component ablations."""

import argparse
import json
from pathlib import Path
import time

import torch

from .config import dataset_config
from .data import load_data, loaders_for
from .diagnostics import AggregationDiagnostics, QuantizationObserver
from .improved import FullUpdateDiagnostics, bare_model, calibration_json
from .models import model_counts, optimizer_for, state_digest
from .run import BestCheckpoint, evaluate, provenance, seed_training, train_epoch, write_json
from .uniform import VARIANTS, calibrate_uniform, install_uniform


def primary_config(dataset, full=False):
    config = dataset_config(dataset, "full" if full else "pilot")
    config.update(variants=["main"], method="zero_initialization_standardized_conditions_uniform_rank",
                  basis="random_orthogonal", calibration_max_nodes=4096, calibration_max_graphs=128,
                  device="cpu")
    if not full:
        config.update(seeds=[44, 45, 46], epochs=30 if dataset == "MNIST" else 100)
    return config


def fit_uniform(config, variant, seed, data, calibration, directory):
    directory.mkdir(parents=True, exist_ok=False)
    dataset = config["dataset"]
    model = bare_model(dataset, seed, data if dataset == "CiteSeer" else None)
    ranks = install_uniform(model, dataset, seed, calibration, variant)
    initial = dict(backbone_sha256=state_digest(model), initial_state_sha256=state_digest(model, "all"),
                   ranks=ranks, calibration_buffer_bytes=sum(b.numel()*b.element_size() for b in model.prompt_aggr.buffers()),
                   **model_counts(model))
    write_json(directory / "initialization.json", initial)
    write_json(directory / "config.json", dict(config, variant=variant, seed=seed, ranks=ranks))
    optimizer = optimizer_for(model, config)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                         patience=5, threshold=.0001)
                 if dataset == "MNIST" else None)
    loaders = loaders_for(data, config, seed) if dataset == "MNIST" else None
    train, validation = (loaders["train"], loaders["val"]) if loaders else (data, data)
    checkpoint = BestCheckpoint(directory / "best.pt")
    seed_training(seed + 300_000)
    device = torch.device("cpu")
    start = time.perf_counter()
    with (directory / "epochs.jsonl").open("w") as stream:
        for epoch in range(1, config["epochs"] + 1):
            training = train_epoch(model, train, optimizer, config, device)
            val = evaluate(model, validation, config, device, "val")
            selected = checkpoint.consider(val["loss"], epoch, model, optimizer)
            if scheduler is not None:
                scheduler.step(val["loss"])
            stream.write(json.dumps(dict(epoch=epoch, train=training, validation=val, selected=selected), allow_nan=False)+"\n")
            stream.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
                print(f"{dataset} {variant} seed={seed} epoch={epoch}/{config['epochs']} "
                      f"val_CE={val['loss']:.4f} val_acc={val['accuracy']:.3f}", flush=True)
    state = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    collector = (AggregationDiagnostics(config["split_seed"], config["diagnostic_max_nodes"])
                 if config["diagnostics"] else None)
    with QuantizationObserver(model) as observer:
        val = evaluate(model, validation, config, device, "val", collector)
    write_json(directory / "diagnostics.json", collector.result() if collector else {"enabled": False})
    write_json(directory / "quantization.json", observer.result())
    if config["diagnostics"]:
        with FullUpdateDiagnostics(model, data.val_mask if dataset == "CiteSeer" else None) as full:
            evaluate(model, validation, config, device, "val")
        write_json(directory / "full_update_diagnostics.json", full.rows)
    result = dict(dataset=dataset, variant=variant, seed=seed, epochs=config["epochs"],
                  best_epoch=checkpoint.epoch, validation=val, test=None, test_evaluations=0,
                  training_seconds=time.perf_counter()-start,
                  mean_activation_effective_bits=observer.result()["mean_activation_effective_bits"], **initial)
    write_json(directory / "result.json", result)
    print(f"FIT DONE {dataset} {variant} seed={seed} best={checkpoint.epoch}; test withheld", flush=True)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=("MNIST", "CiteSeer"))
    parser.add_argument("--variants", choices=VARIANTS, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-configs-only", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        config = json.loads(args.config.read_text())
        if args.dataset is not None and args.dataset != config["dataset"]:
            parser.error("--dataset conflicts with --config")
    else:
        config = primary_config(args.dataset or "MNIST")
        config["variants"] = list(VARIANTS)
    for field in ("seeds", "epochs", "variants", "diagnostics"):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    if args.full_data:
        config["subset_sizes"] = None
    if config["rank"] != 2 or config["num_prompts"] != 20 or config["device"] != "cpu":
        parser.error("This study locks rank=2, num_prompts=20, device=cpu")
    if args.threads < 1 or config["epochs"] < 1 or not config["seeds"] or not config["variants"]:
        parser.error("Positive threads/epochs and nonempty seeds/variants required")
    if len(set(config["variants"])) != len(config["variants"]) or len(set(config["seeds"])) != len(config["seeds"]):
        parser.error("Duplicate variants or seeds")
    if any(variant not in VARIANTS for variant in config["variants"]):
        parser.error("Unknown variant")
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance(output)
    for dataset in ("MNIST", "CiteSeer"):
        write_json(output / "configs" / f"{dataset}_main_pilot.json", primary_config(dataset))
        write_json(output / "configs" / f"{dataset}_main_full.json", primary_config(dataset, full=True))
    write_json(output / "protocol.json", dict(config=config, primary_variant="main",
        selection="All variants predeclared; lowest validation CE checkpoint, earlier epoch on ties; no variant screening",
        comparisons={"main-original": "whole recipe versus original shared LoRAP; capacity differs",
                     "main-none": "whole recipe versus no aggregate prompt; input GPF remains",
                     "main-nonzero_start": "zero versus nonzero left-factor initialization",
                     "main-raw_conditions": "standardized/clipped versus raw conditions",
                     "main-no_conditions": "quantization conditioning; no-condition model has 85 fewer parameters",
                     "spectral_basis-main": "spectral versus random orthogonal right-factor initialization"},
        calibration="training nodes/graphs only; main requires only quantized aggregate/context statistics",
        note="New seeds check initialization robustness; the dataset split is reused and is not a new held-out test set"))
    if args.write_configs_only:
        return
    data, manifest = load_data(config)
    write_json(output / "split_manifest.json", manifest)
    pending, results = [], []
    for seed in config["seeds"]:
        base = bare_model(config["dataset"], seed, data if config["dataset"] == "CiteSeer" else None)
        calibration = calibrate_uniform(base, config["dataset"], data, seed,
                                         include_spectral="spectral_basis" in config["variants"],
                                         max_nodes=config["calibration_max_nodes"],
                                         max_graphs=config["calibration_max_graphs"])
        path = output / f"calibration_seed_{seed}"
        path.mkdir()
        torch.save(calibration, path / "calibration.pt")
        write_json(path / "calibration.json", calibration_json(calibration))
        for variant in config["variants"]:
            directory = output / variant / f"seed_{seed}"
            model, result = fit_uniform(config, variant, seed, data, calibration, directory)
            results.append(result)
            pending.append((model, result, directory))
            write_json(output / "results.json", results)
    # Persist the complete pre-test state before any test evaluation.
    write_json(output / "pre_test_results.json", results)
    # All methods and all runs are fixed before looking at any new test metric.
    for model, result, directory in pending:
        final_test(model, result, config, data, directory)
    write_json(output / "results.json", results)
    from .uniform_report import generate_report
    generate_report(output)


if __name__ == "__main__":
    main()

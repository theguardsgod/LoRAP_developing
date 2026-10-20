"""CUDA-only, paired no-prompt / main / LoRA joint-QAT experiments."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .data import load_data, loaders_for
from .diagnostics import QuantizationObserver
from .full_config import training_config
from .full_models import build_full_backbone, load_full_data
from .full_run import atomic_save, cpu_tree, get_rng, restore_rng, sha_config, train_gpu_epoch
from .full_validation import validation_reproduction, verify_checkpoint
from .improved import calibration_json
from .lora import (DATASETS, VARIANTS, baseline_config, baseline_counts, baseline_optimizer,
                   build_baseline, common_backbone_digest, validate_lora)
from .models import state_digest
from .run import evaluate, provenance, seed_training, write_json
from .uniform import calibrate_uniform


def require_device(name):
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LoRA experiments require CUDA; CPU fallback is disabled")
    torch.cuda.set_device(device)
    torch.ones(1, device=device).sum().item()
    return device


def study_data(config):
    data, manifest = load_full_data(config) if config["task"] == "node" else load_data(config)
    return data, dict(manifest, dataset=config["dataset"])


def calibration_for(root, config, seed, data):
    if "main" not in config["variants"]:
        return None
    path = root / config["dataset"] / f"calibration_seed_{seed}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    model = build_full_backbone(config, seed, data)
    calibration = calibrate_uniform(model, config["family"], data, seed,
                                    include_spectral=False, max_nodes=4096, max_graphs=128)
    atomic_save(calibration, path)
    write_json(path.with_suffix(".json"), calibration_json(calibration))
    return calibration


def fit(root, config, variant, seed, data, calibration, device):
    directory = root / config["dataset"] / variant / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config_hash = sha_config(config)
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["config_sha256"] != config_hash:
            raise ValueError("Completed configuration changed")
        return result
    model = build_baseline(config, variant, seed, data, calibration)
    initial = dict(common_backbone_sha256=common_backbone_digest(model),
                   initial_state_sha256=state_digest(model, "all"), **baseline_counts(model))
    initial_path = directory / "initialization.json"
    if initial_path.exists() and json.loads(initial_path.read_text()) != initial:
        raise ValueError("Initialization changed on resume")
    write_json(initial_path, initial)
    write_json(directory / "config.json", dict(config, variant=variant, seed=seed))
    model.to(device)
    optimizer = baseline_optimizer(model, config)
    write_json(directory / "optimizer_groups.json", [
        dict(name=g["name"], lr=g["lr"], weight_decay=g["weight_decay"],
             parameters=sum(p.numel() for p in g["params"])) for g in optimizer.param_groups])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=5, threshold=.0001)
        if config["task"] == "graph" else None)
    loaders = loaders_for(data, config, seed) if config["task"] == "graph" else None
    gpu_data = data.clone().to(device) if loaders is None else None
    train, val = (loaders["train"], loaders["val"]) if loaders else (gpu_data, gpu_data)
    legacy = training_config(config)
    seed_training(seed + 300_000)
    history, best, elapsed = [], None, 0.
    peak_memory = 0
    last_path = directory / "last.pt"
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["config_sha256"] != config_hash:
            raise ValueError("Resume configuration changed")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(last["scheduler"])
        history, best, elapsed = last["history"], last["best"], last["training_seconds"]
        peak_memory = last["peak_cuda_memory_bytes"]
        restore_rng(last["rng"], loaders, device)
        atomic_save(best, directory / "best.pt")
        (directory / "epochs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        training = train_gpu_epoch(model, train, optimizer, config, device)
        validation = evaluate(model, val, legacy, device, "val")
        if not torch.isfinite(torch.tensor(validation["loss"])):
            raise FloatingPointError("Non-finite validation CE")
        selected = best is None or validation["loss"] < best["validation_loss"]
        if selected:
            best = dict(epoch=epoch, validation_loss=validation["loss"], model=cpu_tree(model.state_dict()))
        if scheduler is not None:
            scheduler.step(validation["loss"])
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        elapsed += seconds
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        history.append(dict(epoch=epoch, train=training, validation=validation, selected=selected,
                            seconds=seconds, learning_rates={g["name"]: g["lr"] for g in optimizer.param_groups}))
        atomic_save(dict(config_sha256=config_hash, epoch=epoch, model=cpu_tree(model.state_dict()),
                         optimizer=cpu_tree(optimizer.state_dict()),
                         scheduler=scheduler.state_dict() if scheduler else None,
                         history=history, best=best, rng=get_rng(loaders, device), training_seconds=elapsed,
                         peak_cuda_memory_bytes=peak_memory), last_path)
        if selected:
            atomic_save(best, directory / "best.pt")
        with (directory / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(history[-1], allow_nan=False) + "\n")
        write_json(directory / "progress.json", dict(epoch=epoch, epochs=config["epochs"],
                   best_epoch=best["epoch"], validation=validation, training_seconds=elapsed))
        if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
            print(f"{config['dataset']} {variant} seed={seed} epoch={epoch}/{config['epochs']} "
                  f"val_CE={validation['loss']:.5f} val_acc={validation['accuracy']:.4f}", flush=True)
    verify_checkpoint(directory, best, config_hash)
    model.load_state_dict(best["model"])
    with QuantizationObserver(model) as observer:
        checked = evaluate(model, val, legacy, device, "val")
    replay = validation_reproduction(best["validation_loss"], checked["loss"])
    quantization = observer.result()
    write_json(directory / "quantization.json", quantization)
    result = dict(dataset=config["dataset"], variant=variant, seed=seed, epochs=config["epochs"],
                  config_sha256=config_hash, best_epoch=best["epoch"], validation=checked,
                  validation_reproduction=replay, test=None, test_evaluations=0,
                  training_seconds=elapsed, peak_cuda_memory_bytes=peak_memory,
                  device=str(device), gpu_name=torch.cuda.get_device_name(device),
                  physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  mean_activation_effective_bits=quantization["mean_activation_effective_bits"], **initial)
    write_json(result_path, result)
    print(f"FIT DONE {config['dataset']} {variant} seed={seed}; test withheld", flush=True)
    return result


def release_tests(root, config):
    """Validate every predeclared fit before allowing any test in this dataset."""
    results = []
    for seed in config["seeds"]:
        paired = []
        for variant in config["variants"]:
            directory = root / config["dataset"] / variant / f"seed_{seed}"
            result = json.loads((directory / "result.json").read_text())
            if result["config_sha256"] != sha_config(config):
                raise ValueError("Test release configuration mismatch")
            if result["epochs"] != config["epochs"]:
                raise ValueError("Incomplete fit")
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            verify_checkpoint(directory, checkpoint, sha_config(config))
            paired.append(result["common_backbone_sha256"])
            results.append(result)
        if len(set(paired)) != 1:
            raise AssertionError("Unpaired initial backbone")
    write_json(root / config["dataset"] / "test_release.json",
               dict(config_sha256=sha_config(config), fits=len(results), variants=config["variants"], seeds=config["seeds"]))
    return results


def test_once(root, config, variant, seed, data, calibration, device):
    directory = root / config["dataset"] / variant / f"seed_{seed}"
    result = json.loads((directory / "result.json").read_text())
    if result["config_sha256"] != sha_config(config):
        raise ValueError("Test configuration changed")
    release = root / config["dataset"] / "test_release.json"
    if not release.exists() or json.loads(release.read_text())["config_sha256"] != sha_config(config):
        raise RuntimeError("Dataset tests have not been released")
    if result["test_evaluations"]:
        return result
    marker = directory / "test_started.json"
    if marker.exists():
        raise RuntimeError(f"Interrupted test requires manual inspection: {directory}")
    model = build_baseline(config, variant, seed, data, calibration).to(device)
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    verify_checkpoint(directory, checkpoint, sha_config(config))
    model.load_state_dict(checkpoint["model"])
    evaluation_data = (data.clone().to(device) if config["task"] == "node" else
                       loaders_for(data, config, seed)["test"])
    # Exclusive creation ensures concurrent callers cannot test twice.
    with marker.open("x") as stream:
        json.dump(dict(started_unix=time.time(), checkpoint_epoch=checkpoint["epoch"]), stream)
    result["test"] = evaluate(model, evaluation_data, training_config(config), device, "test")
    result["test_evaluations"] = 1
    write_json(directory / "result.json", result)
    print(f"TEST {config['dataset']} {variant} seed={seed}: {result['test']['accuracy']:.4f}", flush=True)
    return result


def worker(root):
    import fcntl
    from .lora_report import generate_report

    study = json.loads((root / "study.json").read_text())
    torch.set_num_threads(study["threads"])
    device = require_device(study["device"])
    # One owner of checkpoints and final test release per output directory.
    with (root / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for config in study["configs"]:
            data, manifest = study_data(config)
            write_json(root / config["dataset"] / "split_manifest.json", manifest)
            for seed in config["seeds"]:
                calibration = calibration_for(root, config, seed, data)
                paired = []
                for variant in config["variants"]:
                    result = fit(root, config, variant, seed, data, calibration, device)
                    paired.append(result["common_backbone_sha256"])
                    if len(set(paired)) != 1:
                        raise AssertionError("Unpaired initial backbone")
                    generate_report(root)
            if not study["smoke"]:
                release_tests(root, config)
                for seed in config["seeds"]:
                    calibration = calibration_for(root, config, seed, data)
                    for variant in config["variants"]:
                        test_once(root, config, variant, seed, data, calibration, device)
                        generate_report(root)
        generate_report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--node-epochs", type=int, default=200)
    parser.add_argument("--graph-epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=2.)
    parser.add_argument("--quantization", choices=("a2q", "fixed4"), default="a2q")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", help="Override dataset location when selecting one dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Two epochs, first seed, small graph subset; no tests")
    parser.add_argument("--full-data", action="store_true", help="Use complete official graph splits; set epochs and seeds explicitly")
    parser.add_argument("--resume", action="store_true", help="Resume the saved protocol and source snapshot")
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        worker(args.worker_root)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    root = args.output_dir.resolve()
    if args.resume:
        if not (root / "study.json").exists():
            parser.error("No saved study to resume")
    else:
        validate_lora(args.lora_rank, args.lora_alpha)
        if args.smoke and args.full_data:
            parser.error("--smoke and --full-data are mutually exclusive")
        if min(args.node_epochs, args.graph_epochs, args.threads) < 1:
            parser.error("Epochs and threads must be positive")
        for key in ("datasets", "variants", "seeds"):
            values = getattr(args, key)
            if len(values) != len(set(values)):
                parser.error(f"Duplicate {key}")
        if args.data_root and len(args.datasets) != 1:
            parser.error("--data-root requires exactly one dataset")
        if args.quantization == "fixed4" and args.datasets != ["MNIST"]:
            parser.error("--quantization fixed4 requires --datasets MNIST")
        require_device(args.device)
        torch.set_num_threads(args.threads)
        root.mkdir(parents=True, exist_ok=False)
        configs = []
        for name in args.datasets:
            config = baseline_config(name, args.quantization)
            config.update(variants=args.variants, seeds=args.seeds[:1] if args.smoke else args.seeds,
                          device=args.device, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                          epochs=2 if args.smoke else (args.graph_epochs if name == "MNIST" else args.node_epochs))
            if args.smoke and name == "MNIST":
                config["subset_sizes"] = dict(train=128, val=64, test=64)
            elif args.full_data and name == "MNIST":
                config.update(subset_sizes=None, profile=config["profile"] + "_full")
            if args.data_root:
                config["data_root"] = args.data_root
            configs.append(config)
            write_json(root / name / "config.json", config)
        write_json(root / "study.json", dict(configs=configs, device=args.device, threads=args.threads,
                   smoke=args.smoke, method="LoRA joint-QAT", protocol_version=1))
        provenance(root)
        (root / "RUN_COMMANDS.md").write_text(
            "Resume the immutable saved protocol (other CLI settings are ignored):\n\n"
            f"`{sys.executable} -B -m runs.lorap_study.lora_run --resume --output-dir {root}`\n")
    print(f"OUTPUT_DIRECTORY={root}", flush=True)
    # All fits execute exactly the code whose hashes were captured at launch.
    command = [sys.executable, "-B", "-m", "runs.lorap_study.lora_run", "--worker-root", str(root)]
    subprocess.run(command, cwd=root / "source_snapshot", check=True)


if __name__ == "__main__":
    main()

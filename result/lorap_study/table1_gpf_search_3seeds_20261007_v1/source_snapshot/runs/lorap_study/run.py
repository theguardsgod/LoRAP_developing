"""Run with python -m runs.lorap_study.run --profile pilot --dataset both."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import DATASETS, VARIANTS, dataset_config
from .data import load_data, loaders_for
from .diagnostics import AggregationDiagnostics, QuantizationObserver
from .models import build_model, model_counts, optimizer_for, state_digest

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BestCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.loss, self.epoch = float("inf"), None

    def consider(self, loss, epoch, model, optimizer=None):
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        if loss >= self.loss:
            return False
        self.loss, self.epoch = loss, epoch
        state = dict(model=model.state_dict(), epoch=epoch, validation_loss=loss)
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, self.path)
        return True


def require_finite(value, description):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(description)


def train_epoch(model, data, optimizer, config, device):
    model.train()
    loss_sum, correct, count, budget_sum, batches = 0., 0, 0, 0., 0
    is_node = config["dataset"] == "CiteSeer"
    for batch in [data] if is_node else data:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, budget = model(batch)
        labels = batch.y if is_node else batch.y.reshape(-1)
        if is_node:
            logits, labels = logits[batch.train_mask], labels[batch.train_mask]
        task_loss = F.cross_entropy(logits, labels)
        store_loss = config["a_loss"] * F.relu(budget - config["a_storage"]).square()
        require_finite(logits, "Non-finite training logits")
        require_finite(task_loss + store_loss, "Non-finite training objective")
        (task_loss + store_loss).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                require_finite(parameter.grad, f"Non-finite gradient: {name}")
        optimizer.step()
        loss_sum += float(task_loss.detach()) * labels.numel()
        correct += int((logits.argmax(1) == labels).sum())
        count += labels.numel()
        budget_sum += float(budget.detach())
        batches += 1
    return dict(loss=loss_sum / count, accuracy=correct / count,
                legacy_budget_proxy=budget_sum / batches)


@torch.no_grad()
def evaluate(model, data, config, device, split, diagnostics=None):
    model.eval()
    loss_sum, correct, count, budget_sum, batches = 0., 0, 0, 0., 0
    is_node = config["dataset"] == "CiteSeer"
    for batch in [data] if is_node else data:
        batch = batch.to(device)
        mask = getattr(batch, f"{split}_mask") if is_node else None
        logits, budget = model(batch, diagnostics=diagnostics, node_mask=mask)
        labels = batch.y if is_node else batch.y.reshape(-1)
        if is_node:
            logits, labels = logits[mask], labels[mask]
        require_finite(logits, "Non-finite evaluation logits")
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(1) == labels).sum())
        count += labels.numel()
        budget_sum += float(budget)
        batches += 1
    return dict(loss=loss_sum / count, accuracy=correct / count, examples=count,
                legacy_budget_proxy=budget_sum / batches)


def run_one(config, variant, seed, dataset, directory, device):
    directory.mkdir(parents=True, exist_ok=False)
    effective = dict(config, variant=variant, seed=seed, device=str(device))
    write_json(directory / "config.json", effective)
    model = build_model(config["dataset"], variant, seed,
                        data=dataset if config["dataset"] == "CiteSeer" else None,
                        bit=config["bit"], num_prompts=config["num_prompts"], rank=config["rank"])
    initial = dict(backbone_sha256=state_digest(model), prompt_base_sha256=state_digest(model, "prompt_base"),
                   **model_counts(model))
    write_json(directory / "initialization.json", initial)
    model = model.to(device)
    optimizer = optimizer_for(model, config)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                          patience=5, threshold=0.0001)
                 if config["dataset"] == "MNIST" else None)
    if config["dataset"] == "CiteSeer":
        train, validation, test = dataset, dataset, dataset
    else:
        loaders = loaders_for(dataset, config, seed)
        train, validation, test = (loaders[split] for split in ("train", "val", "test"))
    checkpoint = BestCheckpoint(directory / "best.pt")
    seed_training(seed + 300_000)
    start = time.perf_counter()
    with (directory / "epochs.jsonl").open("w") as log:
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            training = train_epoch(model, train, optimizer, config, device)
            val = evaluate(model, validation, config, device, "val")
            selected = checkpoint.consider(val["loss"], epoch, model, optimizer)
            if scheduler is not None:
                scheduler.step(val["loss"])
            row = dict(epoch=epoch, train=training, validation=val, selected=selected,
                       seconds=time.perf_counter() - epoch_start,
                       learning_rates={g["name"]: g["lr"] for g in optimizer.param_groups})
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
                print(f"{config['dataset']} {variant} seed={seed} epoch={epoch}/{config['epochs']} "
                      f"train={training['loss']:.4f} val={val['loss']:.4f} "
                      f"val_acc={val['accuracy']:.4f} seconds={row['seconds']:.2f}", flush=True)
    training_seconds = time.perf_counter() - start
    selected = torch.load(checkpoint.path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    # Test labels are first consumed here, after the validation-selected state
    # has been reloaded. Diagnostics never select or tune a checkpoint.
    test_metrics = evaluate(model, test, config, device, "test")
    collector = AggregationDiagnostics(config["split_seed"], config["diagnostic_max_nodes"]) if config["diagnostics"] else None
    with QuantizationObserver(model) as observer:
        validation_metrics = evaluate(model, validation, config, device, "val", collector)
    if collector is not None:
        write_json(directory / "diagnostics.json", collector.result())
    write_json(directory / "quantization.json", observer.result())
    result = dict(dataset=config["dataset"], variant=variant, seed=seed,
                  profile=config["profile"], epochs=config["epochs"], best_epoch=checkpoint.epoch,
                  validation=validation_metrics, test=test_metrics, test_evaluations=1,
                  training_seconds=training_seconds, total_seconds=time.perf_counter() - start,
                  mean_activation_effective_bits=observer.result()["mean_activation_effective_bits"],
                  **initial)
    write_json(directory / "result.json", result)
    print(f"COMPLETED {config['dataset']} {variant} seed={seed}: "
          f"test={test_metrics['accuracy']:.4f}, best_epoch={checkpoint.epoch}", flush=True)
    return result


def provenance(output):
    patterns = ("runs/lorap_study/*.py", "prompt/*.py", "quantize_function/*.py",
                "quantize_method/*.py", "utils/*.py")
    hashes = {}
    for pattern in patterns:
        for source in sorted(ROOT.glob(pattern)):
            relative = source.relative_to(ROOT)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = output / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    versions = {}
    for package in ("torch", "torch-geometric", "torch-scatter", "torch-sparse", "numpy", "scikit-learn", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    write_json(output / "provenance.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
               python=sys.executable, python_version=platform.python_version(), platform=platform.platform(),
               versions=versions, cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
               torch_threads=torch.get_num_threads(), argv=sys.argv, source_sha256=hashes))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--dataset", choices=(*DATASETS, "both"), default="both")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=str, help="Override root for a single dataset")
    parser.add_argument("--config", type=Path, help="Load a generated single-dataset configuration")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-configs-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("threads and epochs must be positive")
    if args.data_root and args.dataset == "both" and not args.config:
        parser.error("--data-root requires a single dataset")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available() and not args.write_configs_only:
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly or run in a GPU environment")
    if args.config:
        configs = [json.loads(args.config.read_text())]
    else:
        configs = [dataset_config(name, args.profile) for name in DATASETS
                   if args.dataset in (name, "both")]
    for config in configs:
        for field in ("epochs", "seeds", "data_root", "diagnostics"):
            value = getattr(args, field)
            if value is not None:
                config[field] = value
    output = (args.output_dir or ROOT / "result" / "lorap_study" /
              (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance(output)
    for name in DATASETS:
        write_json(output / "full_configs" / f"{name}.json", dataset_config(name, "full"))
    write_json(output / "study.json", dict(configs=configs, variants=list(VARIANTS) if args.variant == "all" else [args.variant]))
    print(f"OUTPUT_DIRECTORY={output}", flush=True)
    if args.write_configs_only:
        return
    results = []
    for config in configs:
        dataset, manifest = load_data(config)
        write_json(output / config["dataset"] / "split_manifest.json", manifest)
        for seed in config["seeds"]:
            paired_digest, paired_prompt = None, None
            for variant in VARIANTS if args.variant == "all" else (args.variant,):
                directory = output / config["dataset"] / variant / f"seed_{seed}"
                try:
                    result = run_one(config, variant, seed, dataset, directory, device)
                except Exception as error:
                    write_json(directory / "failure.json", dict(error=repr(error)))
                    raise
                if paired_digest is not None and paired_digest != result["backbone_sha256"]:
                    raise AssertionError("Paired backbone initialization mismatch")
                paired_digest = result["backbone_sha256"]
                if variant in ("original", "conditioned"):
                    if paired_prompt is not None and paired_prompt != result["prompt_base_sha256"]:
                        raise AssertionError("Paired prompt initialization mismatch")
                    paired_prompt = result["prompt_base_sha256"]
                results.append(result)
                write_json(output / "results.json", results)
    from .report import generate_report
    generate_report(output)


if __name__ == "__main__":
    main()

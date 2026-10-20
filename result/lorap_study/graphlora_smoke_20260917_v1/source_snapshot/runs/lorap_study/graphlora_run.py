"""Paired MNIST GraphLoRA-adapted joint-QAT; pilot is validation-only."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .data import loaders_for
from .diagnostics import QuantizationObserver
from .full_config import training_config
from .full_run import atomic_save, cpu_tree, get_rng, restore_rng, sha_config, train_gpu_epoch
from .full_validation import validation_reproduction, verify_checkpoint
from .graphlora import (GraphLoRAAdaptedGIN, backbone_digest, build_graphlora,
                       graphlora_optimizer, model_counts, training_objective)
from .lora_run import release_tests, require_device, study_data
from .models import state_digest
from .run import ROOT, evaluate, provenance, seed_training, write_json
from .table_v_run import table_config


def read(path):
    return json.loads(Path(path).read_text())


class StopRequested(Exception):
    pass


def study_config(mode, seeds, epochs, profile, rank=2, contrast_weight=.1, structure_weight=.1):
    config = table_config(mode, seeds, epochs, smoke=profile == 'smoke')
    config.update(variants=['no_prompt', 'graphlora'], profile='graphlora_adapted_' + profile,
        graphlora_rank=rank, contrast_weight=contrast_weight, structure_weight=structure_weight,
        contrast_tau=.5, contrast_supervised_weight=.2, smmd_weight=0., cross_graph_source=None)
    return config


def train_graph_epoch(model, training, optimizer, config, device):
    if not isinstance(model, GraphLoRAAdaptedGIN):
        return train_gpu_epoch(model, training, optimizer, config, device)
    if device.type != 'cuda':
        raise RuntimeError('GraphLoRA experiments require CUDA')
    model.train()
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    examples = batches = 0
    for batch in training:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        objective, logits, metrics = training_objective(model, batch, config)
        if not bool(torch.isfinite(objective).all()):
            raise FloatingPointError('Non-finite GraphLoRA objective')
        objective.backward()
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None])
        if not bool(finite.all()):
            raise FloatingPointError('Non-finite GraphLoRA gradient')
        optimizer.step()
        size = batch.y.numel()
        totals += torch.stack((metrics['loss'].detach().double() * size,
            (logits.argmax(1) == batch.y.reshape(-1)).sum().double(), metrics['budget'].detach().double().reshape(()),
            metrics['contrast'].detach().double() * size, metrics['structure'].detach().double() * size,
            metrics['storage'].detach().double().reshape(()) * size))
        examples += size; batches += 1
    values = totals.cpu().tolist()
    return dict(loss=values[0]/examples, accuracy=values[1]/examples, legacy_budget_proxy=values[2]/batches,
        contrast_loss=values[3]/examples, structure_loss=values[4]/examples, storage_loss=values[5]/examples,
        examples=examples)


# The fit/checkpoint implementation below preserves lora_run's established
# training order, scheduler, earliest-minimum selection and one-shot test rules.


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
    model = build_graphlora(config, variant, seed, data)
    initial = dict(common_backbone_sha256=backbone_digest(model),
                   initial_state_sha256=state_digest(model, "all"), **model_counts(model))
    initial_path = directory / "initialization.json"
    if initial_path.exists() and json.loads(initial_path.read_text()) != initial:
        raise ValueError("Initialization changed on resume")
    write_json(initial_path, initial)
    write_json(directory / "config.json", dict(config, variant=variant, seed=seed))
    model.to(device)
    optimizer = graphlora_optimizer(model, config)
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
        if (root.parent / 'STOP').exists():
            raise StopRequested('Stopped at a committed epoch boundary')
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        training = train_graph_epoch(model, train, optimizer, config, device)
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


def worker(root):
    study = read(root/'study.json')
    torch.set_num_threads(study['threads'])
    device = require_device(study['device'])
    configs = study['configs']
    with (root/'.worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            data, manifest = study_data(configs[0])
            for config in configs:
                path = root/config['quantization_mode']/'MNIST/split_manifest.json'
                if path.exists():
                    assert read(path) == manifest, 'Dataset split changed'
                write_json(path, manifest)
            for seed in configs[0]['seeds']:
                for config in configs:
                    fingerprints, base_groups = [], []
                    mode = config['quantization_mode']
                    for variant in config['variants']:
                        write_json(root/'active.json', dict(phase='training', mode=mode, variant=variant, seed=seed))
                        result = fit(root/mode, config, variant, seed, data, None, device)
                        fingerprints.append(result['common_backbone_sha256'])
                        groups = read(root/mode/'MNIST'/variant/f'seed_{seed}'/'optimizer_groups.json')
                        base_groups.append([g for g in groups if g['name'] != 'graphlora'])
                        assert result['prompt_parameters'] == 0
                    assert len(set(fingerprints)) == 1 and all(g == base_groups[0] for g in base_groups)
                    write_json(root/mode/f'pairing_seed_{seed}.json', dict(passed=True,
                        common_backbone_sha256=fingerprints[0], base_optimizer_groups=base_groups[0]))
            if study['profile'] == 'full':
                for config in configs:
                    release_tests(root/config['quantization_mode'], config)
                write_json(root/'TEST_RELEASE.json', dict(configs={c['quantization_mode']: sha_config(c) for c in configs},
                                                         released_unix=time.time()))
                for seed in configs[0]['seeds']:
                    for config in configs:
                        for variant in config['variants']:
                            write_json(root/'active.json', dict(phase='test', mode=config['quantization_mode'], variant=variant, seed=seed))
                            test_once(root/config['quantization_mode'], config, variant, seed, data, None, device)
            results = [read(root/c['quantization_mode']/'MNIST'/v/f'seed_{s}'/'result.json')
                       for c in configs for s in c['seeds'] for v in c['variants']]
            assert all(r['test_evaluations'] == (1 if study['profile'] == 'full' else 0) for r in results)
            write_json(root/'completion_audit.json', dict(passed=True, profile=study['profile'],
                fits_completed=len(results), paired_initializations_and_optimizers=True,
                minimum_validation_checkpoints_verified=True, test_evaluations_per_model=1 if study['profile']=='full' else 0))
            write_json(root/'active.json', dict(phase='complete'))
        except StopRequested as error:
            write_json(root/'active.json', dict(phase='stopped', reason=str(error), stopped_unix=time.time()))


def coordinator(root):
    from .graphlora_report import generate_report
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, digest in read(root/'provenance.json')['source_sha256'].items():
            assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest() == digest, name
        with (root/'worker.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, '-B', '-m', 'runs.lorap_study.graphlora_run',
                '--worker-root', str(root)], cwd=root/'source_snapshot', stdout=stream, stderr=subprocess.STDOUT)
            write_json(root/'worker.json', dict(pid=process.pid, started_unix=time.time()))
            try:
                while process.poll() is None:
                    generate_report(root)
                    time.sleep(10)
                if process.returncode:
                    write_json(root/'active.json', dict(phase='failed', exit_code=process.returncode))
                generate_report(root)
                if process.returncode:
                    raise RuntimeError(f'GraphLoRA worker failed ({process.returncode}); inspect worker.log')
            finally:
                if process.poll() is None:process.terminate()
                process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--profile', choices=('smoke', 'pilot', 'full'), default='pilot')
    parser.add_argument('--modes', choices=('fixed4', 'a2q'), nargs='+', default=['fixed4', 'a2q'])
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--contrast-weight', type=float, default=.1)
    parser.add_argument('--structure-weight', type=float, default=.1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--coordinator-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:worker(args.worker_root);return
    if args.coordinator_root:coordinator(args.coordinator_root);return
    if args.output_dir is None:parser.error('--output-dir is required')
    root = args.output_dir.resolve()
    if not args.resume:
        epochs = args.epochs if args.epochs is not None else dict(smoke=2, pilot=30, full=200)[args.profile]
        seeds = args.seeds or (list(range(41, 51)) if args.profile == 'full' else [41])
        if min(epochs, args.rank, args.threads) < 1:parser.error('Epochs, rank and threads must be positive')
        if len(set(seeds)) != len(seeds) or len(set(args.modes)) != len(args.modes):parser.error('Duplicate seeds or modes')
        if any(not math.isfinite(w) or w < 0 for w in (args.contrast_weight, args.structure_weight)):
            parser.error('Auxiliary weights must be finite and nonnegative')
        require_device(args.device); torch.set_num_threads(args.threads)
        root.mkdir(parents=True, exist_ok=False)
        configs = [study_config(m, seeds, epochs, args.profile, args.rank, args.contrast_weight, args.structure_weight) for m in args.modes]
        write_json(root/'study.json', dict(configs=configs, profile=args.profile, device=args.device,
            threads=args.threads, method='GraphLoRA-adapted joint-QAT', protocol_version=1))
        for config in configs:write_json(root/config['quantization_mode']/'MNIST/config.json', config)
        provenance(root)
        recorded = read(root/'provenance.json')
        for folder in ('graphlora_official', 'glora_official'):
            destination = root/'source_snapshot/runs/lorap_study'/folder
            shutil.copytree(ROOT/'runs/lorap_study'/folder, destination)
            for path in destination.rglob('*'):
                if path.is_file():recorded['source_sha256'][str(path.relative_to(root/'source_snapshot'))] = hashlib.sha256(path.read_bytes()).hexdigest()
        notes = ROOT/'runs/lorap_study/GRAPHLORA_METHOD.md'
        destination = root/'source_snapshot/runs/lorap_study/GRAPHLORA_METHOD.md'
        shutil.copy2(notes, destination)
        recorded['source_sha256'][str(destination.relative_to(root/'source_snapshot'))] = hashlib.sha256(destination.read_bytes()).hexdigest()
        recorded['method_source'] = read(ROOT/'runs/lorap_study/graphlora_official/manifest.json')
        write_json(root/'provenance.json', recorded)
        (root/'RUN_COMMANDS.md').write_text(f'{sys.executable} -B -m runs.lorap_study.graphlora_run --resume --output-dir {root}\n'
            '\n创建结果根目录 STOP 文件将在当前轮提交断点后暂停；--resume 会删除 STOP 并恢复。\n')
    else:
        if not (root/'study.json').exists():parser.error('No saved experiment to resume')
        (root/'STOP').unlink(missing_ok=True)
    subprocess.run([sys.executable, '-B', '-m', 'runs.lorap_study.graphlora_run', '--coordinator-root', str(root)],
                   cwd=root/'source_snapshot', check=True)


def test_once(root, config, variant, seed, data, calibration, device):
    barrier = read(root.parent / "TEST_RELEASE.json")
    assert barrier["configs"][config["quantization_mode"]] == sha_config(config)
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
    model = build_graphlora(config, variant, seed, data).to(device)
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


if __name__ == '__main__':main()

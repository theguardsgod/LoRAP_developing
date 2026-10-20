"""A resumable CUDA-only full-split fit or one final test evaluation."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .data import loaders_for
from .diagnostics import AggregationDiagnostics, QuantizationObserver
from .full_config import training_config, validate
from .full_models import build_full_backbone, load_full_data
from .full_validation import validation_reproduction, verify_checkpoint
from .improved import FullUpdateDiagnostics, calibration_json
from .models import model_counts, optimizer_for, state_digest
from .run import evaluate, seed_training, write_json
from .uniform import calibrate_uniform, install_uniform


def sha_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training required: CUDA is unavailable; CPU fallback is disabled")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Also catches driver/runtime initialization failures before fitting.
    torch.ones(1, device=device).sum().item()
    return device


def calibration_for(root, config, seed, data):
    directory = root/config["dataset"]/f"calibration_seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory/"calibration.pt"
        if path.exists():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            if saved["config_sha256"] != sha_config(config):
                raise ValueError("Calibration configuration changed")
            return saved["calibration"]
        model = build_full_backbone(config, seed, data)
        calibration = calibrate_uniform(model, config["family"], data, seed,
            include_spectral="spectral_basis" in config["variants"],
            max_nodes=config["calibration_max_nodes"], max_graphs=config["calibration_max_graphs"])
        atomic_save(dict(config_sha256=sha_config(config), calibration=calibration), path)
        write_json(directory/"calibration.json", calibration_json(calibration))
        return calibration


def get_rng(loaders, device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=="cuda" else None,
                loaders={name: loader.generator.get_state() for name, loader in (loaders or {}).items()})


def restore_rng(state, loaders, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type=="cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    for name, generator_state in state["loaders"].items():
        loaders[name].generator.set_state(generator_state)


def train_gpu_epoch(model, training, optimizer, config, device):
    if device.type != "cuda":
        raise RuntimeError("Full training must use CUDA")
    model.train()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    count = batches = 0
    for batch in [training] if config["task"]=="node" else training:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, budget = model(batch)
        labels = batch.y.reshape(-1)
        if config["task"]=="node":
            logits, labels = logits[batch.train_mask], labels[batch.train_mask]
        loss = F.cross_entropy(logits, labels)
        objective = loss+config["a_loss"]*F.relu(budget-config["a_storage"]).square()
        if not bool(torch.isfinite(objective).all()):
            raise FloatingPointError("Non-finite training objective")
        objective.backward()
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None])
        if not bool(finite.all()):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        totals += torch.stack((loss.detach().double()*labels.numel(),
                               (logits.argmax(1)==labels).sum().double(), budget.detach().double().reshape(())))
        count += labels.numel()
        batches += 1
    values = totals.cpu().tolist()
    return dict(loss=values[0]/count, accuracy=values[1]/count,
                legacy_budget_proxy=values[2]/batches, examples=count)


def fit(root, config, variant, seed, data, device):
    directory = root/config["dataset"]/variant/f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config_hash = sha_config(config)
    if (directory/"result.json").exists():
        result = json.loads((directory/"result.json").read_text())
        if result["config_sha256"] != config_hash:
            raise ValueError("Completed run configuration changed")
        return result
    calibration = calibration_for(root, config, seed, data)
    model = build_full_backbone(config, seed, data)
    ranks = install_uniform(model, config["family"], seed, calibration, variant)
    initial = dict(backbone_sha256=state_digest(model), initial_state_sha256=state_digest(model,"all"),
                   ranks=ranks, calibration_buffer_bytes=sum(b.numel()*b.element_size() for b in model.prompt_aggr.buffers()),
                   **model_counts(model))
    write_json(directory/"config.json", dict(config, variant=variant, seed=seed))
    if (directory/"initialization.json").exists():
        if json.loads((directory/"initialization.json").read_text()) != initial:
            raise ValueError("Initialization changed on resume")
    else:
        write_json(directory/"initialization.json", initial)
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = optimizer_for(model, config)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=5, threshold=.0001)
                 if config["task"]=="graph" else None)
    loaders = loaders_for(data, config, seed) if config["task"]=="graph" else None
    gpu_data = data.clone().to(device) if config["task"]=="node" else None
    training, validation = (loaders["train"],loaders["val"]) if loaders else (gpu_data,gpu_data)
    legacy = training_config(config)
    seed_training(seed+300_000)
    history, best, elapsed = [], None, 0.
    last_path = directory/"last.pt"
    if last_path.exists():
        last = torch.load(last_path, weights_only=False, map_location="cpu")
        if last["config_sha256"] != config_hash:
            raise ValueError("Resume configuration changed")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(last["scheduler"])
        history, best, elapsed = last["history"], last["best"], last["training_seconds"]
        restore_rng(last["rng"], loaders, device)
        atomic_save(best, directory/"best.pt")
        # last.pt is the committed source of truth after an interrupted write.
        (directory/"epochs.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history))
    for epoch in range(len(history)+1, config["epochs"]+1):
        if (root/"STOP").exists():
            raise InterruptedError("Paused at an epoch boundary; resume from last.pt")
        started = time.perf_counter()
        train = train_gpu_epoch(model, training, optimizer, config, device)
        val = evaluate(model, validation, legacy, device, "val")
        if not np.isfinite(val["loss"]):
            raise FloatingPointError("Non-finite validation loss")
        selected = best is None or val["loss"] < best["validation_loss"]
        if selected:
            best = dict(epoch=epoch, validation_loss=val["loss"], model=cpu_tree(model.state_dict()))
        if scheduler is not None:
            scheduler.step(val["loss"])
        elapsed += time.perf_counter()-started
        history.append(dict(epoch=epoch, train=train, validation=val, selected=selected,
                            learning_rates={g["name"]: g["lr"] for g in optimizer.param_groups}))
        last = dict(config_sha256=config_hash, epoch=epoch, model=cpu_tree(model.state_dict()),
                    optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict() if scheduler else None,
                    history=history, best=best, rng=get_rng(loaders,device), training_seconds=elapsed)
        atomic_save(last, last_path)
        if selected:
            atomic_save(best, directory/"best.pt")
        with (directory/"epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(history[-1],allow_nan=False)+"\n")
        write_json(directory/"progress.json",dict(phase="fit",epoch=epoch,epochs=config["epochs"],
            best_epoch=best["epoch"],validation=val,training_seconds=elapsed,device=str(device),
            physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),updated_unix=time.time()))
        if epoch==1 or epoch%10==0 or epoch==config["epochs"]:
            print(f"{config['dataset']} {variant} seed={seed} epoch={epoch}/{config['epochs']} val_CE={val['loss']:.5f}",flush=True)
    model.load_state_dict(best["model"])
    atomic_save(best, directory/"best.pt")
    verify_checkpoint(directory, best, config_hash)
    checked = evaluate(model, validation, legacy, device, "val")
    replay = validation_reproduction(best["validation_loss"], checked["loss"])
    collector = AggregationDiagnostics(config["split_seed"],config["diagnostic_max_nodes"]) if config["diagnostics"] else None
    with QuantizationObserver(model) as observer:
        val = evaluate(model, validation, legacy, device, "val", collector)
    diagnostic_replay = validation_reproduction(best["validation_loss"], val["loss"])
    write_json(directory/"diagnostics.json",collector.result() if collector else {"enabled":False})
    quantization = observer.result()
    write_json(directory/"quantization.json",quantization)
    if config["diagnostics"]:
        with FullUpdateDiagnostics(model,gpu_data.val_mask if gpu_data is not None else None) as full:
            evaluate(model,validation,legacy,device,"val")
        write_json(directory/"full_update_diagnostics.json",full.rows)
    result = dict(dataset=config["dataset"],variant=variant,seed=seed,epochs=config["epochs"],
        config_sha256=config_hash,best_epoch=best["epoch"],validation=val,test=None,test_evaluations=0,
        validation_reproduction=replay,diagnostic_validation_reproduction=diagnostic_replay,
        training_seconds=elapsed,device=str(device),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=torch.cuda.get_device_name(device),peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
        mean_activation_effective_bits=quantization["mean_activation_effective_bits"],
        **initial)
    write_json(directory/"result.json",result)
    return result


def test_once(root,config,variant,seed,data,device):
    directory=root/config["dataset"]/variant/f"seed_{seed}"
    result=json.loads((directory/"result.json").read_text())
    if result["config_sha256"] != sha_config(config):
        raise ValueError("Test configuration changed")
    if result["test_evaluations"]:
        return result
    release=root/config["dataset"]/"test_release.json"
    if not release.exists():
        raise RuntimeError("All configured fits for this dataset must finish before tests are released")
    released=json.loads(release.read_text())
    if released["config_sha256"] != sha_config(config) or len(released["fits"]) != len(config["seeds"])*len(config["variants"]):
        raise ValueError("Test release does not match the configured study")
    calibration=calibration_for(root,config,seed,data)
    model=build_full_backbone(config,seed,data)
    install_uniform(model,config["family"],seed,calibration,variant)
    if state_digest(model)!=result["backbone_sha256"] or state_digest(model,"all")!=result["initial_state_sha256"]:
        raise AssertionError("Test model initialization fingerprint changed")
    checkpoint=torch.load(directory/"best.pt",weights_only=False,map_location="cpu")
    verify_checkpoint(directory, checkpoint, sha_config(config))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    validation=data.clone().to(device) if config["task"]=="node" else loaders_for(data,config,seed)["val"]
    checked=evaluate(model,validation,training_config(config),device,"val")
    result["test_validation_reproduction"]=validation_reproduction(checkpoint["validation_loss"],checked["loss"])
    started=directory/"test_started.json"
    with started.open("x") as stream:
        json.dump(dict(started_unix=time.time(),checkpoint_epoch=checkpoint["epoch"]),stream)
    testing=data.clone().to(device) if config["task"]=="node" else loaders_for(data,config,seed)["test"]
    result["test"]=evaluate(model,testing,training_config(config),device,"test")
    result["test_evaluations"]=1
    result["test_device"]=str(device)
    result["test_physical_gpu"]=os.environ.get("CUDA_VISIBLE_DEVICES")
    write_json(directory/"result.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--variant",required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--phase",choices=("fit","test"),required=True)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    validate(config)
    if args.variant not in config["variants"] or args.seed not in config["seeds"]:
        parser.error("Job is outside the configured study")
    torch.set_num_threads(args.threads)
    device=require_cuda()
    root=args.root.resolve()
    directory=root/config["dataset"]/args.variant/f"seed_{args.seed}"
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/".job.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            data,manifest=load_full_data(config)
            expected=json.loads((root/config["dataset"]/"split_manifest.json").read_text())
            if manifest!=expected:
                raise ValueError("Official split manifest changed")
            result=(fit if args.phase=="fit" else test_once)(root,config,args.variant,args.seed,data,device)
            print(json.dumps({k:result[k] for k in ("dataset","variant","seed","best_epoch","test_evaluations")}),flush=True)
        except Exception as error:
            write_json(directory/f"{args.phase}_failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                                                                time_unix=time.time()))
            raise


if __name__=="__main__":
    main()

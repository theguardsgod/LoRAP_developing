"""Resumable four-GPU queue with immutable source and 24 paired training runs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from ..run import write_json
from ..residual_channel.experiment import sha
from .run import CALIBRATION, ROOT, SOURCE, VARIANTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--calibration",type=Path,default=CALIBRATION)
    parser.add_argument("--gpus",nargs="+",type=int,default=[0,1,2,3])
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    if len(set(args.gpus))!=len(args.gpus) or any(g<0 or g>=torch.cuda.device_count() for g in args.gpus):
        raise ValueError("Invalid GPU lanes")
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/".lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot=root/"source_snapshot"
    if not (root/"study.json").exists():
        repo=Path(__file__).resolve().parents[3]
        manifest={}
        for folder in ("runs/lorap_study","prompt","quantize_function","quantize_method","utils"):
            for path in (repo/folder).rglob("*.py"):
                name=path.relative_to(repo)
                target=snapshot/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
                manifest[str(name)]=sha(path)
        write_json(root/"source_manifest.json",manifest)
        shutil.copy2(repo/"runs/lorap_study/residual_adapt/README.md",root/"METHOD.md")
        write_json(root/"study.json",dict(source=str(args.source.resolve()),calibration=str(args.calibration.resolve()),
            datasets=["MNIST","CIFAR10"],seeds=[41,42,43],variants=list(VARIANTS),epochs=20,
            split="full official train and validation",gpus=args.gpus,
            gate_lr=.01,backbone_lr=.0001,frozen_quantizers=True,rank=2,side_bits=8,
            bn_interventions=["gate_only_bn","none_bn","unit_gate_bn"],bn_calibration_graphs=1024,
            test_evaluations=0,created_unix=time.time()))
        write_json(root/"queue.json",[dict(dataset=d,seed=s,variant=v,status="pending")
            for s in (41,42,43) for d in ("MNIST","CIFAR10") for v in VARIANTS])
    study=json.loads((root/"study.json").read_text())
    if study["source"]!=str(args.source.resolve()) or study["calibration"]!=str(args.calibration.resolve()):
        raise ValueError("Source or calibration changed")
    for name,digest in json.loads((root/"source_manifest.json").read_text()).items():
        if sha(snapshot/name)!=digest:
            raise ValueError("Source snapshot changed: "+name)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(snapshot),
             CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    with (root/"tests.log").open("a") as log:
        subprocess.run([sys.executable,"-B","-m","unittest","runs.lorap_study.residual_adapt.test_adapt","-v"],
            cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=str(args.gpus[0])),stdout=log,stderr=subprocess.STDOUT,check=True)
    queue=json.loads((root/"queue.json").read_text())
    for job in queue:
        path=root/job["dataset"]/job["variant"]/f"seed_{job['seed']}/result.json"
        if job.get("pid"):
            command=Path(f"/proc/{job['pid']}/cmdline")
            if command.exists() and str(root).encode() in command.read_bytes() and b"residual_adapt.run" in command.read_bytes():
                raise RuntimeError("Existing worker is still running")
        job["status"]="complete" if path.exists() else "pending"
    active={}
    (root/"logs").mkdir(exist_ok=True)
    while True:
        stopping=(root/"STOP").exists()
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is None:
                continue
            log.close()
            path=root/job["dataset"]/job["variant"]/f"seed_{job['seed']}/result.json"
            status="complete" if code==0 and path.exists() else "paused" if stopping else "failed"
            job.update(status=status,exit_code=code,finished_unix=time.time())
            print(f"{job['dataset']} {job['variant']} {job['seed']}: {status}",flush=True)
            del active[gpu]
        if not stopping:
            for gpu in args.gpus:
                if gpu in active:
                    continue
                job=next((j for j in queue if j["status"]=="pending"),None)
                if job is None:
                    continue
                log=(root/"logs"/f"{job['dataset']}_{job['variant']}_{job['seed']}.log").open("a")
                command=[sys.executable,"-B","-m","runs.lorap_study.residual_adapt.run","--root",str(root),
                    "--source",study["source"],"--calibration",study["calibration"],"--dataset",job["dataset"],
                    "--variant",job["variant"],"--seed",str(job["seed"]),"--epochs",str(study["epochs"])]
                process=subprocess.Popen(command,cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                job.update(status="running",pid=process.pid,gpu=gpu,started_unix=time.time())
                active[gpu]=(process,job,log)
                print(f"GPU {gpu}: {job['dataset']} {job['variant']} seed={job['seed']}",flush=True)
        write_json(root/"queue.json",queue)
        progress=[]
        for _,job,_ in active.values():
            path=root/job["dataset"]/job["variant"]/f"seed_{job['seed']}/progress.json"
            progress.append(dict(job=job,progress=json.loads(path.read_text()) if path.exists() else None))
        write_json(root/"status.json",dict(total=len(queue),complete=sum(j["status"]=="complete" for j in queue),
            running=progress,failed=[j for j in queue if j["status"]=="failed"],stopping=stopping,updated_unix=time.time()))
        if not active:
            break
        time.sleep(3)
    if not all(j["status"]=="complete" for j in queue):
        raise RuntimeError("Queue stopped before all runs completed; inspect status and logs")


if __name__=="__main__":
    main()

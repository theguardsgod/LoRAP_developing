"""Wait for the fixed training queue, audit selected models on GPU, then report."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from ..run import write_json
from ..residual_channel.experiment import sha
from .run import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--gpus",type=int,nargs="+",default=[0,1,2,3])
    args=parser.parse_args()
    root=args.root.resolve()
    snapshot=Path(__file__).resolve().parents[3]
    if snapshot!=root/"analysis_snapshot":
        raise ValueError("Run from the archived analysis_snapshot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    if len(set(args.gpus))!=len(args.gpus) or any(g<0 or g>=torch.cuda.device_count() for g in args.gpus):
        raise ValueError("Invalid GPU lanes")
    lock=(root/".analysis_lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,digest in json.loads((root/"analysis_source_manifest.json").read_text()).items():
        assert sha(snapshot/name)==digest, name
    print("Waiting for all 24 training jobs before GPU budget audit",flush=True)
    while True:
        status=json.loads((root/"status.json").read_text())
        if status["failed"] or status["stopping"]:
            raise RuntimeError("Training stopped or failed; audit was not started")
        if status["complete"]==status["total"]==24 and not status["running"]:
            break
        time.sleep(5)
    study=json.loads((root/"study.json").read_text())
    queue=[dict(dataset=d,seed=s,status="pending") for d in study["datasets"] for s in study["seeds"]]
    for job in queue:
        if (root/job["dataset"]/f"budget_seed_{job['seed']}/result.json").exists():
            job["status"]="complete"
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(snapshot),
             CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    active={}
    while True:
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is None:
                continue
            log.close()
            result=root/job["dataset"]/f"budget_seed_{job['seed']}/result.json"
            job.update(status="complete" if code==0 and result.exists() else "failed",exit_code=code)
            print(f"Budget {job['dataset']} {job['seed']}: {job['status']}",flush=True)
            del active[gpu]
        if not (root/"STOP").exists():
            for gpu in args.gpus:
                if gpu in active:
                    continue
                job=next((j for j in queue if j["status"]=="pending"),None)
                if job is None:
                    continue
                log=(root/"logs"/f"budget_{job['dataset']}_{job['seed']}.log").open("a")
                process=subprocess.Popen([sys.executable,"-B","-m","runs.lorap_study.residual_adapt.budget",
                    "--root",str(root),"--dataset",job["dataset"],"--seed",str(job["seed"])],
                    cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                job.update(status="running",gpu=gpu,pid=process.pid)
                active[gpu]=(process,job,log)
                print(f"GPU {gpu}: budget {job['dataset']} {job['seed']}",flush=True)
        write_json(root/"budget_queue.json",queue)
        if not active:
            break
        time.sleep(3)
    if not all(j["status"]=="complete" for j in queue):
        raise RuntimeError("Budget audit incomplete; inspect budget_queue.json and logs")
    with (root/"analysis.log").open("a") as log:
        subprocess.run([sys.executable,"-B","-m","runs.lorap_study.residual_adapt.report","--root",str(root)],
            cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=""),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    print("Completed 60 GPU budget observations and artifact-only report",flush=True)


if __name__=="__main__":
    main()

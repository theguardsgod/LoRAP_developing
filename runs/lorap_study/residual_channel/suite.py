"""Frozen-source, four-GPU queue for the 20 checkpoint feasibility study."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..run import write_json
from .experiment import DEFAULT_ROOT, DEFAULT_SOURCE, MODES, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0,1,2,3])
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/".lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = root/"source_snapshot"
    if not (root/"study.json").exists():
        repo = Path(__file__).resolve().parents[3]
        manifest = {}
        for folder in ("runs/lorap_study", "prompt", "quantize_function", "quantize_method", "utils"):
            for path in (repo/folder).rglob("*.py"):
                relative = path.relative_to(repo)
                target = snapshot/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                manifest[str(relative)] = sha(path)
        write_json(root/"source_manifest.json", manifest)
        write_json(root/"study.json", dict(source=str(source), gpus=args.gpus,
            datasets=["MNIST","CIFAR10"], seeds=list(range(41,51)), modes=list(MODES),
            checkpoint_variant="none", rank=2, side_bits=8, calibration_graphs=512,
            basis_fit_graphs=256, training_holdout_graphs=256, validation="full official split",
            model_forwards="CUDA only", optimizer_updates=0, additional_test_evaluations=0,
            source_execution="source_snapshot", created_unix=time.time()))
        write_json(root/"queue.json", [dict(dataset=d,seed=s,status="pending")
            for s in range(41,51) for d in ("MNIST","CIFAR10")])
    study = json.loads((root/"study.json").read_text())
    if study["source"] != str(source):
        raise ValueError("Source study changed")
    for name, digest in json.loads((root/"source_manifest.json").read_text()).items():
        if sha(snapshot/name) != digest:
            raise ValueError(f"Source snapshot changed: {name}")
    # Run tests from the same immutable source as workers; persist actual output.
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(snapshot),
                       CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       CUDA_VISIBLE_DEVICES=str(args.gpus[0]), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (root/"tests.log").open("a") as log:
        subprocess.run([sys.executable,"-B","-m","unittest",
                        "runs.lorap_study.residual_channel.test_core","-v"],
                       cwd=snapshot,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    queue = json.loads((root/"queue.json").read_text())
    for job in queue:
        path = root/job["dataset"]/f"seed_{job['seed']}"/"result.json"
        if job.get("pid") and Path(f"/proc/{job['pid']}/cmdline").exists():
            if b"runs.lorap_study.residual_channel.experiment" in Path(f"/proc/{job['pid']}/cmdline").read_bytes():
                raise RuntimeError("Existing worker still running")
        job["status"] = "complete" if path.exists() else "pending"
    active = {}
    (root/"logs").mkdir(exist_ok=True)
    while True:
        for gpu, (process, job, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            path = root/job["dataset"]/f"seed_{job['seed']}"/"result.json"
            job.update(status="complete" if code==0 and path.exists() else "failed",
                       exit_code=code, finished_unix=time.time())
            print(f"{job['dataset']} {job['seed']}: {job['status']}",flush=True)
            del active[gpu]
        for gpu in args.gpus:
            if gpu in active:
                continue
            pending = next((job for job in queue if job["status"]=="pending"),None)
            if pending is None:
                continue
            env = dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))
            command = [sys.executable,"-B","-m","runs.lorap_study.residual_channel.experiment",
                       "--dataset",pending["dataset"],"--seed",str(pending["seed"]),
                       "--source",str(source),"--root",str(root)]
            log = (root/"logs"/f"{pending['dataset']}_{pending['seed']}.log").open("a")
            process = subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT)
            pending.update(status="running",pid=process.pid,gpu=gpu,started_unix=time.time())
            active[gpu] = (process,pending,log)
            print(f"GPU {gpu}: {pending['dataset']} {pending['seed']}",flush=True)
        write_json(root/"queue.json",queue)
        write_json(root/"status.json",dict(total=len(queue),
            complete=sum(j["status"]=="complete" for j in queue),
            failed=[j for j in queue if j["status"]=="failed"],
            running=[j for j in queue if j["status"]=="running"],updated_unix=time.time()))
        if not active:
            break
        time.sleep(2)
    if any(j["status"] != "complete" for j in queue):
        raise RuntimeError("Some evaluations failed; inspect logs before resuming")


if __name__ == "__main__":
    main()

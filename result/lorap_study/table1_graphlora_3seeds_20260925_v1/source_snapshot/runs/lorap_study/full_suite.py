"""Durable multi-GPU queue, frozen source snapshot and dataset-level test release."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .full_config import DATASETS, VARIANTS, full_config, validate
from .full_models import load_full_data
from .full_report import generate_report
from .full_audit import audit_results
from .full_run import sha_config
from .run import provenance, write_json


def job_id(phase,dataset,variant,seed):
    return f"{phase}_{dataset}_{variant}_{seed}"


def prepare(root,datasets,seeds,epochs,variants,gpus):
    root.mkdir(parents=True,exist_ok=False)
    provenance(root)
    configs=[]
    for name in datasets:
        config=full_config(name)
        config.update(seeds=seeds,epochs=epochs,variants=variants)
        validate(config)
        _,manifest=load_full_data(config)
        write_json(root/"configs"/f"{name}.json",config)
        write_json(root/name/"split_manifest.json",manifest)
        configs.append(config)
    study=dict(configs=configs,gpus=gpus,selection="minimum validation CE; earlier epoch on ties",
               test_release="after all planned fits of each dataset; no variant screening",
               training_device="CUDA required",source_execution="frozen source_snapshot",
               created_unix=time.time())
    write_json(root/"study.json",study)
    queue=[]
    for phase in ("fit","test"):
        for seed in seeds:
            for variant in variants:
                for config in configs:
                    name=config["dataset"]
                    queue.append(dict(id=job_id(phase,name,variant,seed),phase=phase,dataset=name,
                        task=config["task"],variant=variant,seed=seed,status="pending",attempts=0))
    write_json(root/"queue.json",queue)
    (root/"logs").mkdir()
    command=(f"cd {root}/source_snapshot\nPYTHONDONTWRITEBYTECODE=1 {sys.executable} -B "
             f"-m runs.lorap_study.full_suite --root {root} --resume --gpus {' '.join(map(str,gpus))}")
    (root/"RUN_COMMANDS.md").write_text("# GPU 完整训练运行命令\n\n恢复既定队列：\n\n```sh\n"+command+
        f"\n```\n\n状态：读取 `{root}/REPORT.md` 或 `status.json`。\n\n"
        f"暂停：创建 `{root}/STOP`；进程会在当前轮次结束后暂停。恢复命令移除 STOP。\n\n"
        "只在能访问 CUDA 的宿主机环境运行；沙箱未暴露 GPU 时应通过获准的宿主机执行入口启动。\n")
    generate_report(root,queue)
    return study,queue


def fit_complete(root,job):
    path=root/job["dataset"]/job["variant"]/f"seed_{job['seed']}/result.json"
    if not path.exists():
        return False
    result=json.loads(path.read_text())
    return job["phase"]=="fit" or result["test_evaluations"]==1


def release_tests(root,study,queue):
    for config in study["configs"]:
        jobs=[j for j in queue if j["dataset"]==config["dataset"] and j["phase"]=="fit"]
        path=root/config["dataset"]/"test_release.json"
        if not path.exists() and all(j["status"]=="complete" for j in jobs):
            results=[json.loads((root/j["dataset"]/j["variant"]/f"seed_{j['seed']}/result.json").read_text()) for j in jobs]
            assert all(r["test_evaluations"]==0 for r in results)
            write_json(path,dict(config_sha256=sha_config(config),released_unix=time.time(),fits=results))


def choose_job(root,queue,preferred_task):
    eligible=[j for j in queue if j["status"]=="pending" and
              (j["phase"]=="fit" or (root/j["dataset"]/"test_release.json").exists())]
    # Complete released tests promptly; reserve lanes for both task sizes.
    eligible.sort(key=lambda j:(j["phase"]!="test",j["task"]!=preferred_task,queue.index(j)))
    return eligible[0] if eligible else None


def run_queue(root,study,queue,gpus,threads=2,source_snapshot=None,source_manifest=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU-only queue will not start CPU training")
    if any(gpu<0 or gpu>=torch.cuda.device_count() for gpu in gpus):
        raise ValueError("Requested GPU index is unavailable")
    snapshot=Path(source_snapshot) if source_snapshot else root/"source_snapshot"
    manifest=Path(source_manifest) if source_manifest else root/"provenance.json"
    sources=json.loads(manifest.read_text())["source_sha256"]
    for name,digest in sources.items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Source snapshot modified: {name}")
    for job in queue:
        if fit_complete(root,job):
            job["status"]="complete"
        elif job["status"] in ("running","paused","failed"):
            if job["status"]=="running" and job.get("pid"):
                command=Path(f"/proc/{job['pid']}/cmdline")
                if command.exists():
                    arguments=command.read_bytes().split(b"\0")
                    if b"runs.lorap_study.full_run" in arguments and str(root).encode() in arguments:
                        raise RuntimeError(f"Existing worker {job['pid']} is still running; do not launch a duplicate queue")
            job.update(status="pending",attempts=0)
    active={}
    last_report=0
    while True:
        stopping=(root/"STOP").exists()
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is None:
                continue
            log.close()
            job.update(exit_code=code,finished_unix=time.time())
            if code==0 and fit_complete(root,job):
                job["status"]="complete"
            elif stopping:
                job["status"]="paused"
            elif job["phase"]=="fit" and job["attempts"]<2:
                job["status"]="pending"
            else:
                job["status"]="failed"
            print(f"{job['id']}: {job['status']}",flush=True)
            del active[gpu]
        release_tests(root,study,queue)
        if not stopping:
            for lane,gpu in enumerate(gpus):
                if gpu in active:
                    continue
                preferred="node" if lane<len(gpus)//2 else "graph"
                job=choose_job(root,queue,preferred)
                if job is None:
                    continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",
                         PYTHONPATH=str(snapshot),OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
                command=[sys.executable,"-B","-m","runs.lorap_study.full_run","--root",str(root),
                    "--config",str(root/"configs"/f"{job['dataset']}.json"),"--variant",job["variant"],
                    "--seed",str(job["seed"]),"--phase",job["phase"],"--threads",str(threads)]
                log=(root/"logs"/f"{job['id']}.log").open("a")
                process=subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT)
                job.update(status="running",gpu=gpu,pid=process.pid,started_unix=time.time(),attempts=job["attempts"]+1)
                active[gpu]=(process,job,log)
                print(f"GPU {gpu}: {job['id']}",flush=True)
        write_json(root/"queue.json",queue)
        if time.time()-last_report>20 or not active:
            generate_report(root,queue)
            last_report=time.time()
        if not active:
            if stopping or not any(j["status"]=="pending" and
                (j["phase"]=="fit" or (root/j["dataset"]/"test_release.json").exists()) for j in queue):
                break
        time.sleep(3)
    generate_report(root,queue)
    audit_results(root,json.loads((root/"results.json").read_text()))
    return 0 if all(j["status"]=="complete" for j in queue) else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    parser.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(41,51)))
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--gpus",nargs="+",type=int,default=[0,1,2,3])
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--source-snapshot",type=Path)
    parser.add_argument("--source-manifest",type=Path)
    args=parser.parse_args(argv)
    if bool(args.source_snapshot)!=bool(args.source_manifest) or (args.source_snapshot and not args.resume):
        parser.error("An alternate frozen snapshot and manifest must be supplied together, with --resume")
    root=args.root.resolve()
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or args.threads<1:
        parser.error("Distinct GPUs and positive CPU helper threads required")
    torch.set_num_threads(args.threads)
    if args.resume:
        study=json.loads((root/"study.json").read_text())
        queue=json.loads((root/"queue.json").read_text())
    else:
        study,queue=prepare(root,args.datasets,args.seeds,args.epochs,args.variants,args.gpus)
    if args.prepare_only:
        print(f"Prepared {sum(j['phase']=='fit' for j in queue)} GPU fits at {root}")
        return
    with (root/".coordinator.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            (root/"STOP").unlink(missing_ok=True)
        write_json(root/"coordinator.json",dict(pid=os.getpid(),gpus=args.gpus,started_unix=time.time()))
        code=run_queue(root,study,queue,args.gpus,args.threads,args.source_snapshot,args.source_manifest)
        raise SystemExit(code)


if __name__=="__main__":
    main()

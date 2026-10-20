"""Fixed GPU matrix; finish every validation selection before new test metrics."""

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
from .config import ROOT,REPO,DATASETS,MODELS,SEEDS,VARIANTS,configuration


def execute_phase(root,snapshot,phase,gpus,env):
    jobs=[dict(dataset=d,model=m,seed=s,status='pending') for d in DATASETS for m in MODELS for s in SEEDS]
    filename='fit_result.json' if phase=='fit' else 'result.json'
    for job in jobs:
        if (root/job['dataset']/job['model']/f"seed_{job['seed']}"/filename).exists(): job['status']='complete'
    active={}
    while True:
        stopping=(root/'STOP').exists()
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is None: continue
            log.close()
            path=root/job['dataset']/job['model']/f"seed_{job['seed']}"/filename
            job.update(status='complete' if code==0 and path.exists() else 'paused' if stopping else 'failed',exit_code=code)
            print(phase,job['dataset'],job['model'],job['seed'],job['status'],flush=True)
            del active[gpu]
        if not stopping:
            for gpu in gpus:
                if gpu in active: continue
                job=next((j for j in jobs if j['status']=='pending'),None)
                if job is None: continue
                log=(root/'logs'/f"{phase}_{job['dataset']}_{job['model']}_{job['seed']}.log").open('a')
                process=subprocess.Popen([sys.executable,'-B','-m','runs.lorap_study.quant_predict.run',
                    '--root',str(root),'--dataset',job['dataset'],'--model',job['model'],'--seed',str(job['seed']),'--phase',phase],
                    cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                job.update(status='running',pid=process.pid,gpu=gpu,started_unix=time.time())
                active[gpu]=(process,job,log)
                print('GPU',gpu,phase,job['dataset'],job['model'],job['seed'],flush=True)
        running=[]
        for _,job,_ in active.values():
            progress=root/job['dataset']/job['model']/f"seed_{job['seed']}/progress.json"
            running.append(dict(job=job,progress=json.loads(progress.read_text()) if progress.exists() else None))
        write_json(root/f'{phase}_queue.json',jobs)
        write_json(root/'status.json',dict(phase=phase,total=len(jobs),complete=sum(j['status']=='complete' for j in jobs),
            failed=[j for j in jobs if j['status']=='failed'],running=running,stopping=stopping,updated_unix=time.time()))
        if not active: break
        time.sleep(2)
    if not all(j['status']=='complete' for j in jobs): raise RuntimeError(f'{phase} incomplete; inspect logs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    if len(set(args.gpus))!=len(args.gpus) or any(g<0 or g>=torch.cuda.device_count() for g in args.gpus):
        raise ValueError('Invalid GPU lanes')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot=root/'source_snapshot'
    if not (root/'study.json').exists():
        manifest={}
        for folder in ('runs/lorap_study','prompt','quantize_function','quantize_method','utils'):
            for path in (REPO/folder).rglob('*.py'):
                relative=path.relative_to(REPO);target=snapshot/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                manifest[str(relative)]=sha(path)
        write_json(root/'source_manifest.json',manifest)
        write_json(root/'study.json',dict(datasets=list(DATASETS),models=list(MODELS),seeds=list(SEEDS),variants=list(VARIANTS),
            reused_baselines=27,adaptation_fits=81,epochs=100,error_loss_weight=.1,
            test_release='after all 81 fits and validation selections',new_test_metric_evaluations=81,
            created_unix=time.time(),gpus=args.gpus,configurations={d:{m:configuration(d,m) for m in MODELS} for d in DATASETS}))
        shutil.copy2(REPO/'runs/lorap_study/quant_predict/README.md',root/'METHOD.md')
        write_json(root/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            python=sys.version,gpus=[torch.cuda.get_device_name(i) for i in args.gpus]))
    for name,digest in json.loads((root/'source_manifest.json').read_text()).items(): assert sha(snapshot/name)==digest
    for phase in ('fit','test'):
        queue=root/f'{phase}_queue.json'
        if queue.exists():
            for job in json.loads(queue.read_text()):
                cmdline=Path(f"/proc/{job.get('pid',-1)}/cmdline")
                if cmdline.exists() and b'quant_predict.run' in cmdline.read_bytes() and str(root).encode() in cmdline.read_bytes():
                    raise RuntimeError('A worker is still running')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(snapshot),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    (root/'logs').mkdir(exist_ok=True)
    if not (root/'tests_passed.json').exists():
        with (root/'tests.log').open('w') as log:
            subprocess.run([sys.executable,'-B','-m','unittest','runs.lorap_study.quant_predict.test_predict','-v'],
                cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=str(args.gpus[0])),stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(root/'tests_passed.json',dict(count=11,log_sha256=sha(root/'tests.log'),manifest_sha256=sha(root/'source_manifest.json')))
    execute_phase(root,snapshot,'fit',args.gpus,env)
    fits={}
    for d in DATASETS:
        for m in MODELS:
            for seed in SEEDS:
                directory=root/d/m/f'seed_{seed}'
                names=['fit_result.json','config.json','split.json','calibration.pt']
                names += [str(Path(v)/n) for v in VARIANTS for n in ('best.pt','last.pt','inference.pt','fit_result.json')]
                fits[str(directory.relative_to(root))]={name:sha(directory/name) for name in names}
    release=dict(fits=fits,all_fits_complete=True,selections_frozen=True)
    path=root/'test_release.json'
    if path.exists(): assert json.loads(path.read_text())==release
    else: write_json(path,release)
    execute_phase(root,snapshot,'test',args.gpus,env)
    print('All 81 adaptations and 81 new fixed test conditions completed',flush=True)


if __name__=='__main__': main()

"""Immutable four-GPU queue; all validation selections precede test release."""

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
from .config import ROOT,REPO,DATASETS,MODELS,SEEDS,MODES,configuration


def execute_phase(root,snapshot,phase,gpus,env):
    queue=[dict(dataset=d,model=m,seed=s,status='pending') for d in DATASETS for m in MODELS for s in SEEDS]
    filename='fit_result.json' if phase=='fit' else 'result.json'
    for job in queue:
        path=root/job['dataset']/job['model']/f"seed_{job['seed']}"/filename
        if path.exists():job['status']='complete'
    active={}
    while True:
        stopping=(root/'STOP').exists()
        for gpu,(p,j,log) in list(active.items()):
            code=p.poll()
            if code is None:continue
            log.close()
            path=root/j['dataset']/j['model']/f"seed_{j['seed']}"/filename
            j.update(status='complete' if code==0 and path.exists() else 'paused' if stopping else 'failed',exit_code=code)
            print(phase,j['dataset'],j['model'],j['seed'],j['status'],flush=True)
            del active[gpu]
        if not stopping:
            for gpu in gpus:
                if gpu in active:continue
                j=next((j for j in queue if j['status']=='pending'),None)
                if j is None:continue
                log=(root/'logs'/f"{phase}_{j['dataset']}_{j['model']}_{j['seed']}.log").open('a')
                p=subprocess.Popen([sys.executable,'-B','-m','runs.lorap_study.residual_transfer.run',
                    '--root',str(root),'--dataset',j['dataset'],'--model',j['model'],'--seed',str(j['seed']),'--phase',phase],
                    cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                j.update(status='running',pid=p.pid,gpu=gpu,started_unix=time.time())
                active[gpu]=(p,j,log)
                print('GPU',gpu,phase,j['dataset'],j['model'],j['seed'],flush=True)
        running=[]
        for _,j,_ in active.values():
            progress=root/j['dataset']/j['model']/f"seed_{j['seed']}/progress.json"
            running.append(dict(job=j,progress=json.loads(progress.read_text()) if progress.exists() else None))
        write_json(root/f'{phase}_queue.json',queue)
        write_json(root/'status.json',dict(phase=phase,total=len(queue),complete=sum(j['status']=='complete' for j in queue),
            failed=[j for j in queue if j['status']=='failed'],running=running,stopping=stopping,updated_unix=time.time()))
        if not active:break
        time.sleep(2)
    if not all(j['status']=='complete' for j in queue):
        raise RuntimeError(f'{phase} incomplete; inspect logs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if len(set(args.gpus))!=len(args.gpus) or any(g<0 or g>=torch.cuda.device_count() for g in args.gpus):
        raise ValueError('Invalid GPU lanes')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot=root/'source_snapshot'
    if not (root/'study.json').exists():
        source=Path(__file__).resolve().parents[3]
        manifest={}
        for folder in ('runs/lorap_study','prompt','quantize_function','quantize_method','utils'):
            for path in (source/folder).rglob('*.py'):
                relative=path.relative_to(source)
                target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target);manifest[str(relative)]=sha(path)
        write_json(root/'source_manifest.json',manifest)
        write_json(root/'study.json',dict(datasets=list(DATASETS),models=list(MODELS),seeds=list(SEEDS),modes=list(MODES),
            baseline_epochs=200,gate_epochs=100,new_baselines=21,reused_baselines=6,gate_fits=27,
            test_release='after all 27 fits and validation selections',test_metric_evaluations_per_case=4,
            created_unix=time.time(),gpus=args.gpus,configurations={d:{m:configuration(d,m) for m in MODELS} for d in DATASETS}))
        shutil.copy2(source/'runs/lorap_study/residual_transfer/README.md',root/'METHOD.md')
    for name,digest in json.loads((root/'source_manifest.json').read_text()).items():
        assert sha(snapshot/name)==digest,name
    for phase in ('fit','test'):
        queue_path=root/f'{phase}_queue.json'
        if queue_path.exists():
            for j in json.loads(queue_path.read_text()):
                cmdline=Path(f"/proc/{j.get('pid',-1)}/cmdline")
                if cmdline.exists() and b'residual_transfer.run' in cmdline.read_bytes() and str(root).encode() in cmdline.read_bytes():
                    raise RuntimeError('A worker is still running')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(snapshot),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    (root/'logs').mkdir(exist_ok=True)
    if not (root/'tests_passed.json').exists():
        with (root/'tests.log').open('w') as log:
            subprocess.run([sys.executable,'-B','-m','unittest','runs.lorap_study.residual_transfer.test_transfer','-v'],
                cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=str(args.gpus[0])),stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(root/'tests_passed.json',dict(count=12,log_sha256=sha(root/'tests.log'),source_manifest_sha256=sha(root/'source_manifest.json')))
    execute_phase(root,snapshot,'fit',args.gpus,env)
    fits={}
    for d in DATASETS:
        for m in MODELS:
            for s in SEEDS:
                directory=root/d/m/f'seed_{s}'
                fits[str(directory.relative_to(root))]={name:sha(directory/name) for name in
                    ('fit_result.json','config.json','split.json','gate/best.pt','baseline/best.pt','calibration.pt')}
    release=dict(fits=fits,all_gate_fits_complete=True,test_selections_frozen=True)
    if (root/'test_release.json').exists():
        assert json.loads((root/'test_release.json').read_text())==release
    else:
        write_json(root/'test_release.json',release)
    execute_phase(root,snapshot,'test',args.gpus,env)
    print('All 27 combinations and 108 fixed test conditions completed',flush=True)


if __name__=='__main__':
    main()

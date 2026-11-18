"""Fresh paired MNIST-GAT stability screening, continuation and final audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch_scatter import segment_csr

from . import portable as p
from .stabilized_report import report


def run_workers(root, launch, phase, workers, device):
    children, streams = [], []
    try:
        for index in range(workers):
            stream = (launch / f'{phase}_worker_{index}.log').open('xb')
            streams.append(stream)
            command = [sys.executable, '-u', '-B', '-m',
                'runs.lorap_study.table1_graphlora.portable', 'worker', '--root', str(root), '--device', device]
            child = subprocess.Popen(command, cwd=root/'source_snapshot',
                env=dict(os.environ, PYTHONPATH=str(root/'source_snapshot')), stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            children.append(child)
        p.write_json(launch/f'{phase}_workers.json', dict(command=command, pids=[c.pid for c in children]))
        study = p.read(root/'study.json')
        while True:
            state = dict(phase=phase, updated_unix=time.time(), complete=False,
                         fits=0, tests=0, failures=[], progress=[])
            for job in study['jobs']:
                folder = root/job['directory']
                state['fits'] += (folder/'done.json').exists()
                state['tests'] += (folder/'test.json').exists()
                if (folder/'failure.json').exists():
                    state['failures'].append(job['directory'])
                if (folder/'progress.json').exists():
                    state['progress'].append(dict(job=job['directory'], **p.read(folder/'progress.json')))
            p.write_json(launch/'pipeline_status.json', state)
            if all(c.poll() is not None for c in children):
                break
            time.sleep(15)
        codes = [c.wait() for c in children]
        p.write_json(launch/f'{phase}_exits.json', dict(returncodes=codes))
        if any(codes) or state['failures']:
            raise RuntimeError(f'{phase} failed; retain failure records and diagnose before proceeding')
    finally:
        for stream in streams:
            stream.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--launch',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    root, launch = args.root.resolve(), args.launch.resolve()
    smoke=root.with_name(root.name+'_smoke')
    if root.exists() or smoke.exists():
        raise FileExistsError('This launcher requires fresh roots; explicit worker resume is separate')
    assert args.workers > 0
    assert torch.cuda.is_available() and args.device.startswith('cuda:')
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    x=torch.tensor([1.,2.],device=args.device,requires_grad=True)
    y=segment_csr(x,torch.tensor([0,2],device=args.device))
    y.sum().backward()
    assert y.item()==3 and x.grad.tolist()==[1.,1.]
    p.write_json(launch/'cuda_check.json',dict(passed=True,torch=torch.__version__,
        cuda=torch.version.cuda,device=args.device,gpu=torch.cuda.get_device_name(args.device)))
    cells=['QAT_MNIST_GAT']
    p.prepare(smoke,args.data_root,cells,[41],smoke=True,learning_rate=.001,gradient_clip_norm=1.)
    run_workers(smoke,launch,'smoke',1,args.device)
    smoke_audit=p.read(smoke/'completion_audit.json')
    assert smoke_audit['passed'] and smoke_audit['tests']==0
    p.prepare(root,args.data_root,cells,[41,42,43],learning_rate=.001,gradient_clip_norm=1.,stability_epochs=30)
    p.write_json(root/'provenance.json',p.read(launch/'provenance.json'))
    (root/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    run_workers(root,launch,'screening',args.workers,args.device)
    subprocess.run([sys.executable,'-B','-m','runs.lorap_study.table1_graphlora.portable',
        'release-stability','--root',str(root)],cwd=root/'source_snapshot',
        env=dict(os.environ,PYTHONPATH=str(root/'source_snapshot')),check=True)
    run_workers(root,launch,'formal',args.workers,args.device)
    assert p.read(root/'completion_audit.json')['passed']
    report(root)
    p.write_json(launch/'pipeline_status.json',dict(phase='complete',complete=True,fits=6,tests=6,
        updated_unix=time.time(),report=str(root/'REPORT.md')))


if __name__=='__main__':
    main()

"""Portable, hash-checked continuation of the three paused CIFAR10 GAT runs."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_scatter import segment_csr
from . import portable as p

REPO=Path(__file__).resolve().parents[3]
PACKAGE=REPO/'handoff/graphlora_20261118'
OriginalBundle=p.Bundle


def fingerprint_dataset(dataset):
    """Hash graph content/order, not pickle bytes or machine-specific file paths."""
    h=hashlib.sha256()
    def walk(name,value):
        h.update(name.encode())
        if torch.is_tensor(value):
            tensor=value.detach().cpu().contiguous()
            h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
            h.update(memoryview(tensor.numpy()).cast('B'))
        elif isinstance(value,dict):
            for key in sorted(value):walk(str(key),value[key])
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):walk(str(index),item)
        else:h.update(repr(value).encode())
    data=dataset._data
    walk('data',{key:data[key] for key in sorted(data.keys())})
    walk('slices',dataset.slices)
    return dict(graphs=len(dataset),sha256=h.hexdigest())


def dataset_fingerprints(data_root):
    result={}
    for split in ['train','val','test']:
        dataset=GNNBenchmarkDataset(str(Path(data_root)/'cifar10'),'CIFAR10',split=split)
        result[split]=fingerprint_dataset(dataset)
        del dataset
    return result


def bind_data(data_root):
    """Relocate only dataset I/O; registered configs and optimizer/RNG state stay intact."""
    data_root=Path(data_root).resolve()
    class RelocatedBundle(OriginalBundle):
        def __init__(self,config):
            moved=dict(config,data_root=str(data_root/config['dataset'].lower()))
            super().__init__(moved)
    p.Bundle=RelocatedBundle


def safe_extract(archive,target):
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar.getmembers():
            path=Path(item.name)
            if path.is_absolute() or '..' in path.parts or not (item.isdir() or item.isfile()):
                raise ValueError(f'Unsafe archive member: {item.name}')
        tar.extractall(target,filter='data')


def verify_payload(root,manifest):
    for name,expected in manifest['files'].items():
        if p.sha(root/name)!=expected:raise ValueError(f'Checkpoint payload changed: {name}')
    study=p.read(root/'study.json')
    assert study['cells']==['A2Q_CIFAR10_GAT'] and study['variants']==['graphlora']
    assert study['seeds']==[41,42,43] and len(study['jobs'])==3
    gate=p.read(root/'STABILITY_GATE.json')
    assert gate['passed'] and gate['study_hash']==p.config_hash(study)
    for job in study['jobs']:
        folder=root/job['directory'];last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert last['config_hash']==p.job_key(study,job)
        assert last['epoch']==manifest['paused_epochs'][str(job['seed'])]
        assert last['epoch']==len(last['history'])<job['epochs']==200
        assert p.read(folder/'history.json')==last['history']
        assert last['rng']['cuda'] is not None
        assert last['optimizer']['state']
        assert not (folder/'done.json').exists() and not (folder/'failure.json').exists()
    p.verify_sources(root)


def prepare(root,data_root):
    if root.exists():raise FileExistsError('Use a NEW output directory. Resume existing runs with run, not prepare.')
    manifest=p.read(PACKAGE/'payload_manifest.json');archive=PACKAGE/'paused_checkpoints.tar.gz'
    if p.sha(archive)!=manifest['archive_sha256']:raise ValueError('Checkpoint archive SHA256 mismatch')
    root.mkdir(parents=True)
    safe_extract(archive,root)
    verify_payload(root,manifest)
    actual=dataset_fingerprints(data_root)
    if actual!=manifest['dataset_fingerprints']:
        raise ValueError('Dataset graph content/order differs from the original server; do not resume')
    p.write_json(root/'migration.json',dict(package_sha256=manifest['archive_sha256'],
        data_root=str(data_root.resolve()),dataset_fingerprints=actual,
        original_paused_epochs=manifest['paused_epochs'],prepared_unix=time.time(),
        note='Only dataset I/O relocated; original study/config hashes, Adam, scheduler and RNG preserved'))
    print(json.dumps(dict(prepared=str(root),paused_epochs=manifest['paused_epochs'],new_none_training=0)),flush=True)


def configure_runtime(root,device,check_data=False):
    p.verify_sources(root)
    migration=p.read(root/'migration.json')
    if check_data and dataset_fingerprints(Path(migration['data_root']))!=migration['dataset_fingerprints']:
        raise ValueError('Dataset changed since preparation')
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    assert torch.__version__.split('+')[0]=='2.7.0', 'Resume using PyTorch 2.7.0'
    torch.set_num_threads(4);torch.cuda.set_device(device)
    x=torch.ones(2,device=device,requires_grad=True)
    segment_csr(x,torch.tensor([0,2],device=device)).sum().backward()
    assert x.grad.tolist()==[1.,1.]
    bind_data(Path(migration['data_root']))
    return p.read(root/'study.json')


def summarize(root):
    study=p.read(root/'study.json');audit=p.read(root/'completion_audit.json')
    assert audit['passed'] and audit['fits']==audit['tests']==3
    manifest=p.test_manifest(root,study);assert manifest==p.read(root/'TEST_LOCK.json')
    rows=[]
    for job,case in zip(study['jobs'],manifest['cases']):
        folder=root/job['directory'];history=p.read(folder/'history.json');done=p.read(folder/'done.json');test=p.read(folder/'test.json')
        assert len(history)==200 and [h['epoch'] for h in history]==list(range(1,201))
        assert done['selected_epoch']==min(history,key=lambda h:(h['validation']['loss'],h['epoch']))['epoch']
        assert test['completed_test_evaluations']==1 and test['examples']==10000
        assert test['test_lock_sha256']==manifest['sha256'] and test['checkpoint_file_sha256']==case['checkpoint_file_sha256']
        rows.append(dict(seed=job['seed'],accuracy=test['metric'],test_loss=test['loss'],
            selected_epoch=done['selected_epoch'],validation_loss=done['validation']['loss'],
            raw_gradient_peak=max(h['train']['gradient_norm_max'] for h in history),
            clipped_gradient_peak=max(h['train']['gradient_norm_after_clip_max'] for h in history),
            clipping_fraction=sum(h['train']['gradient_clipped_batches'] for h in history)/sum(h['train']['gradient_batches'] for h in history)))
    scores=[row['accuracy'] for row in rows]
    result=dict(cell='A2Q_CIFAR10_GAT',recipe='gat_stabilized_v1',mean=statistics.mean(scores),
        sample_sd=statistics.stdev(scores),n=3,seeds=[41,42,43],new_none_training=0,
        other_completed_configurations_retrained=False,scope='Only three migrated models; not the entire original study')
    p.write_json(root/'handoff_summary.json',result)
    with (root/'handoff_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# A2Q–CIFAR10–GAT 迁移续跑结果','',
        f"测试准确率：{result['mean']:.2f} ± {result['sample_sd']:.2f}%（样本标准差，n=3）。",'',
        '| 种子 | 测试准确率 (%) | 最佳轮次 |','|---|---:|---:|']
    for row in rows:lines.append(f"| {row['seed']} | {row['accuracy']:.2f} | {row['selected_epoch']} |")
    lines+=['','仅迁移三个未完成模型，从原断点恢复 Adam、调度器和 RNG，训练至总计 200 轮。',
        '未重新训练 None 或已完成配置。主学习率 0.0001，其余分组为原注册值的 0.1 倍，梯度裁剪 1.0，原始梯度拒绝阈值 1e6。',
        '测试范围经迁移缩小为本配置三个模型。已完成的其他配置保留在原服务器，不纳入这里的 completion_audit。',
        '不同服务器/硬件上的浮点与稀疏归约不保证逐位相同，实际设备记录在 runtime 日志中。']
    (root/'HANDOFF_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result),flush=True)


def run(root,devices):
    with (root/'.handoff_run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configure_runtime(root,devices[0],check_data=True)
        study=p.read(root/'study.json')
        failures=[j['directory'] for j in study['jobs'] if (root/j['directory']/'failure.json').exists()]
        if failures:raise RuntimeError(f'Preserve and diagnose failures before retrying: {failures}')
        runtime=dict(started_unix=time.time(),torch=torch.__version__,cuda=torch.version.cuda,
            python=sys.version,devices=[dict(device=d,gpu=torch.cuda.get_device_name(d)) for d in devices],
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        stamp=str(time.time_ns());p.write_json(root/f'runtime_{stamp}.json',runtime)
        (root/'STOP').unlink(missing_ok=True)
        children=[];streams=[]
        try:
            for index,device in enumerate(devices):
                stream=(root/f'worker_{stamp}_{index}.log').open('xb');streams.append(stream)
                command=[sys.executable,'-u','-B','-m','runs.lorap_study.table1_graphlora.handoff','worker',
                         '--root',str(root),'--devices',device]
                children.append(subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
            while True:
                p.generate(root,study)
                print((root/'status.json').read_text(),flush=True)
                if all(c.poll() is not None for c in children):break
                time.sleep(30)
            codes=[c.wait() for c in children];p.write_json(root/f'worker_exits_{stamp}.json',dict(returncodes=codes))
            if any(codes):raise RuntimeError('Worker failed; inspect logs/failure.json; do not silently retry')
        finally:
            for stream in streams:stream.close()
        if (root/'STOP').exists():
            print('Paused at saved epoch boundaries; use run again to resume.',flush=True)
            return
        summarize(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker','report'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--devices',nargs='+',default=['cuda:0'])
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':
        if args.data_root is None:parser.error('--data-root required for prepare')
        prepare(root,args.data_root)
    elif args.action=='run':run(root,args.devices)
    elif args.action=='worker':
        configure_runtime(root,args.devices[0]);p.worker(root,args.devices[0])
    else:summarize(root)


if __name__=='__main__':main()

"""Fit paired frozen gates; release test metrics only after the full matrix is fit."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid

from ..diagnostics import QuantizationObserver
from ..full_run import atomic_save,cpu_tree,require_cuda,sha_config
from ..models import state_digest
from ..residual_adapt.run import update_best,verify_selection
from ..residual_channel.experiment import sha
from ..run import write_json
from .config import ROOT,SOURCE,DATASETS,MODELS,MODES,configuration
from .models import build,forward,optimizer,storage_proxy,message_quantizers,project_quantizers
from .channel import calibrate,GatedModel,Diagnostics,metadata


def load_data(c,device):
    dataset=Planetoid(c['data_root'],c['dataset'],split='public')
    data=dataset[0].to(device)
    manifest=dict(dataset=c['dataset'],nodes=data.num_nodes,edges=data.num_edges,features=data.num_features,
        indices={s:getattr(data,s+'_mask').nonzero().flatten().cpu().tolist() for s in ('train','val','test')},
        processed_sha256=sha(dataset.processed_paths[0]))
    return data,manifest


@torch.no_grad()
def evaluate(model,data,split,c=None):
    model.eval()
    logits=model(data) if c is None else forward(model,c,data)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Nonfinite model output')
    mask=getattr(data,split+'_mask')
    logits,y=logits[mask],data.y[mask]
    return dict(loss=float(F.cross_entropy(logits,y)),accuracy=float((logits.argmax(1)==y).double().mean()),
                examples=int(mask.sum()))


def finite_gradients(model):
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(v).all()) for v in gradients):
        raise FloatingPointError('Missing or nonfinite gradients')


def epoch_seed(seed,epoch):
    # Makes baseline dropout reproducible, including resumes at an epoch boundary.
    torch.manual_seed(914000+seed*1000+epoch)


def check_stop(root):
    if (root/'STOP').exists():
        raise InterruptedError('Paused at epoch boundary')


def save_epoch(path,last,best):
    atomic_save(last,path/'last.pt')
    atomic_save(best,path/'best.pt')
    (path/'epochs.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in last['history']))


def baseline(c,data,seed,directory,root,device):
    path=directory/'baseline'
    path.mkdir(exist_ok=True)
    model=build(c,data,seed).to(device)
    initial=state_digest(model,'all')
    source=SOURCE/c['dataset']/'none'/f'seed_{seed}'
    if c['architecture']=='GIN' and c['dataset'] in ('Cora','CiteSeer'):
        hashes={str(source/n):sha(source/n) for n in ('best.pt','last.pt','config.json','result.json','test_started.json')}
        checkpoint=torch.load(source/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True)
        assert all(bool((p.detach().abs().round()>=1).all()) for n,p in model.named_parameters() if n.endswith('.bit'))
        validation=evaluate(model,data,'val',c)
        expected=float(checkpoint['validation_loss'])
        assert abs(validation['loss']-expected)<5e-5,(validation,expected)
        result=dict(reused=True,source=str(source),source_hashes=hashes,epoch=checkpoint['epoch'],
            selected_validation=validation,initial_sha256=initial,model_sha256=state_digest(model,'all'),
            historical_validation_loss=expected,replay_absolute_delta=abs(validation['loss']-expected))
        atomic_save(dict(model=cpu_tree(model.state_dict()),epoch=checkpoint['epoch'],validation_loss=validation['loss']),path/'best.pt')
        write_json(path/'result.json',result)
        return model,result
    if (path/'result.json').exists():
        saved=torch.load(path/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model'])
        return model,json.loads((path/'result.json').read_text())
    opt=optimizer(model,c)
    history,best,start=[],None,0
    if (path/'last.pt').exists():
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(last['model']);opt.load_state_dict(last['optimizer'])
        history,best,start=last['history'],last['best'],last['epoch']+1
    for epoch in range(start,c['baseline_epochs']+1):
        check_stop(root)
        train=None
        if epoch:
            epoch_seed(seed,epoch)
            model.train();opt.zero_grad(set_to_none=True)
            logits=forward(model,c,data)
            ce=F.cross_entropy(logits[data.train_mask],data.y[data.train_mask])
            proxy=storage_proxy(model,c)
            objective=ce+c['a_loss']*F.relu(proxy-c['a_storage']).square()
            objective.backward();finite_gradients(model);opt.step();project_quantizers(model,c)
            train=dict(loss=float(ce.detach()),objective=float(objective.detach()),storage_proxy=float(proxy.detach()))
        val=evaluate(model,data,'val',c)
        best=update_best(best,val['loss'],epoch,model)
        history.append(dict(epoch=epoch,validation=val,train=train))
        last=dict(epoch=epoch,history=history,best=best,model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()))
        save_epoch(path,last,best)
        write_json(directory/'progress.json',dict(phase='baseline',epoch=epoch,total=c['baseline_epochs'],best_epoch=best['epoch']))
        if epoch%25==0:
            print(c['dataset'],c['architecture'],seed,'baseline',epoch,val,flush=True)
    verify_selection(last,best)
    model.load_state_dict(best['model'])
    val=evaluate(model,data,'val',c)
    assert val['loss']==best['validation_loss']
    result=dict(reused=False,source_hashes={},epoch=best['epoch'],selected_validation=val,
                initial_sha256=initial,model_sha256=state_digest(model,'all'),epochs=c['baseline_epochs'],
                last_validation=history[-1]['validation'],exact_best_replay=True)
    write_json(path/'result.json',result)
    return model,result


def fit(args,c,data,directory,root,device):
    if (directory/'fit_result.json').exists():
        return
    backbone,base=baseline(c,data,args.seed,directory,root,device)
    backbone.eval()
    if (directory/'calibration.pt').exists():
        codes=torch.load(directory/'calibration.pt',map_location=device,weights_only=False)
    else:
        codes,manifest=calibrate(backbone,c,data)
        atomic_save(cpu_tree(codes),directory/'calibration.pt')
        write_json(directory/'calibration.json',manifest)
    wrapped=GatedModel(backbone,c,codes).to(device)
    frozen=state_digest(wrapped.backbone,'all')
    buffers=state_digest(wrapped.codes,'all')
    opt=torch.optim.Adam([wrapped.gates],lr=c['gate_lr'],weight_decay=0.)
    path=directory/'gate'
    path.mkdir(exist_ok=True)
    history,best,start=[],None,0
    if (path/'last.pt').exists():
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        wrapped.load_state_dict(last['model']);opt.load_state_dict(last['optimizer'])
        history,best,start=last['history'],last['best'],last['epoch']+1
    for epoch in range(start,c['gate_epochs']+1):
        check_stop(root)
        wrapped.eval()
        grad=train=None
        if epoch:
            opt.zero_grad(set_to_none=True)
            logits=wrapped(data)
            loss=F.cross_entropy(logits[data.train_mask],data.y[data.train_mask])
            loss.backward();finite_gradients(wrapped)
            grad=wrapped.gates.grad.detach().cpu().tolist()
            opt.step();wrapped.project()
            train=dict(loss=float(loss.detach()),gate_gradient=grad)
        val=evaluate(wrapped,data,'val')
        if epoch==0:
            assert val==base['selected_validation'],(val,base['selected_validation'])
        assert state_digest(wrapped.backbone,'all')==frozen
        assert state_digest(wrapped.codes,'all')==buffers
        best=update_best(best,val['loss'],epoch,wrapped)
        history.append(dict(epoch=epoch,validation=val,train=train,gates=wrapped.gates.detach().cpu().tolist()))
        last=dict(epoch=epoch,history=history,best=best,model=cpu_tree(wrapped.state_dict()),optimizer=cpu_tree(opt.state_dict()))
        save_epoch(path,last,best)
        write_json(directory/'progress.json',dict(phase='gate',epoch=epoch,total=c['gate_epochs'],best_epoch=best['epoch']))
        if epoch%25==0:
            print(c['dataset'],c['architecture'],args.seed,'gate',epoch,val,wrapped.gates.tolist(),flush=True)
    verify_selection(last,best)
    wrapped.load_state_dict(best['model'])
    selected=evaluate(wrapped,data,'val')
    assert selected['loss']==best['validation_loss']
    validation={}
    for mode in MODES:
        wrapped.mode=mode
        collector=Diagnostics(data.val_mask)
        plain=evaluate(wrapped,data,'val')
        wrapped.observer=collector
        observed=evaluate(wrapped,data,'val')
        wrapped.observer=None
        assert plain==observed
        validation[mode]=dict(**plain,local_diagnostics=collector.rows)
    wrapped.mode='gate'
    with torch.no_grad():
        wrapped.bypass_values=True
        with_side=wrapped(data)
        wrapped.mode='none'
        without_side=wrapped(data)
        assert torch.equal(with_side,without_side)
        wrapped.bypass_values=False
    for p,digest in base['source_hashes'].items():
        assert sha(p)==digest
    result=dict(dataset=c['dataset'],architecture=c['architecture'],seed=args.seed,
        base=base,gate_epochs=c['gate_epochs'],best_gate_epoch=best['epoch'],
        gates=best['model']['gates'].tolist(),last_gates=last['model']['gates'].tolist(),
        last_validation=history[-1]['validation'],validation=validation,
        frozen_backbone_sha256=frozen,code_sha256=buffers,backbone_unchanged=True,
        diagnostic_switch_exact=True,bypass_side_exact=True,gate_parameters=len(codes),
        device=str(device),gpu_name=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        test_metric_evaluations=0)
    write_json(directory/'fit_result.json',result)
    write_json(directory/'progress.json',dict(phase='fit_complete'))


def bit_audit(wrapped,c,data,stats):
    layers=[]
    for i,(quantizer,code) in enumerate(zip(message_quantizers(wrapped.backbone,c),wrapped.codes)):
        width=code.basis.shape[0]*code.basis.shape[1]
        _,bits=metadata(quantizer)
        total=float(bits.sum())*width if hasattr(quantizer,'gama') else float(bits.sum())*data.num_nodes
        extra=int(code.increments.sum()) if wrapped.mode=='higher' else 0
        total+=extra*data.num_nodes
        side=code.basis.shape[0]*c['rank']*8 if wrapped.mode in ('unit','gate') else 0
        layers.append(dict(layer=i+1,main_bits_per_node=total/data.num_nodes,
            side_bits_per_node=side,total_message_bits_per_node=total/data.num_nodes+side,
            logical_message_bytes=total/8+side*data.num_nodes/8))
    other={n:s for n,s in stats.items() if not any(v in n for v in ('q_xw','q_gat_fea','weight_quant','q_gat_attn'))}
    active_bytes=0
    if wrapped.mode in ('unit','gate'):
        active_bytes=sum(code.basis.numel()*4+code.scale.numel()*4 for code in wrapped.codes)
        if wrapped.mode=='gate':
            active_bytes+=wrapped.gates.numel()*4
    elif wrapped.mode=='higher':
        active_bytes=sum(code.increments.numel()*code.increments.element_size() for code in wrapped.codes)
    return dict(layers=layers,active_extra_state_bytes=active_bytes,
        total_logical_activation_bytes=sum(s['effective_bit_sum']/8 for s in other.values())+sum(r['logical_message_bytes'] for r in layers),
        other_activation_quantizers=other,note='INT8 code traffic; weighted aggregation remains floating point. Logical quantizer tensor bytes, not peak memory or measured bandwidth.')


def test(args,c,data,directory,root,device):
    if not (root/'test_release.json').exists():
        raise RuntimeError('All fits must be released before test metrics')
    if (directory/'result.json').exists():
        return
    release=json.loads((root/'test_release.json').read_text())
    released=release['fits'][str(directory.relative_to(root))]
    for name,digest in released.items():
        assert sha(directory/name)==digest, name
    marker=directory/'test_started.json'
    if marker.exists():
        raise RuntimeError('An interrupted test is not silently repeated')
    result=json.loads((directory/'fit_result.json').read_text())
    backbone=build(c,data,args.seed).to(device)
    codes=torch.load(directory/'calibration.pt',map_location=device,weights_only=False)
    wrapped=GatedModel(backbone,c,codes).to(device)
    wrapped.load_state_dict(torch.load(directory/'gate/best.pt',map_location=device,weights_only=False)['model'])
    before=state_digest(wrapped,'all')
    write_json(marker,dict(started_unix=time.time(),modes=list(MODES)))
    metrics={}
    for mode in MODES:
        wrapped.mode=mode
        with QuantizationObserver(wrapped.backbone) as observer:
            value=evaluate(wrapped,data,'test')
        metrics[mode]=dict(**value,bit_audit=bit_audit(wrapped,c,data,observer.stats))
        write_json(directory/f'test_{mode}.json',metrics[mode])
    assert state_digest(wrapped,'all')==before
    for p,digest in result['base']['source_hashes'].items():
        assert sha(p)==digest
    result.update(test=metrics,test_metric_evaluations=len(MODES),test_state_unchanged=True,source_unchanged=True)
    write_json(directory/'result.json',result)
    print(c['dataset'],c['architecture'],args.seed,'complete', {k:v['accuracy'] for k,v in metrics.items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--phase',choices=('fit','test'),default='fit')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device=require_cuda()
    root=args.root.resolve()
    directory=root/args.dataset/args.model/f'seed_{args.seed}'
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=configuration(args.dataset,args.model)
    if args.smoke:
        c.update(baseline_epochs=2,gate_epochs=2)
    path=directory/'config.json'
    if path.exists():
        assert json.loads(path.read_text())==c
    else:
        write_json(path,c)
    data,manifest=load_data(c,device)
    split_path=directory/'split.json'
    if split_path.exists():
        assert json.loads(split_path.read_text())==manifest
    else:
        write_json(split_path,manifest)
    if args.phase=='fit':
        fit(args,c,data,directory,root,device)
    else:
        test(args,c,data,directory,root,device)


if __name__=='__main__':
    main()

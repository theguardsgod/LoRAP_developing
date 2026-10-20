"""GPU-only adaptation; optional teacher is never used by predict()."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from ..full_run import atomic_save,cpu_tree,require_cuda
from ..models import state_digest
from ..residual_adapt.run import update_best,verify_selection
from ..residual_channel.experiment import sha
from ..residual_transfer.models import build,forward
from ..residual_transfer.channel import quantize,metadata
from ..residual_transfer.run import load_data,evaluate,finite_gradients,check_stop,save_epoch
from ..run import write_json
from .config import ROOT,SOURCE,DATASETS,MODELS,VARIANTS,configuration
from .model import QuantOnlyModel,calibrate,local_target,PredictionCallback
from .diagnostics import Diagnostics


def load_source(dataset,architecture,seed,device):
    path=SOURCE/dataset/architecture/f'seed_{seed}'
    names=('config.json','split.json','baseline/best.pt','baseline/result.json','result.json')
    hashes={name:sha(path/name) for name in names}
    bc=json.loads((path/'config.json').read_text())
    data,split=load_data(bc,device)
    assert split==json.loads((path/'split.json').read_text())
    b=build(bc,data,seed).to(device).eval()
    b.load_state_dict(torch.load(path/'baseline/best.pt',map_location=device,weights_only=False)['model'])
    b.requires_grad_(False)
    base=json.loads((path/'result.json').read_text())
    val=evaluate(b,data,'val',bc)
    assert val['loss']==base['validation']['none']['loss']
    assert val['accuracy']==base['validation']['none']['accuracy']
    return b,bc,data,split,dict(path=str(path),hashes=hashes,validation=val,test=base['test']['none'])


@torch.no_grad()
def fixed_input_diagnostics(wrapped,data,c):
    """Each predictor sees the SAME unadapted backbone activations for comparison."""
    collector=Diagnostics(data.val_mask,c)
    class Capture:
        def quantize(self,i,x,quantizer): return quantize(x,quantizer)
        def __call__(self,i,x,q,op,quantizer):
            _,delta=wrapped.apply_quantized(i,q,op,*metadata(quantizer))
            collector(i,local_target(x,q,op),delta)
            return op(q)
    forward(wrapped.backbone,wrapped.backbone_config,data,Capture())
    return collector.rows


def export(wrapped,c,data,seed,kind,path):
    stats=[{n:v for n,v in p.named_buffers()} for p in wrapped.predictors]
    inference_config={k:c[k] for k in ('context_clip','rank','num_prompts')}
    atomic_save(cpu_tree(dict(backbone_config=wrapped.backbone_config,prompt_config=inference_config,
        dimensions=dict(nodes=data.num_nodes,features=data.num_features,classes=int(data.y.max())+1),
        seed=seed,kind=kind,stats=stats,backbone=wrapped.backbone.state_dict(),
        predictors=wrapped.predictors.state_dict(),format='quantized-input-predictor-v1')),
        path)


def load_export(path,device):
    item=torch.load(path,map_location=device,weights_only=False)
    dims=item['dimensions']
    # Construction uses saved output dimensionality, never deployment labels.
    shape=SimpleNamespace(num_nodes=dims['nodes'],num_features=dims['features'],
        num_node_features=dims['features'],y=torch.tensor([dims['classes']-1]))
    b=build(item['backbone_config'],shape,item['seed']).to(device)
    b.load_state_dict(item['backbone'])
    wrapped=QuantOnlyModel(b,item['backbone_config'],item['prompt_config'],item['stats'],item['seed'],item['kind']).to(device)
    wrapped.predictors.load_state_dict(item['predictors'])
    return wrapped.eval()


@torch.no_grad()
def inference_audit(wrapped,data,path,device):
    expected=wrapped(data)
    loaded=load_export(path,device)
    # Raising teacher verifies no hidden reference call in exported prediction.
    with patch('runs.lorap_study.quant_predict.model.local_target',side_effect=AssertionError('teacher used during inference')):
        actual=loaded(data)
        assert torch.equal(expected,actual)
        class Poison(PredictionCallback):
            def __call__(self,i,x,q,op,quantizer):
                return super().__call__(i,torch.full_like(x,float('nan')),q,op,quantizer)
        poisoned=forward(loaded.backbone,loaded.backbone_config,data,Poison(loaded))
        assert torch.equal(actual,poisoned)
        unlabeled=data.clone();del unlabeled.y
        assert torch.equal(actual,loaded(unlabeled))
    return dict(export_exact=True,poisoned_prequant_argument_exact=True,
        teacher_disabled_exact=True,label_free_inference_exact=True)


def fit(args,c,b,bc,data,split,base,directory,root,device):
    if (directory/'fit_result.json').exists(): return
    frozen=state_digest(b,'all')
    stats,scales,records=calibrate(b,bc,c,data)
    atomic_save(cpu_tree(dict(inference_stats=stats,training_loss_scales=scales)),directory/'calibration.pt')
    write_json(directory/'calibration.json',records)
    rows={};initials={}
    for kind in VARIANTS:
        path=directory/kind;path.mkdir(exist_ok=True)
        wrapped=QuantOnlyModel(b,bc,c,stats,args.seed,kind).to(device).eval()
        initials[kind]=state_digest(wrapped.predictors,'all')
        if kind=='error': assert initials['task']==initials['error']
        initial_val=evaluate(wrapped,data,'val')
        assert initial_val==base['validation']
        initial_buffers={n:v.clone() for n,v in wrapped.predictors.named_buffers()}
        opt=torch.optim.Adam(wrapped.predictors.parameters(),lr=c['learning_rate'],weight_decay=0.)
        history,best,start=[],None,0
        if (path/'last.pt').exists():
            last=torch.load(path/'last.pt',map_location=device,weights_only=False)
            wrapped.predictors.load_state_dict(last['model']);opt.load_state_dict(last['optimizer'])
            history,best,start=last['history'],cpu_tree(last['best']),last['epoch']+1
        for epoch in range(start,c['epochs']+1):
            check_stop(root)
            train=None
            if epoch:
                opt.zero_grad(set_to_none=True)
                if kind=='error':
                    logits,aux=wrapped.supervised_forward(data,data.train_mask,scales)
                else:
                    logits=wrapped(data);aux=None
                ce=F.cross_entropy(logits[data.train_mask],data.y[data.train_mask])
                loss=ce if aux is None else ce+c['error_loss_weight']*aux
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
                loss.backward();finite_gradients(wrapped.predictors)
                assert all(p.grad is None for p in b.parameters())
                opt.step()
                train=dict(loss=float(loss.detach()),ce=float(ce.detach()),
                    normalized_error_loss=None if aux is None else float(aux.detach()))
            # Selection always invokes the reference-free forward.
            val=evaluate(wrapped,data,'val')
            best=update_best(best,val['loss'],epoch,wrapped.predictors)
            history.append(dict(epoch=epoch,validation=val,train=train))
            last=dict(epoch=epoch,history=history,best=best,model=cpu_tree(wrapped.predictors.state_dict()),optimizer=cpu_tree(opt.state_dict()))
            save_epoch(path,last,best)
            write_json(directory/'progress.json',dict(variant=kind,epoch=epoch,total=c['epochs'],best_epoch=best['epoch']))
            if epoch%25==0:
                print(c['dataset'],c['architecture'],args.seed,kind,epoch,val,flush=True)
        verify_selection(last,best)
        wrapped.predictors.load_state_dict(best['model'])
        assert state_digest(b,'all')==frozen
        assert all(torch.equal(v,initial_buffers[n]) for n,v in wrapped.predictors.named_buffers())
        selected=evaluate(wrapped,data,'val')
        assert selected['loss']==best['validation_loss']
        with torch.no_grad():
            plain=wrapped(data)
            diagnostic=Diagnostics(data.val_mask,c)
            observed=wrapped.diagnose(data,diagnostic)
            assert torch.equal(plain,observed)
            training_diagnostic=Diagnostics(data.train_mask,c)
            wrapped.diagnose(data,training_diagnostic)
        export(wrapped,c,data,args.seed,kind,path/'inference.pt')
        audit=inference_audit(wrapped,data,path/'inference.pt',device)
        rows[kind]=dict(best_epoch=best['epoch'],validation=selected,last_validation=history[-1]['validation'],
            initial_prompt_sha256=initials[kind],selected_prompt_sha256=state_digest(wrapped.predictors,'all'),
            frozen_backbone_sha256=frozen,backbone_unchanged=True,normalization_unchanged=True,
            diagnostic_switch_exact=True,inference_audit=audit,
            parameters=sum(p.numel() for p in wrapped.predictors.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in wrapped.predictors.parameters()),
            buffer_bytes=sum(p.numel()*p.element_size() for p in wrapped.predictors.buffers()),
            layers=[dict(parameters=p.parameter_count,target=p.target_parameters,relative_gap=p.parameter_gap) for p in wrapped.predictors],
            local_validation=diagnostic.rows,local_training=training_diagnostic.rows,
            fixed_baseline_input_validation=fixed_input_diagnostics(wrapped,data,c),
            extra_residual_message_bits=0)
        write_json(path/'fit_result.json',rows[kind])
    for name,digest in base['hashes'].items(): assert sha(Path(base['path'])/name)==digest
    write_json(directory/'fit_result.json',dict(dataset=c['dataset'],architecture=c['architecture'],seed=args.seed,
        config=c,base=base,variants=rows,device=str(device),gpu=torch.cuda.get_device_name(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),test_metric_evaluations=0))
    write_json(directory/'progress.json',dict(phase='fit_complete'))


def test(args,c,data,directory,root,device):
    if (directory/'result.json').exists(): return
    release=json.loads((root/'test_release.json').read_text())
    for name,digest in release['fits'][str(directory.relative_to(root))].items():
        assert sha(directory/name)==digest
    marker=directory/'test_started.json'
    if marker.exists(): raise RuntimeError('Interrupted test is not silently repeated')
    write_json(marker,dict(started_unix=time.time(),variants=list(VARIANTS)))
    result=json.loads((directory/'fit_result.json').read_text())
    for kind in VARIANTS:
        wrapped=load_export(directory/kind/'inference.pt',device)
        frozen=state_digest(wrapped,'all')
        with patch('runs.lorap_study.quant_predict.model.local_target',side_effect=AssertionError('teacher in test')):
            metrics=evaluate(wrapped,data,'test')
        assert state_digest(wrapped,'all')==frozen
        result['variants'][kind]['test']=metrics
        write_json(directory/kind/'test.json',metrics)
    result['test_metric_evaluations']=len(VARIANTS)
    write_json(directory/'result.json',result)
    print(c['dataset'],c['architecture'],args.seed,'tested',
          {k:result['variants'][k]['test']['accuracy'] for k in VARIANTS},flush=True)


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
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
    device=require_cuda()
    root=args.root.resolve();directory=root/args.dataset/args.model/f'seed_{args.seed}'
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=configuration(args.dataset,args.model,args.smoke)
    b,bc,data,split,base=load_source(args.dataset,args.model,args.seed,device)
    for name,value in (('config.json',c),('split.json',split)):
        path=directory/name
        if path.exists(): assert json.loads(path.read_text())==value
        else: write_json(path,value)
    if args.phase=='fit': fit(args,c,b,bc,data,split,base,directory,root,device)
    else: test(args,c,data,directory,root,device)


if __name__=='__main__': main()

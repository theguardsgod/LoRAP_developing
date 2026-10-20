"""Full-split frozen fitting and validation-only candidate selection."""
import argparse,json,time
from pathlib import Path
import torch
from .registry import CELLS,METHODS,candidates
from .data import Bundle,labels_and_output
from .backends import build,load_state,predict
from .adapters import FrozenModel,calibrate,count
from .common import (cpu_tree,digest,save,write_json,config_hash,seed_all,rng_state,
                     restore_rng,consider,finite_gradients,file_lock)
from .train import evaluate,task_loss


def calibration(root,c,b,model,seed,device,limit=None):
    path=Path(root)/c['cell']/f'seed_{seed}'/'calibration.pt';key=digest(model)
    with file_lock(path.with_suffix('.lock')):
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['backbone_sha256']!=key:raise ValueError('Calibration backbone changed')
            return saved['stats']
        stats=calibrate(model,c,b,device,seed,max_graphs=128 if limit is None else min(limit,128))
        if digest(model)!=key:raise AssertionError('Calibration mutated frozen state')
        save(path,dict(stats=stats,backbone_sha256=key,split='train',max_nodes=4096,max_graphs=128))
        return cpu_tree(stats)


def epoch(model,c,b,seed,device,number,opt,hp,stage,limit=None):
    model.train();totals=torch.zeros(4,device=device,dtype=torch.float64);count=0
    teacher=model.kind=='error' or stage=='predictor_warmup'
    for batch in b.batches('train',seed,number,device,limit=limit):
        opt.zero_grad(set_to_none=True);out,_=model(batch,teacher=teacher)
        logits,y=labels_and_output(c,batch,out,'train');task=task_loss(c,logits,y)
        error=torch.stack(model.losses).mean() if teacher else task.new_zeros(())
        objective=error if stage=='predictor_warmup' else task+hp['error_weight']*error
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite adaptation objective')
        objective.backward();finite_gradients(model);opt.step()
        metric=(logits.flatten()-y).abs().sum() if c['regression'] else (logits.argmax(1)==y).sum()
        totals+=torch.stack((task.detach().double()*len(y),metric.detach().double(),error.detach().double()*len(y),objective.detach().double()*len(y)));count+=len(y)
    values=totals.cpu().tolist()
    return dict(loss=values[0]/count,metric=values[1]/count*(1 if c['regression'] else 100),
                error_loss=values[2]/count,objective=values[3]/count,examples=count)


@torch.no_grad()
def validation_error(model,c,b,seed,device,limit=None):
    model.eval();total=0.;nodes=0
    def observer(i,aggregate,context,target,delta):
        nonlocal total,nodes
        mask=batch.val_mask if c['node'] else slice(None)
        difference=(delta[mask]-target.flatten(1)[mask].to(delta)).square().mean(1)/model.loss_scales[i].to(delta)
        total+=float(difference.sum());nodes+=len(difference)
    for batch in b.batches('val',seed,0,device,limit=limit):model(batch,teacher=True,observer=observer)
    return total/nodes


def fit(root,c,seed,method,hp,device,limit=None):
    basepath=Path(root)/c['cell']/f'seed_{seed}'
    path=basepath/method/f"candidate_{hp['candidate']}";path.mkdir(parents=True,exist_ok=True)
    config=dict(cell_config=c,seed=seed,method=method,hyperparameters=hp,subset_limit=limit);key=config_hash(config)
    if (path/'done.json').exists():
        done=json.loads((path/'done.json').read_text())
        if done['config_hash']!=key:raise ValueError('Completed adaptation configuration changed')
        return done
    if (path/'config.json').exists() and json.loads((path/'config.json').read_text())!=config:
        raise ValueError('Interrupted adaptation configuration changed')
    write_json(path/'config.json',config);seed_all(seed+915000)
    if not (basepath/'baseline/done.json').exists():raise FileNotFoundError('Complete the paired backbone first')
    b=Bundle(c);backbone=build(c,b,seed).to(device)
    checkpoint=torch.load(basepath/'baseline/best.pt',map_location='cpu',weights_only=False)
    load_state(backbone,checkpoint['model']);base_digest=digest(backbone)
    stats=calibration(root,c,b,backbone,seed,device,limit)
    model=FrozenModel(backbone,c,stats,hp,method,seed).to(device)
    # Gate-only warmup learns a quantized-input error predictor. Its outer gates
    # remain fixed until the following task-only stage.
    stages=['predictor_warmup','adapter'] if method=='gate_only' else ['adapter']
    complete_state=None
    if (path/'last.pt').exists():
        complete_state=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        if complete_state['config_hash']!=key:raise ValueError('Resume configuration mismatch')
        load_state(model,complete_state['model'])
    history=[] if complete_state is None else complete_state['history']
    stage_results={}
    if complete_state is not None:stage_results=complete_state['stage_results']
    for stage in stages:
        if stage in stage_results:continue
        if stage=='predictor_warmup':
            model.bank.outer_gates.requires_grad_(False)
            maximum=50 if c['node'] else 20
        else:
            if method=='gate_only':model.bank.freeze_for_gates()
            maximum=c['adapter_epochs']
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.Adam(params,lr=hp['lr'],weight_decay=0.)
        start_epoch=1;best=None
        if complete_state is not None and complete_state['stage']==stage:
            start_epoch=complete_state['epoch']+1;best=complete_state['best'];opt.load_state_dict(complete_state['optimizer'])
            restore_rng(complete_state['rng'],device)
        if best is None:
            val=evaluate(model,c,b,device,seed,limit=limit)
            criterion=validation_error(model,c,b,seed,device,limit) if stage=='predictor_warmup' else val['loss']
            best=consider(best,criterion,0,model)
            save(path/(stage+'_best.pt'),best)
            if stage=='adapter':write_json(path/'initialization.json',dict(validation=val,backbone_sha256=base_digest,
                 state_sha256=digest(model),adapter_parameters=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('backbone.')),
                 trainable_parameters=sum(p.numel() for p in params),
                 adapter_buffer_bytes=sum(t.numel()*t.element_size() for n,t in model.named_buffers() if not n.startswith('backbone.')),
                 mlp_parameter_matches=model.bank.parameter_matches if method=='mlp' else None,
                 lora_targets=model.lora_targets))
        for number in range(start_epoch,maximum+1):
            start=time.monotonic();train=epoch(model,c,b,seed,device,number,opt,hp,stage,limit)
            val=evaluate(model,c,b,device,seed,limit=limit)
            criterion=validation_error(model,c,b,seed,device,limit) if stage=='predictor_warmup' else val['loss']
            previous=best;best=consider(best,criterion,number,model)
            if best is not previous:save(path/(stage+'_best.pt'),best)
            if digest(backbone)!=base_digest:raise AssertionError('Frozen backbone state changed')
            history.append(dict(stage=stage,epoch=number,train=train,val=val,criterion=criterion,seconds=time.monotonic()-start))
            save(path/'last.pt',dict(config_hash=key,stage=stage,epoch=number,model=model.state_dict(),optimizer=opt.state_dict(),
                 best=best,history=history,rng=rng_state(device),stage_results=stage_results))
            write_json(path/'history.json',history)
            write_json(path/'progress.json',dict(cell=c['cell'],seed=seed,method=method,candidate=hp['candidate'],stage=stage,
                       epoch=number,total=maximum,best_criterion=best['validation_loss'],latest_val=val))
            if number==1 or number%25==0:print(f"{c['cell']} s{seed} {method}/c{hp['candidate']} {stage} {number}/{maximum} val={val['loss']:.5f} metric={val['metric']:.3f}",flush=True)
            if stage=='adapter' and number>=c['min_epochs'] and number-best['epoch']>=c['stop_patience']:break
        load_state(model,best['model'])
        stage_results[stage]=dict(epoch=best['epoch'],criterion=best['validation_loss'])
        # Persist transitions as well: interruption must not repeat warmup.
        if stage=='predictor_warmup':
            save(path/'warmup_selected.pt',best)
            complete_state=None
        else:save(path/'best.pt',best)
    final=evaluate(model,c,b,device,seed,limit=limit)
    if abs(final['loss']-best['validation_loss'])>1e-6:raise AssertionError('Frozen best checkpoint replay differs')
    result=dict(cell=c['cell'],seed=seed,method=method,hyperparameters=hp,config_hash=key,
                validation=final,selected_epoch=best['epoch'],stage_results=stage_results,
                backbone_sha256=base_digest,backbone_unchanged=digest(backbone)==base_digest,
                checkpoint_sha256=digest(model),test_evaluations=0,full_split=limit is None,
                seconds=sum(row['seconds'] for row in history))
    write_json(path/'done.json',result);model.close()
    return result


def select_cell(root,c):
    path=Path(root)/c['cell'];selection={}
    for method in METHODS:
        rows=[]
        for hp in candidates(method):
            file=path/'seed_41'/method/f"candidate_{hp['candidate']}"/'done.json'
            if not file.exists():raise FileNotFoundError(file)
            row=json.loads(file.read_text())
            if row['test_evaluations']!=0 or not row['full_split']:raise ValueError('Invalid tuning run')
            rows.append(row)
        best=min(rows,key=lambda r:(r['validation']['loss'],r['hyperparameters']['candidate']))
        selection[method]=dict(hyperparameters=best['hyperparameters'],validation=best['validation'],
                               source_config_hash=best['config_hash'])
    record=dict(cell=c['cell'],selection=selection,criterion='validation loss',test_evaluations=0)
    write_json(path/'selection.json',record);return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='cuda:2')
    p.add_argument('--cells',nargs='+',required=True);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--methods',nargs='+',default=list(METHODS));p.add_argument('--candidate',type=int)
    p.add_argument('--epochs',type=int);p.add_argument('--smoke-limit',type=int)
    a=p.parse_args();torch.set_num_threads(4)
    for name in a.cells:
        c=dict(CELLS[name])
        if a.epochs is not None:c['adapter_epochs']=a.epochs
        for method in a.methods:
            hps=candidates(method) if a.seed==41 else [json.loads((Path(a.root)/name/'selection.json').read_text())['selection'][method]['hyperparameters']]
            if a.candidate is not None:hps=[h for h in hps if h['candidate']==a.candidate]
            for hp in hps:fit(a.root,c,a.seed,method,hp,a.device,a.smoke_limit)


if __name__=='__main__':main()

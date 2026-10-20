"""Training/validation entrypoint. Contains no calls to the test split."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .registry import CELLS
from .data import Bundle,labels_and_output
from .backends import build,predict,optimizer,project,load_state
from .common import (write_json,save,digest,config_hash,seed_all,rng_state,restore_rng,
                     finite_gradients,consider)


def task_loss(c,out,y):
    return F.l1_loss(out.reshape(-1),y) if c['regression'] else F.cross_entropy(out,y)


@torch.no_grad()
def evaluate(model,c,b,device,seed,split='val',limit=None,allow_test=False):
    if split=='test' and not allow_test:raise RuntimeError('Test gate is locked')
    model.eval();totals=torch.zeros(3,device=device,dtype=torch.float64);count=0;batches=0
    for batch in b.batches(split,seed,0,device,limit=limit,allow_test=allow_test):
        out,budget=predict(model,c,batch);out,y=labels_and_output(c,batch,out,split)
        loss=task_loss(c,out,y)
        metric=(out.reshape(-1)-y).abs().sum() if c['regression'] else (out.argmax(1)==y).sum()
        totals+=torch.stack((loss.double()*len(y),metric.double(),budget.double()));count+=len(y);batches+=1
    values=totals.cpu().tolist()
    return dict(loss=values[0]/count,metric=values[1]/count*(1 if c['regression'] else 100),
                storage_proxy=values[2]/batches,examples=count)


def train_epoch(model,c,b,device,seed,epoch,opt,limit=None):
    model.train();totals=torch.zeros(3,device=device,dtype=torch.float64);count=0
    for batch in b.batches('train',seed,epoch,device,limit=limit):
        opt.zero_grad(set_to_none=True);out,budget=predict(model,c,batch)
        out,y=labels_and_output(c,batch,out,'train');loss=task_loss(c,out,y)
        objective=loss+c['a_loss']*F.relu(budget-c['a_storage']).square()
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite training loss')
        objective.backward();finite_gradients(model);opt.step();project(model,c)
        metric=(out.reshape(-1)-y).abs().sum() if c['regression'] else (out.argmax(1)==y).sum()
        totals+=torch.stack((loss.detach().double()*len(y),metric.detach().double(),objective.detach().double()*len(y)));count+=len(y)
    v=totals.cpu().tolist()
    return dict(loss=v[0]/count,metric=v[1]/count*(1 if c['regression'] else 100),objective=v[2]/count,examples=count)


def baseline(root,c,seed,device,limit=None):
    override_path=Path(root)/c['cell']/'baseline_override.json'
    if override_path.exists():
        overrides=json.loads(override_path.read_text())['hyperparameters']
        if set(overrides)-{'lr','weight_decay'}:raise ValueError('Unregistered backbone overrides')
        c=dict(c,**overrides)
    path=Path(root)/c['cell']/f'seed_{seed}'/'baseline';path.mkdir(parents=True,exist_ok=True)
    run_config=dict(c,seed=seed,subset_limit=limit,stage='baseline');key=config_hash(run_config)
    if (path/'done.json').exists():
        result=json.loads((path/'done.json').read_text())
        if result['config_hash']!=key:raise ValueError('Completed configuration changed')
        return result
    if (path/'config.json').exists() and json.loads((path/'config.json').read_text())!=run_config:
        raise ValueError('Configuration differs from interrupted run')
    write_json(path/'config.json',run_config)
    seed_all(seed);b=Bundle(c);write_json(path/'split.json',b.manifest())
    model=build(c,b,seed).to(device);initial=digest(model);opt=optimizer(model,c)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5)
               if c['framework']=='A2Q' and not c['node'] else
               torch.optim.lr_scheduler.StepLR(opt,50,.5) if c['framework']!='A2Q' else None)
    best=None;history=[];start_epoch=1
    if (path/'last.pt').exists():
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        if last['config_hash']!=key:raise ValueError('Resume hash mismatch')
        load_state(model,last['model']);opt.load_state_dict(last['optimizer'])
        if scheduler:scheduler.load_state_dict(last['scheduler'])
        restore_rng(last['rng'],device);best=last['best'];history=last['history'];start_epoch=last['epoch']+1
    for epoch in range(start_epoch,c['baseline_epochs']+1):
        start=time.monotonic();training=train_epoch(model,c,b,device,seed,epoch,opt,limit)
        val=evaluate(model,c,b,device,seed,limit=limit)
        previous=best;best=consider(best,val['loss'],epoch,model)
        if best is not previous:save(path/'best.pt',best)
        if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):scheduler.step(val['loss'])
        elif scheduler:scheduler.step()
        record=dict(epoch=epoch,train=training,val=val,seconds=time.monotonic()-start,
                    learning_rates={g['name']:g['lr'] for g in opt.param_groups})
        history.append(record)
        save(path/'last.pt',dict(config_hash=key,epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),
             scheduler=scheduler.state_dict() if scheduler else None,rng=rng_state(device),best=best,history=history))
        write_json(path/'history.json',history)
        write_json(path/'progress.json',dict(cell=c['cell'],seed=seed,stage='baseline',epoch=epoch,
                   total=c['baseline_epochs'],best_val_loss=best['validation_loss'],latest_val=val))
        if epoch==1 or epoch%10==0:print(f"{c['cell']} seed={seed} baseline {epoch}/{c['baseline_epochs']} val_loss={val['loss']:.5f} val_metric={val['metric']:.3f}",flush=True)
    load_state(model,best['model']);val=evaluate(model,c,b,device,seed,limit=limit)
    if abs(val['loss']-best['validation_loss'])>1e-5:raise AssertionError('Best model validation replay differs')
    result=dict(cell=c['cell'],seed=seed,config_hash=key,initial_sha256=initial,
                selected_epoch=best['epoch'],validation=val,backbone_sha256=digest(model),
                test_evaluations=0,full_split=limit is None,seconds=sum(r['seconds'] for r in history))
    write_json(path/'done.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='cuda:2')
    p.add_argument('--cells',nargs='+',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[41])
    p.add_argument('--epochs',type=int);p.add_argument('--smoke-limit',type=int)
    args=p.parse_args();torch.set_num_threads(4)
    for name in args.cells:
        c=dict(CELLS[name])
        if args.epochs is not None:c['baseline_epochs']=args.epochs
        for seed in args.seeds:baseline(args.root,c,seed,args.device,args.smoke_limit)


if __name__=='__main__':main()

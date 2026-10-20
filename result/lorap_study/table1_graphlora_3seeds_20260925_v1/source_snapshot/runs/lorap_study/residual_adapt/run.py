"""GPU short adaptation, epoch-zero selection and post-hoc BN interventions."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from ..data import stratified_indices
from ..full_models import build_full_backbone
from ..full_run import atomic_save, cpu_tree, require_cuda, sha_config
from ..full_validation import validation_reproduction
from ..models import state_digest
from ..run import write_json
from ..residual_channel.core import error_terms
from ..residual_channel.experiment import sha
from .model import AdaptedGIN, digest_subset, is_bn_stat, is_quantizer, optimizer_for, recalibrate_bn

ROOT = Path("/home/cyliu/code/A-2Q/result/lorap_study/residual_adapt_20260913_v1")
SOURCE = Path("/home/cyliu/code/A-2Q/result/lorap_study/full_gpu_20260909_v1")
CALIBRATION = Path("/home/cyliu/code/A-2Q/result/lorap_study/residual_channel_20260912_v2")
VARIANTS = ("gate_only", "none_ft", "gate_ft", "higher16_ft")


def data_loader(data, batch_size=128):
    return DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0)


def epoch_order(n, seed, epoch):
    return torch.randperm(n,generator=torch.Generator().manual_seed(913000+seed+epoch)).tolist()


def update_best(best, loss, epoch, model):
    if not np.isfinite(loss):
        raise FloatingPointError("Nonfinite validation loss")
    if best is None or loss < best["validation_loss"]:
        return dict(epoch=epoch,validation_loss=loss,model=cpu_tree(model.state_dict()))
    return best


def verify_selection(last, best):
    history = last["history"]
    assert [r["epoch"] for r in history]==list(range(last["epoch"]+1))
    selected = min(history,key=lambda r:r["validation"]["loss"])
    assert best["epoch"]==last["best"]["epoch"]==selected["epoch"]
    assert best["validation_loss"]==last["best"]["validation_loss"]==selected["validation"]["loss"]
    assert best["model"].keys()==last["best"]["model"].keys()
    for name,value in best["model"].items():
        assert bool(torch.isfinite(value).all())
        assert torch.equal(value,last["best"]["model"][name])


class LocalDiagnostics:
    """Actual-current-input error, including gate amplitude; no PCA-loss label."""
    def __init__(self):
        self.totals = [{} for _ in range(4)]

    def __call__(self,i,x,q,edges,eps,delta):
        terms = error_terms(x-q,torch.zeros_like(x) if delta is None else delta,edges,eps)
        state = self.totals[i]
        for key,value in terms.items():
            state[key] = state.get(key,0)+value
        state["nodes"] = state.get("nodes",0)+len(x)

    def result(self):
        rows=[]
        for i,values in enumerate(self.totals):
            s={k:float(v) for k,v in values.items()}
            s["mse_reduction"]=1-s["after_sq"]/max(s["before_sq"],1e-30)
            s["cosine"]=s["dot"]/max((s["before_sq"]*s["correction_sq"])**.5,1e-30)
            rows.append(dict(layer=i+1,**s))
        return rows


@torch.no_grad()
def evaluate(model, data, device, diagnostics=False):
    model.eval()
    totals=torch.zeros(3,device=device,dtype=torch.float64)
    collector=LocalDiagnostics() if diagnostics else None
    for batch in data_loader(data):
        batch=batch.to(device)
        logits=model(batch,collector)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite validation outputs")
        y=batch.y.reshape(-1)
        totals+=torch.stack((F.cross_entropy(logits,y,reduction="sum").double(),
                            (logits.argmax(1)==y).sum(),y.new_tensor(len(y)))).double()
    loss,correct,n=totals.cpu().tolist()
    result=dict(loss=loss/n,accuracy=correct/n,examples=int(n))
    if collector is not None:
        result["local_diagnostics"]=collector.result()
    return result


def train_epoch(model, data, optimizer, config, seed, epoch, device):
    model.training_mode()
    order=epoch_order(len(data),seed,epoch)
    totals=torch.zeros(3,device=device,dtype=torch.float64)
    gate_gradient_sum=torch.zeros(4,device=device) if model.gates is not None else None
    batches=0
    for batch in data_loader(data[order]):
        batch=batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch)
        y=batch.y.reshape(-1)
        loss=F.cross_entropy(logits,y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        gradients=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if not gradients or not bool(torch.stack(gradients).all()):
            raise FloatingPointError("Nonfinite or missing gradients")
        if gate_gradient_sum is not None:
            gate_gradient_sum+=model.gates.grad.detach()
        optimizer.step()
        model.project_gates()
        totals+=torch.stack((loss.detach().double()*len(y),(logits.argmax(1)==y).sum(),y.new_tensor(len(y)))).double()
        batches+=1
    loss,correct,n=totals.cpu().tolist()
    import hashlib
    return dict(loss=loss/n,accuracy=correct/n,examples=int(n),batches=batches,
        order_sha256=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),
        average_gate_gradient=(gate_gradient_sum/batches).cpu().tolist() if gate_gradient_sum is not None else None)


def bn_interventions(model, best, initial, training, validation, indices, device, directory):
    rows={}
    # All conditions use the same training-only calibration graphs and algorithm.
    # Restoring a state before each condition prevents accumulated BN updates.
    for name,state,gate in (("gate_only_bn",best["model"],None),
                            ("none_bn",initial,0.),("unit_gate_bn",initial,1.)):
        model.load_state_dict(state)
        if gate is not None:
            with torch.no_grad():
                model.gates.fill_(gate)
        unchanged=digest_subset(model,lambda n:not is_bn_stat(n))
        before=evaluate(model,validation,device)
        records=recalibrate_bn(model,data_loader(training[indices]),device)
        after=evaluate(model,validation,device,True)
        assert unchanged==digest_subset(model,lambda n:not is_bn_stat(n))
        atomic_save(dict(model=cpu_tree(model.state_dict()),validation=after),directory/f"{name}.pt")
        rows[name]=dict(before=before,after=after,gates=model.gate_values(),
                        bn_statistics=records,non_bn_state_unchanged=True)
    model.load_state_dict(best["model"])
    return rows


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device=require_cuda()
    directory=args.root/args.dataset/args.variant/f"seed_{args.seed}"
    directory.mkdir(parents=True,exist_ok=True)
    source=args.source/args.dataset/"none"/f"seed_{args.seed}"
    calibration_path=args.calibration/args.dataset/f"seed_{args.seed}"
    prior=json.loads((source/"config.json").read_text())
    source_hashes={str(source/name):sha(source/name) for name in ("best.pt","last.pt","result.json","test_started.json","config.json")}
    source_hashes[str(calibration_path/"calibration.pt")]=sha(calibration_path/"calibration.pt")
    config=dict(dataset=args.dataset,variant=args.variant,seed=args.seed,epochs=args.epochs,
        training_graph_limit=args.train_limit,validation_graph_limit=args.val_limit,
        backbone_lr=prior["lr"]*.1,gate_lr=.01,batch_size=128,
        frozen_quantizers=True,gate_projection=[0.,1.],side_gradient="stop-gradient true residual and integer transport",
        selection="minimum validation CE including epoch zero; earlier on ties",test_evaluations=0,
        bn_calibration_graphs=1024,source_hashes=source_hashes)
    fingerprint=sha_config(config)
    if (directory/"result.json").exists():
        result=json.loads((directory/"result.json").read_text())
        if result["config_sha256"]!=fingerprint:
            raise ValueError("Completed run configuration changed")
        return
    write_json(directory/"config.json",config)
    data={s:GNNBenchmarkDataset(prior["data_root"],args.dataset,split=s) for s in ("train","val")}
    labels=data["train"]._data.y.cpu().numpy().reshape(-1)
    train_indices=stratified_indices(labels,args.train_limit,913041)
    training=data["train"][train_indices]
    validation=data["val"] if args.val_limit is None else data["val"][:args.val_limit]
    bn_indices=stratified_indices(labels[train_indices],min(1024,len(training)),913042)
    write_json(directory/"indices.json",dict(training=train_indices,bn_indices_within_training=bn_indices,
        validation=list(range(len(validation)))))
    calibration=torch.load(calibration_path/"calibration.pt",map_location="cpu",weights_only=False)
    backbone=build_full_backbone(prior,args.seed,data)
    checkpoint=torch.load(source/"best.pt",map_location="cpu",weights_only=False)
    backbone.load_state_dict(checkpoint["model"])
    model=AdaptedGIN(backbone,calibration,args.variant).to(device)
    original_model_digest=state_digest(model.backbone,"all")
    initial=cpu_tree(model.state_dict())
    quantizer_digest=digest_subset(model,is_quantizer)
    code_digest=digest_subset(model,lambda n:n.startswith("codes."))
    optimizer=optimizer_for(model,config)
    history,best,elapsed,updates=[],None,0.,0
    start_epoch=0
    last_path=directory/"last.pt"
    if last_path.exists():
        saved=torch.load(last_path,map_location="cpu",weights_only=False)
        last=saved
        if saved["config_sha256"]!=fingerprint:
            raise ValueError("Resume configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history,best,elapsed,updates=saved["history"],saved["best"],saved["seconds"],saved["optimizer_updates"]
        start_epoch=saved["epoch"]+1
        atomic_save(best,directory/"best.pt")
        (directory/"epochs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in history))
    for epoch in range(start_epoch,args.epochs+1):
        if (args.root/"STOP").exists():
            raise InterruptedError("Paused at an epoch boundary")
        started=time.perf_counter()
        train=None if epoch==0 else train_epoch(model,training,optimizer,config,args.seed,epoch,device)
        if train:
            updates+=train["batches"]
        val=evaluate(model,validation,device)
        if epoch==0 and args.val_limit is None:
            phase_one=json.loads((calibration_path/"result.json").read_text())
            key="higher_16" if args.variant=="higher16_ft" else "none"
            validation_reproduction(phase_one["validation"][key]["loss"],val["loss"])
        if quantizer_digest!=digest_subset(model,is_quantizer) or code_digest!=digest_subset(model,lambda n:n.startswith("codes.")):
            raise AssertionError("Quantizer or residual calibration changed")
        if args.variant=="gate_only" and state_digest(model.backbone,"all")!=original_model_digest:
            raise AssertionError("Gate-only training changed the backbone or BN statistics")
        old_best_epoch=best["epoch"] if best else None
        best=update_best(best,val["loss"],epoch,model)
        elapsed+=time.perf_counter()-started
        history.append(dict(epoch=epoch,train=train,validation=val,gates=model.gate_values(),
                            selected=best["epoch"]!=old_best_epoch,seconds=elapsed))
        last=dict(epoch=epoch,model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
            history=history,best=best,seconds=elapsed,optimizer_updates=updates,config_sha256=fingerprint)
        atomic_save(last,last_path)
        if history[-1]["selected"]:
            atomic_save(best,directory/"best.pt")
        with (directory/"epochs.jsonl").open("a") as f:
            f.write(json.dumps(history[-1],allow_nan=False)+"\n")
        write_json(directory/"progress.json",dict(phase="fit",epoch=epoch,epochs=args.epochs,
            validation=val,gates=model.gate_values(),best_epoch=best["epoch"],seconds=elapsed,updated_unix=time.time()))
        print(f"{args.dataset} {args.variant} seed={args.seed} {epoch}/{args.epochs} CE={val['loss']:.6f} gates={model.gate_values()}",flush=True)
    verify_selection(last,best)
    model.load_state_dict(best["model"])
    selected=evaluate(model,validation,device,True)
    replay=validation_reproduction(best["validation_loss"],selected["loss"])
    if replay["delta"]!=0:
        raise AssertionError("Deterministic best checkpoint replay changed")
    first=next(iter(data_loader(validation))).to(device)
    with torch.no_grad():
        plain=model(first)
        observed=model(first,LocalDiagnostics())
        if not torch.equal(plain,observed):
            raise AssertionError("Diagnostic switch changed predictions")
        if model.gates is not None:
            bypass=model(first,bypass_messages=True)
            gates=model.gates.clone()
            model.gates.zero_()
            bypass_zero=model(first,bypass_messages=True)
            model.gates.copy_(gates)
            if not torch.equal(bypass,bypass_zero):
                raise AssertionError("Residual gate changed unquantized messages")
    gate_values=model.gate_values()
    write_json(directory/"selected_diagnostics.json",selected["local_diagnostics"])
    interventions={}
    if args.variant=="gate_only":
        write_json(directory/"progress.json",dict(phase="bn_interventions",epoch=args.epochs,updated_unix=time.time()))
        interventions=bn_interventions(model,best,initial,training,validation,bn_indices,device,directory)
        write_json(directory/"bn_interventions.json",interventions)
    for name,digest in source_hashes.items():
        if sha(name)!=digest:
            raise AssertionError("Source artifact changed")
    result=dict(dataset=args.dataset,variant=args.variant,seed=args.seed,config_sha256=fingerprint,
        epochs=args.epochs,training_graphs=len(training),validation_graphs=len(validation),
        initial_validation=history[0]["validation"],last_validation=history[-1]["validation"],
        selected_validation=selected,best_epoch=best["epoch"],gates=gate_values,
        seconds=elapsed,optimizer_updates=updates,test_evaluations=0,
        device=str(device),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),gpu_name=torch.cuda.get_device_name(),
        initial_backbone_sha256=original_model_digest,quantizer_sha256=quantizer_digest,code_sha256=code_digest,
        source_hashes=source_hashes,source_artifacts_unchanged=True,quantizers_unchanged=True,
        frozen_backbone_unchanged=args.variant=="gate_only",deterministic_replay=replay,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        new_gate_parameters=4 if model.gated else 0,
        extra_static_channel_bytes=3568 if model.gated else 0,
        extra_message_bits_per_node_per_layer=[16]*4 if model.gated else
            [int(len(c["channels"]["higher_16"])) for c in calibration] if args.variant=="higher16_ft" else [0]*4,
        bn_interventions={k:dict(before=v["before"],after={m:n for m,n in v["after"].items() if m!="local_diagnostics"},gates=v["gates"])
                          for k,v in interventions.items()})
    write_json(directory/"result.json",result)
    write_json(directory/"progress.json",dict(phase="complete",best_epoch=best["epoch"],selected_validation=selected))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--calibration",type=Path,default=CALIBRATION)
    parser.add_argument("--dataset",choices=("MNIST","CIFAR10"),required=True)
    parser.add_argument("--variant",choices=VARIANTS,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--train-limit",type=int)
    parser.add_argument("--val-limit",type=int)
    run(parser.parse_args())

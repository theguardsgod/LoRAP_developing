"""GPU audit of actual activation bit assignments on selected validation models."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.datasets import GNNBenchmarkDataset

from ..diagnostics import QuantizationObserver
from ..full_models import build_full_backbone
from ..full_run import require_cuda
from ..models import state_digest
from ..run import write_json
from ..residual_channel.experiment import sha
from .model import AdaptedGIN
from .run import ROOT, data_loader


@torch.no_grad()
def measure(model,validation,device):
    before=state_digest(model,"all")
    totals=torch.zeros(3,device=device,dtype=torch.float64)
    message=dict(elements=0,bits=0,side_bits=0)
    legacy=dict(current=torch.zeros((),device=device,dtype=torch.float64))
    budget_sum=torch.zeros((),device=device,dtype=torch.float64)
    batches=0

    def messages(i,x,q,edges,eps,delta):
        n,width=x.shape
        extra=int(getattr(model,f"upgrades_{i}").numel()) if model.variant=="higher16_ft" else 0
        message["elements"]+=n*width
        message["bits"]+=n*(width*4+extra)
        message["side_bits"]+=n*16 if model.gated else 0

    def capture_proxy(module,inputs,output):
        legacy["current"]=legacy["current"]+output[1].detach().double().reshape(())

    handles=[m.register_forward_hook(capture_proxy) for n,m in model.named_modules()
             if n.endswith("fea_quant") and hasattr(m,"gama")]
    try:
        with QuantizationObserver(model) as observer:
            for batch in data_loader(validation):
                batch=batch.to(device)
                legacy["current"]=torch.zeros((),device=device,dtype=torch.float64)
                logits=model(batch,observer=messages)
                y=batch.y.reshape(-1)
                totals+=torch.stack((F.cross_entropy(logits,y,reduction="sum").double(),
                                    (logits.argmax(1)==y).sum(),y.new_tensor(len(y)))).double()
                budget_sum+=legacy["current"]
                batches+=1
        # Message precision is counted explicitly: the mixed-bit control invokes
        # the quantizer functional, so module hooks do not see its message calls.
        feature={n:s for n,s in observer.stats.items() if s["kind"]=="activation" and ".q_xw" not in n}
        feature_bits=sum(s["effective_bit_sum"] for s in feature.values())
        feature_elements=sum(s["elements"] for s in feature.values())
    finally:
        for handle in handles:
            handle.remove()
    assert before==state_digest(model,"all")
    loss,correct,n=totals.cpu().tolist()
    return dict(loss=loss/n,accuracy=correct/n,examples=int(n),state_unchanged=True,
        mean_feature_effective_bits=feature_bits/feature_elements,
        mean_main_activation_effective_bits=(feature_bits+message["bits"])/(feature_elements+message["elements"]),
        feature_quantizer_logical_bytes=feature_bits/8,main_message_logical_bytes=message["bits"]/8,
        side_message_logical_bytes=message["side_bits"]/8,
        total_activation_logical_bytes=(feature_bits+message["bits"]+message["side_bits"])/8,
        legacy_storage_proxy_mean=float(budget_sum/batches),feature_assignments=feature,
        note="Logical quantizer tensor traffic over full validation, not peak memory; includes side codes separately.")


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device=require_cuda()
    root=args.root.resolve()
    directory=root/args.dataset/f"budget_seed_{args.seed}"
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"result.json").exists():
        return
    study=json.loads((root/"study.json").read_text())
    source=Path(study["source"])/args.dataset/"none"/f"seed_{args.seed}"
    calibration_path=Path(study["calibration"])/args.dataset/f"seed_{args.seed}/calibration.pt"
    config=json.loads((source/"config.json").read_text())
    calibration=torch.load(calibration_path,map_location="cpu",weights_only=False)
    original=torch.load(source/"best.pt",map_location="cpu",weights_only=False)["model"]
    validation=GNNBenchmarkDataset(config["data_root"],args.dataset,split="val")
    gate_dir=root/args.dataset/"gate_only"/f"seed_{args.seed}"
    cases=[("frozen","none_ft",None,None),("unit_gate","gate_only",None,1.),
           ("none_bn","none_ft",gate_dir/"none_bn.pt",None),
           ("unit_gate_bn","gate_only",gate_dir/"unit_gate_bn.pt",None),
           ("gate_only_bn","gate_only",gate_dir/"gate_only_bn.pt",None),
           ("higher16_frozen","higher16_ft",None,None)]
    for variant in ("gate_only","none_ft","gate_ft","higher16_ft"):
        cases.append((variant,variant,root/args.dataset/variant/f"seed_{args.seed}/best.pt",None))
    expected={}
    results={v:json.loads((root/args.dataset/v/f"seed_{args.seed}/result.json").read_text())
             for v in ("gate_only","none_ft","gate_ft","higher16_ft")}
    for variant,r in results.items():
        expected[variant]=r["selected_validation"]
    expected["frozen"]=results["none_ft"]["initial_validation"]
    expected["higher16_frozen"]=results["higher16_ft"]["initial_validation"]
    expected["unit_gate"]=results["gate_only"]["bn_interventions"]["unit_gate_bn"]["before"]
    for key,value in results["gate_only"]["bn_interventions"].items():
        expected[key]=value["after"]
    rows={}
    source_hashes={str(source/"best.pt"):sha(source/"best.pt"),str(calibration_path):sha(calibration_path)}
    for name,variant,path,gate in cases:
        backbone=build_full_backbone(config,args.seed,{"train":[validation[0]]})
        backbone.load_state_dict(original)
        model=AdaptedGIN(backbone,calibration,variant).to(device).eval().requires_grad_(False)
        if path is not None:
            source_hashes[str(path)]=sha(path)
            state=torch.load(path,map_location=device,weights_only=False)["model"]
            if name=="none_bn":
                state={k:v for k,v in state.items() if k.startswith("backbone.")}
            model.load_state_dict(state,strict=True)
        if gate is not None:
            with torch.no_grad():
                model.gates.fill_(gate)
        row=measure(model,validation,device)
        if row["loss"]!=expected[name]["loss"] or row["accuracy"]!=expected[name]["accuracy"]:
            raise AssertionError(f"Budget observation changed {name} predictions")
        rows[name]=row
        print(f"{args.dataset} {args.seed} {name} actual bits={row['mean_main_activation_effective_bits']:.6f}",flush=True)
        del model
    for name,digest in source_hashes.items():
        assert sha(name)==digest
    write_json(directory/"result.json",dict(dataset=args.dataset,seed=args.seed,device=str(device),
        gpu_name=torch.cuda.get_device_name(),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_hashes=source_hashes,source_unchanged=True,test_evaluations=0,conditions=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--dataset",choices=("MNIST","CIFAR10"),required=True)
    parser.add_argument("--seed",type=int,required=True)
    run(parser.parse_args())

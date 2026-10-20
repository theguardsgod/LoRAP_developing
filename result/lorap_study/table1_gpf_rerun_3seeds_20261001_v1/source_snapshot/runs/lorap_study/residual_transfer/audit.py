"""GPU equivalence audit for INT8 edge gathering versus the fit reference."""

import argparse
import json
import os
from pathlib import Path

import torch

from ..full_run import require_cuda
from ..models import state_digest
from ..run import write_json
from ..residual_channel.experiment import sha
from .config import ROOT,DATASETS,MODELS,SEEDS,MODES,configuration
from .models import Operator,build
from .channel import GatedModel
from .run import load_data,evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    device=require_cuda()
    root=args.root.resolve()
    revised=Operator.codes
    def reference(self,codes,scale):
        if self.weights is not None:
            return self(codes.to(scale.dtype)*scale[None])
        return revised(self,codes,scale)
    rows=[]
    for dataset in DATASETS:
        for architecture in MODELS:
            c=configuration(dataset,architecture)
            data,_=load_data(c,device)
            for seed in SEEDS:
                directory=root/dataset/architecture/f'seed_{seed}'
                previous=json.loads((directory/'fit_result.json').read_text())
                path=directory/'gate/best.pt'
                codepath=directory/'calibration.pt'
                hashes={str(p):sha(p) for p in (path,codepath)}
                codes=torch.load(codepath,map_location=device,weights_only=False)
                model=GatedModel(build(c,data,seed).to(device),c,codes)
                model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model'])
                before=state_digest(model,'all')
                for mode in MODES:
                    model.mode=mode
                    try:
                        with torch.no_grad():
                            Operator.codes=reference
                            old=model(data)
                            Operator.codes=revised
                            new=model(data)
                            assert torch.equal(old.contiguous().view(torch.uint8),new.contiguous().view(torch.uint8)),(dataset,architecture,seed,mode)
                        val=evaluate(model,data,'val')
                        expected=previous['validation'][mode]
                        assert all(val[k]==expected[k] for k in val),(dataset,architecture,seed,mode,val,expected)
                        rows.append(dict(dataset=dataset,architecture=architecture,seed=seed,mode=mode,
                            logits_bitwise_equal=True,validation_metrics_exact=True))
                    finally:
                        Operator.codes=revised
                assert state_digest(model,'all')==before
                for p,digest in hashes.items():assert sha(p)==digest
                print(dataset,architecture,seed,'INT8 gather equivalence verified',flush=True)
    write_json(root/'transport_audit.json',dict(device=str(device),gpu_name=torch.cuda.get_device_name(),
        conditions=rows,test_metric_evaluations=0,model_updates=0,
        note='Formal v2 training gathers INT8 edge codes before decoding. Compare against algebraically equivalent node decoding; all complete-graph logits match exactly.'))


if __name__=='__main__':
    main()

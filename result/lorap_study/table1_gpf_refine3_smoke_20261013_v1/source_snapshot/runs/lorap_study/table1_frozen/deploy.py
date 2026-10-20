"""Standalone state artifacts without labels, training errors, or a teacher API."""
from types import SimpleNamespace
import torch
from torch import nn
from torch_geometric.data import Data
from .backends import build,load_state,predict
from .adapters import FrozenModel
from .common import save


class Metadata:
    def __init__(self,c,features,classes,nodes=None):
        self.features=features;self.classes=classes
        self.data=Data(num_nodes=nodes) if c['node'] else None
        self.c=c;self.train=self;self.num_classes=classes
    def __getitem__(self,index):
        if self.c['regression']:return Data(x=torch.zeros(1,1,dtype=torch.long))
        return Data(x=torch.zeros(1,self.features-2),pos=torch.zeros(1,2))


class InferenceOnly(nn.Module):
    def __init__(self,body,c):
        super().__init__();self.body=body.requires_grad_(False).eval();self.c=c
    def forward(self,data):
        if hasattr(self.body,'instrument'):
            return self.body(data,teacher=False)[0]
        return predict(self.body,self.c,data)[0]


def export(path,model,c,b,seed,method,hp=None):
    stats=[]
    if method!='none' and model.bank is not None:
        for i in range(len(model.instrument.layers)):
            stats.append({k:getattr(model.bank,f'{k}_{i}').detach().cpu() for k in
                          ('feature_mean','feature_std','context_mean','context_std','output_scale','rms_floor')})
    # gpf and weight LoRA require no aggregation statistics; error loss scales
    # and the observed training error vectors are intentionally absent.
    save(path,dict(format='table1_quantized_inference_v2',cell_config=c,seed=seed,method=method,
                   hyperparameters=hp,metadata=dict(features=b.features,classes=b.classes,
                   nodes=b.data.num_nodes if c['node'] else None),normalization=stats,
                   model=model.state_dict(),teacher_available=False,labels_in_artifact=False))


def load(path,device):
    artifact=torch.load(path,map_location='cpu',weights_only=False)
    if artifact['format']!='table1_quantized_inference_v2':raise ValueError('Unrecognized artifact')
    c=artifact['cell_config'];b=Metadata(c,**artifact['metadata']);seed=artifact['seed']
    backbone=build(c,b,seed)
    if artifact['method']=='none':model=backbone
    else:
        stats=[]
        for row in artifact['normalization']:
            # Constructor placeholders are overwritten by the learned state.
            d=len(row['feature_mean']);stats.append(dict(row,basis=torch.eye(d),error_scale=torch.ones(())))
        model=FrozenModel(backbone,c,stats,artifact['hyperparameters'],artifact['method'],seed)
    load_state(model,artifact['model']);return InferenceOnly(model.to(device),c).to(device).eval()

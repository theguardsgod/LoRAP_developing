"""The prediction API never receives an unquantized message or an error target.

The optional teacher callback belongs to the training/diagnostic harness. It is
absent from ordinary prediction, including validation checkpoint selection.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..residual_transfer.channel import metadata, quantize
from ..residual_transfer.models import forward


@torch.no_grad()
def context_from_quantized(q, edges, scale, bits):
    """Actual incoming edges; endpoint occupancy is NOT true clipping rate."""
    flat=q.detach().flatten(1)
    scale=scale.detach().expand_as(flat).clamp_min(1e-12)
    bits=bits.detach().expand_as(flat)
    qmax=2**(bits-1)-1
    codes=(flat/scale).round()
    # The b=1 constant codebook has no useful endpoint/clipping indicator.
    endpoint=((codes.abs()==qmax)&(qmax>0)).float().mean(1)
    source_stats=torch.stack((scale.log().mean(1),bits.mean(1)/8,endpoint),1)
    src,dst=edges
    degree=flat.new_zeros(len(flat)).index_add_(0,dst,flat.new_ones(len(dst)))
    means=flat.new_zeros(len(flat),3).index_add_(0,dst,source_stats[src])/degree.clamp_min(1)[:,None]
    return torch.cat((degree.log1p()[:,None],means),1).detach()


class Predictor(nn.Module):
    def __init__(self, stats, c, kind):
        super().__init__()
        self.kind,self.context_clip=kind,c['context_clip']
        for name,value in stats.items():
            self.register_buffer(name,value.detach().clone())
        d=self.feature_mean.numel()
        p,r=c['num_prompts'],c['rank']
        target=p*d+p+p*r+r*d+4*p+5
        self.target_parameters=target
        if kind=='mlp':
            estimate=(target-d)/(2*d+5)
            self.hidden_width=min({max(1,int(estimate)),max(1,int(estimate)+1)},
                key=lambda h:(abs(h*(2*d+5)+d-target),h))
            self.net=nn.Sequential(nn.Linear(d+4,self.hidden_width),nn.ReLU(),nn.Linear(self.hidden_width,d))
            nn.init.zeros_(self.net[2].weight);nn.init.zeros_(self.net[2].bias)
        else:
            self.a=nn.Linear(d,p)
            self.condition=nn.Linear(4,p,bias=False)
            self.gate=nn.Linear(4,1)
            self.left=nn.Parameter(torch.empty(p,r))
            self.right=nn.Parameter(torch.zeros(r,d))
            nn.init.xavier_uniform_(self.left)
            nn.init.zeros_(self.condition.weight)
            nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        self.parameter_count=sum(p.numel() for p in self.parameters())
        self.parameter_gap=abs(self.parameter_count-target)/target
        if self.parameter_gap>.05:
            raise ValueError('MLP parameter mismatch exceeds 5%')

    def forward(self, aggregate, context):
        h=(aggregate-self.feature_mean)/self.feature_std
        c=((context.detach()-self.context_mean)/self.context_std).clamp(-self.context_clip,self.context_clip)
        if self.kind=='mlp':
            out=self.net(torch.cat((h,c),1))
        else:
            weights=(self.a(h)+self.condition(c)).softmax(1)
            out=2*torch.sigmoid(self.gate(c))*(weights@(self.left@self.right))
        return out*self.output_scale


@torch.no_grad()
def local_target(x,q,operator):
    """Teacher-only same-input aggregation difference; fixed GAT attention."""
    return (operator.detached()(x.detach().double())-
            operator.detached()(q.detach().double())).flatten(1)


class PredictionCallback:
    """Deliberately ignores the legacy callback's prequantized argument."""
    def __init__(self, owner):
        self.owner=owner

    def quantize(self,i,x,quantizer):
        return quantize(x,quantizer)

    def __call__(self,i,_unused_prequantized,q,operator,quantizer):
        out,_=self.owner.apply_quantized(i,q,operator,*metadata(quantizer))
        return out


class TeacherCallback(PredictionCallback):
    def __init__(self,owner,mask,loss_scales,observer):
        super().__init__(owner)
        self.mask,self.loss_scales,self.observer=mask,loss_scales,observer
        self.losses=[]

    def __call__(self,i,x,q,operator,quantizer):
        out,delta=self.owner.apply_quantized(i,q,operator,*metadata(quantizer))
        target=local_target(x,q,operator)
        if self.mask is not None:
            self.losses.append((delta[self.mask]-target[self.mask].to(delta)).square().mean()/self.loss_scales[i])
        if self.observer is not None:
            self.observer(i,target,delta.detach())
        return out


class QuantOnlyModel(nn.Module):
    def __init__(self,backbone,backbone_config,c,stats,seed,kind):
        super().__init__()
        self.backbone,self.backbone_config=backbone,backbone_config
        self.backbone.requires_grad_(False)
        # Task/error share every initial tensor; neither perturbs backbone RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+915000)
            self.predictors=nn.ModuleList([Predictor(s,c,kind) for s in stats])
        self.eval()

    def apply_quantized(self,i,q,operator,scale,bits):
        aggregate=operator(q)
        context=context_from_quantized(q,operator.edges,scale,bits)
        delta=self.predictors[i](aggregate.flatten(1),context)
        return aggregate+delta.view_as(aggregate),delta

    def forward(self,data):
        if self.backbone.training:
            raise RuntimeError('Frozen backbone must remain in eval mode')
        return forward(self.backbone,self.backbone_config,data,PredictionCallback(self))

    def supervised_forward(self,data,mask,loss_scales):
        if mask is None or loss_scales is None:
            raise ValueError('Training mask and teacher loss scales are required')
        callback=TeacherCallback(self,mask,loss_scales,None)
        logits=forward(self.backbone,self.backbone_config,data,callback)
        return logits,torch.stack(callback.losses).mean()

    @torch.no_grad()
    def diagnose(self,data,observer):
        return forward(self.backbone,self.backbone_config,data,TeacherCallback(self,None,None,observer))


@torch.no_grad()
def calibrate(backbone,bc,c,data):
    """Inference normalization uses ONLY quantized tensors at training nodes.

    Error normalization is kept separately, never in a predictor state_dict.
    """
    stats,loss_scales,records=[],[],[]
    ids=data.train_mask.nonzero().flatten()
    class Capture:
        def quantize(self,i,x,quantizer): return quantize(x,quantizer)
        def __call__(self,i,x,q,op,quantizer):
            aggregate=op(q)
            h=aggregate.flatten(1)[ids]
            ctx=context_from_quantized(q,op.edges,*metadata(quantizer))[ids]
            floor=c['normalization_floor']
            rms=h.square().mean().sqrt().clamp_min(floor)
            stats.append(dict(feature_mean=h.mean(0),feature_std=h.std(0,unbiased=False).clamp_min(rms*.1+floor),
                context_mean=ctx.mean(0),context_std=ctx.std(0,unbiased=False).clamp_min(floor),
                output_scale=rms))
            target=local_target(x,q,op)[ids]
            loss_scales.append(target.square().mean().float().clamp_min(floor**2))
            records.append(dict(layer=2 if bc['architecture']=='GIN' else i+1,
                width=h.size(1),training_node_ids=ids.cpu().tolist(),
                error_normalization=float(loss_scales[-1]),
                output_scale=float(stats[-1]['output_scale'])))
            return aggregate
    forward(backbone,bc,data,Capture())
    return stats,torch.stack(loss_scales),records

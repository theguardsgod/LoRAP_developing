"""Ten frozen-backbone controls; all prediction branches use inference tensors."""
import itertools,math
import torch
from torch import nn
from torch.nn import functional as F
from prompt.GPF import GPF_plus
from .instrument import Instrument
from .backends import predict


def count(module):return sum(p.numel() for p in module.parameters())


def mlp_widths(d,n,target):
    cost=2*d+5
    estimate=(target/n-d)/cost
    options=range(max(1,int(estimate)-2),max(2,int(estimate)+4))
    best=min(itertools.combinations_with_replacement(options,n),
             key=lambda widths:(abs(sum(h*cost+d for h in widths)-target),widths))
    actual=sum(h*cost+d for h in best)
    if abs(actual-target)/target>.05:raise ValueError((d,n,target,best,actual))
    return best,actual


class Factors(nn.Module):
    def __init__(self,p,r,d,kind,basis=None):
        super().__init__()
        self.left=nn.Parameter(torch.empty(p,r));self.right=nn.Parameter(torch.empty(r,d))
        if kind in ('conditioned','spectral'):
            nn.init.zeros_(self.left)
            with torch.no_grad():
                value=basis[:r].float() if kind=='spectral' else torch.linalg.qr(torch.randn(d,r),mode='reduced').Q.T
                self.right.copy_(value)
        else:
            nn.init.xavier_uniform_(self.left)
            if kind in ('lorap','gpf_lorap'):nn.init.xavier_uniform_(self.right)
            else:nn.init.zeros_(self.right)


class PromptBank(nn.Module):
    def __init__(self,stats,hp,kind):
        super().__init__();self.kind=kind;self.stats_keys=[]
        self.scores=nn.ModuleDict();self.conditions=nn.ModuleDict();self.gates=nn.ModuleDict()
        self.factors=nn.ModuleList();self.mlps=nn.ModuleDict();self.outer_gates=nn.ModuleList()
        self.parameter_matches=[]
        p,r=hp['num_prompts'],hp['rank']
        groups={}
        for i,s in enumerate(stats):
            d=len(s['feature_mean']);groups.setdefault(d,[]).append(i)
            for key in ('feature_mean','feature_std','context_mean','context_std','output_scale','rms_floor'):
                self.register_buffer(f'{key}_{i}',s[key].float().clone())
            if kind=='gate_only':
                # Extra zero gates must not perturb the task/error predictor's
                # paired random initialization stream.
                with torch.random.fork_rng(devices=[]):gate=nn.Linear(4,1)
                nn.init.zeros_(gate.weight);nn.init.zeros_(gate.bias);self.outer_gates.append(gate)
        for d,ids in groups.items():
            group=str(d);n=len(ids)
            # Score, condition and gate shared across equal-width layers;
            # only useful per-layer bases contribute additional parameters.
            target=p*d+p+4*p+5+n*(p*r+r*d)
            if kind=='mlp':
                widths,actual=mlp_widths(d,n,target)
                self.parameter_matches.append(dict(width=d,layers=ids,target=target,actual=actual,gap=abs(actual-target)/target,hidden=list(widths)))
                for i,h in zip(ids,widths):
                    net=nn.Sequential(nn.Linear(d+4,h),nn.ReLU(),nn.Linear(h,d))
                    nn.init.zeros_(net[2].weight);nn.init.zeros_(net[2].bias);self.mlps[str(i)]=net
            else:
                self.scores[group]=nn.Linear(d,p)
                if kind not in ('lorap','gpf_lorap'):
                    self.conditions[group]=nn.Linear(4,p,bias=False);nn.init.zeros_(self.conditions[group].weight)
                    self.gates[group]=nn.Linear(4,1);nn.init.zeros_(self.gates[group].weight);nn.init.zeros_(self.gates[group].bias)
        if kind!='mlp':
            for s in stats:
                self.factors.append(Factors(p,r,len(s['feature_mean']),kind,s.get('basis')))

    def z(self,i,c):
        return ((c.detach()-getattr(self,f'context_mean_{i}'))/getattr(self,f'context_std_{i}')).clamp(-5,5)

    def forward(self,i,h,c):
        kind=self.kind;z=self.z(i,c);key=str(h.size(1))
        if kind in ('lorap','gpf_lorap'):
            weights=self.scores[key](h).softmax(1)
            return weights@self.factors[i].left@self.factors[i].right
        if kind in ('conditioned','spectral'):
            rms=h.square().mean(1,keepdim=True).clamp_min(1e-12).sqrt()
            feature=h/rms;amplitude=rms.detach().clamp_min(getattr(self,f'rms_floor_{i}'))
        else:
            feature=(h-getattr(self,f'feature_mean_{i}'))/getattr(self,f'feature_std_{i}')
            amplitude=getattr(self,f'output_scale_{i}')
        if kind=='mlp':return self.mlps[str(i)](torch.cat((feature,z),1))*amplitude
        weights=(self.scores[key](feature)+self.conditions[key](z)).softmax(1)
        result=(weights@self.factors[i].left@self.factors[i].right)*amplitude*2*torch.sigmoid(self.gates[key](z))
        if kind=='gate_only':result=result*(2*torch.sigmoid(self.outer_gates[i](z)))
        return result

    def freeze_for_gates(self):
        self.requires_grad_(False)
        self.outer_gates.requires_grad_(True)


class WeightFactors(nn.Module):
    def __init__(self,inputs,outputs,rank):
        super().__init__()
        self.left=nn.Parameter(torch.empty(rank,inputs));self.right=nn.Parameter(torch.zeros(outputs,rank))
        nn.init.kaiming_uniform_(self.left,a=math.sqrt(5))
    def forward(self,x):return F.linear(F.linear(x,self.left),self.right)


class FrozenModel(nn.Module):
    def __init__(self,backbone,c,stats,hp,kind,seed):
        super().__init__();self.backbone=backbone.requires_grad_(False).eval();self.c=c;self.kind=kind
        self.bank=None;self.input_prompt=None;self.lora=nn.ModuleDict();self.lora_targets=[]
        self.losses=[];self.deltas={};self.observer=None;self.mask=None;self.handles=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+915000)
            if kind not in ('gpf','weight_lora'):self.bank=PromptBank(stats,hp,kind)
            if kind in ('gpf','gpf_lorap'):self._input_prompt(hp)
            if kind=='weight_lora':self._weight_lora(hp['rank'])
        self.instrument=Instrument(backbone,c,self._aggregation) if kind not in ('gpf','weight_lora') else None
        self.loss_scales=[s['error_scale'] for s in stats]
        self.eval()

    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self

    def _input_prompt(self,hp):
        if self.c['node']:
            module=self.backbone.conv1 if hasattr(self.backbone,'conv1') else self.backbone.convs[0]
            if hasattr(module,'lin'):d=module.lin.in_features
            elif hasattr(module,'in_channels'):d=module.in_channels
            else:d=module.nn[0].in_features
            post=False
        else:
            module=getattr(self.backbone,'embeding',None)
            if module is None:module=self.backbone.embedding
            post=isinstance(module,nn.Embedding);d=module.embedding_dim if post else module.in_features
        self.input_prompt=GPF_plus(d,hp['num_prompts'])
        if post:
            self.handles.append(module.register_forward_hook(lambda m,a,o:self.input_prompt.add(o)))
        else:
            self.handles.append(module.register_forward_pre_hook(lambda m,a:(self.input_prompt.add(a[0]),*a[1:])))

    def _weight_lora(self,rank):
        from .vendor.linear_quantized import LinearQuantized
        for name,module in list(self.backbone.named_modules()):
            native=hasattr(module,'fea_quant') and hasattr(module,'weight_quant')
            dqlinear=isinstance(module,LinearQuantized)
            dqconv=hasattr(module,'layer_quantizers') and hasattr(module,'weight')
            floating=isinstance(module,nn.Linear) and not native and not dqlinear
            if not (native or dqlinear or dqconv or floating):continue
            if native or dqlinear or floating:inputs,outputs=module.in_features,module.out_features
            else:inputs,outputs=module.weight.shape
            key=str(len(self.lora));factors=WeightFactors(inputs,outputs,rank);self.lora[key]=factors
            self.lora_targets.append(name);state={}
            if native:
                def capture(m,args,out,state=state):state['q']=out[0] if isinstance(out,tuple) else out
                self.handles.append(module.fea_quant.register_forward_hook(capture))
                def add(m,args,out,key=key,state=state):
                    delta=self.lora[key](state['q'])
                    return (out[0]+delta,*out[1:]) if isinstance(out,tuple) else out+delta
                self.handles.append(module.register_forward_hook(add))
            elif floating:
                self.handles.append(module.register_forward_hook(lambda m,args,out,key=key:out+self.lora[key](args[0])))
            else:
                quantizers=module.layer_quant if dqlinear else module.layer_quantizers
                suffix='_low' if 'inputs_low' in quantizers else ''
                def capture(m,args,out,state=state):state['q']=out
                self.handles.append(quantizers['inputs'+suffix].register_forward_hook(capture))
                def add(m,args,key=key,state=state):return (args[0]+self.lora[key](state['q']),)
                self.handles.append(quantizers['features'+suffix].register_forward_pre_hook(add))

    def _aggregation(self,i,aggregate,context,target):
        delta=self.bank(i,aggregate.flatten(1),context)
        if target is not None:
            mask=self.mask if self.mask is not None else slice(None)
            self.losses.append((delta[mask]-target.flatten(1)[mask].to(delta)).square().mean()/self.loss_scales[i].to(delta))
        if self.observer is not None:self.observer(i,aggregate,context,target,delta)
        self.deltas[i]=delta.detach()
        return aggregate+delta.view_as(aggregate)

    def forward(self,data,teacher=False,observer=None):
        if self.backbone.training:raise RuntimeError('Backbone must stay frozen in eval')
        self.losses=[];self.deltas={};self.observer=observer
        self.mask=data.train_mask if self.c['node'] and teacher else None
        if self.instrument:self.instrument.reset(teacher)
        out,budget=predict(self.backbone,self.c,data)
        self.observer=None;self.mask=None
        return out,budget

    def close(self):
        if self.instrument:self.instrument.remove()
        for handle in self.handles:handle.remove()


@torch.no_grad()
def calibrate(backbone,c,b,device,seed,max_nodes=4096,max_graphs=128):
    """One shared training-only calibration, with a separate teacher artifact."""
    samples={};generators={}
    def capture(i,h,context,target):
        mask=batch.train_mask if c['node'] else slice(None)
        h=h.flatten(1)[mask];ctx=context[mask];target=target.flatten(1)[mask]
        generator=generators.setdefault(i,torch.Generator(device=device).manual_seed(seed+600000+i))
        keys=torch.rand(len(h),device=device,generator=generator)
        old=samples.get(i)
        values=dict(features=h.detach(),context=ctx.detach(),target=target.detach())
        if old:
            keys=torch.cat((old['keys'],keys));values={k:torch.cat((old[k],v)) for k,v in values.items()}
        select=keys.topk(min(len(keys),max_nodes)).indices
        samples[i]={k:v[select] for k,v in values.items()};samples[i]['keys']=keys[select]
    # Keep the returned original aggregate independent from the sampled tensor.
    def callback(i,h,context,target):
        capture(i,h,context,target);return h
    instrument=Instrument(backbone,c,callback);backbone.eval()
    try:
        for batch in b.batches('train',seed,0,device,limit=max_graphs):
            instrument.reset(True);predict(backbone,c,batch)
    finally:instrument.remove()
    stats=[]
    for i in sorted(samples):
        sample=samples[i];h=sample['features'];ctx=sample['context'];error=sample['target']
        rms=h.square().mean().sqrt().clamp_min(1e-4)
        cov=error.T@error
        values,vectors=torch.linalg.eigh(cov)
        basis=vectors.flip(1).T
        sign=basis[torch.arange(len(basis),device=device),basis.abs().argmax(1)].sign()
        basis=basis*torch.where(sign==0,torch.ones_like(sign),sign)[:,None]
        stats.append(dict(feature_mean=h.mean(0),feature_std=h.std(0,unbiased=False).clamp_min(rms*.1+1e-4),
            context_mean=ctx.mean(0),context_std=ctx.std(0,unbiased=False).clamp_min(1e-4),
            output_scale=rms,rms_floor=h.square().mean(1).sqrt().mean().mul(.1).clamp_min(1e-4),
            basis=basis.float(),error_scale=error.square().mean().float().clamp_min(1e-8),
            singular_values=values.flip(0).clamp_min(0).sqrt(),sampled_nodes=len(h),split='train'))
    return stats

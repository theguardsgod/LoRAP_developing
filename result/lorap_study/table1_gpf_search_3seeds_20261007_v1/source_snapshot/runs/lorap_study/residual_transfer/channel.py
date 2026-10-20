"""Head-wise residual codes and exactly matched additional message-bit budgets."""

import math
import torch
from torch import nn

from ..residual_channel.core import canonical_basis
from .models import forward,message_quantizers


def metadata(quantizer):
    if hasattr(quantizer,'gama'):
        return quantizer.gama.detach().abs(),quantizer.bit.detach().abs().round()
    return quantizer.alpha.detach().abs().T,quantizer.bit.detach().abs().round().T


@torch.no_grad()
def mixed_quantize(x,quantizer,extra):
    scale,bits=metadata(quantizer)
    new_bits=bits+extra.to(x.dtype)[None]
    # Form the ratio first so unchanged channels multiply by exactly one.
    new_scale=scale*((2**(bits-1)-1)/(2**(new_bits-1)-1))
    if hasattr(quantizer,'gama'):
        return quantizer.quant_fea_func.apply(x,new_scale,new_bits)
    return quantizer.quant_weight_func.apply(x.T,new_scale.T,new_bits.T).T


def quantize(x,quantizer):
    return quantizer(x) if hasattr(quantizer,'gama') else quantizer(x.T).T


class BaseCapture:
    def __init__(self,observer):
        self.observer=observer

    def quantize(self,i,x,q):
        return quantize(x,q)

    def __call__(self,i,x,q,operator,quantizer):
        self.observer(i,x.detach(),q.detach(),operator.detached(),quantizer)
        return operator(q)


@torch.no_grad()
def calibrate(model,c,data):
    model.eval()
    info,records=[],[]
    ids=data.train_mask.nonzero().flatten()
    if len(ids)>c['calibration_max_nodes']:
        generator=torch.Generator(device=ids.device).manual_seed(914041)
        ids=ids[torch.randperm(len(ids),generator=generator,device=ids.device)[:c['calibration_max_nodes']]].sort().values

    def capture(i,x,q,operator,quantizer):
        residual=x-q
        error=operator(residual)
        bases,spectra=[],[]
        for head in range(x.size(1)):
            basis,spectrum=canonical_basis(error[ids,head],rank=c['rank'])
            bases.append(basis)
            spectra.append(spectrum)
        basis=torch.stack(bases)
        values=torch.einsum('nhd,hdr->nhr',residual,basis)
        scale=values[ids].abs().amax(0).div(127).clamp_min(1e-12)
        width=x.size(1)*x.size(2)
        budget=x.size(1)*c['rank']*c['side_bits']
        levels=math.ceil(budget/width)
        old_bits=metadata(quantizer)[1]
        assert float(old_bits.max())+levels<=c['higher_max_bits']
        costs=[]
        for extra in range(levels+1):
            high=mixed_quantize(x.flatten(1),quantizer,torch.full((width,),extra,device=x.device)).view_as(x)
            costs.append(operator(x-high)[ids].double().square().sum(0).flatten())
        costs=torch.stack(costs)
        increments=torch.zeros(width,dtype=torch.long,device=x.device)
        columns=torch.arange(width,device=x.device)
        for _ in range(budget):
            next_level=(increments+1).clamp_max(levels)
            gains=costs[increments,columns]-costs[next_level,columns]
            gains[increments==levels]=-torch.inf
            index=gains.argmax()
            increments[index]+=1
        assert int(increments.sum())==budget
        info.append(dict(basis=basis,scale=scale,increments=increments))
        records.append(dict(layer=i+1,heads=x.size(1),width_per_head=x.size(2),rank_per_head=c['rank'],
            basis_training_node_ids=ids.cpu().tolist(),scale_training_node_ids=ids.cpu().tolist(),
            eigenvalues=[v.cpu().tolist() for v in spectra],extra_message_bits_per_node=budget,
            increment_limit=levels,increments=increments.cpu().tolist(),
            head_orthogonality_error=float((basis.transpose(1,2)@basis-torch.eye(c['rank'],device=x.device)).abs().max())))

    forward(model,c,data,BaseCapture(capture))
    return info,records


class Code(nn.Module):
    def __init__(self,info):
        super().__init__()
        for name,value in info.items():
            self.register_buffer(name,value.detach().clone())

    @torch.no_grad()
    def correction(self,residual,operator):
        values=torch.einsum('nhd,hdr->nhr',residual,self.basis)
        normalized=values/self.scale[None]
        codes=(normalized.abs().add(.5).floor()*normalized.sign()).clamp(-127,127).to(torch.int8)
        transported=operator.codes(codes,self.scale)
        delta=torch.einsum('nhr,hdr->nhd',transported,self.basis)
        return delta,dict(codes=codes,values=values,decoded=codes.to(values.dtype)*self.scale[None])


class GatedModel(nn.Module):
    def __init__(self,backbone,c,calibration):
        super().__init__()
        self.backbone,self.config=backbone,c
        self.codes=nn.ModuleList([Code(row) for row in calibration])
        self.gates=nn.Parameter(torch.zeros(len(calibration),device=next(backbone.parameters()).device))
        self.backbone.requires_grad_(False)
        self.mode='gate'
        self.observer=None
        self.bypass_values=False
        self.eval()

    def quantize(self,i,x,quantizer):
        if self.mode=='higher':
            return mixed_quantize(x,quantizer,self.codes[i].increments)
        return quantize(x,quantizer)

    def apply_layer(self,i,x,q,operator,quantizer):
        if self.bypass_values:
            # For GAT, attention has already been computed from quantized values.
            q=x
        out=operator(q)
        delta=details=None
        if self.mode in ('gate','unit'):
            delta,details=self.codes[i].correction(x.detach()-q.detach(),operator.detached())
            out=out+(self.gates[i] if self.mode=='gate' else 1.)*delta
        if self.observer is not None:
            with torch.no_grad():
                self.observer(i,x.detach(),q.detach(),operator.detached(),
                    None if delta is None else (self.gates[i].detach() if self.mode=='gate' else 1.)*delta,
                    self.codes[i],details,self.gates[i].detach() if self.mode=='gate' else 1.)
        return out

    def forward(self,data):
        # A callable adapter avoids overriding nn.Module.__call__ and hooks.
        owner=self
        class Callback:
            def quantize(self,i,x,q): return owner.quantize(i,x,q)
            def __call__(self,*args): return owner.apply_layer(*args)
        return forward(self.backbone,self.config,data,Callback())

    @torch.no_grad()
    def project(self):
        self.gates.clamp_(0,1)


class Diagnostics:
    def __init__(self,mask):
        self.mask=mask
        self.rows=[]

    @torch.no_grad()
    def __call__(self,i,x,q,operator,delta,code,details,strength):
        error=operator((x-q).double())[self.mask]
        delta=torch.zeros_like(error) if delta is None else delta[self.mask].double()
        before,after=error.square().sum(),(error-delta).square().sum()
        dot=(error*delta).sum()
        norm=delta.square().sum()
        row=dict(layer=i+1,nodes=int(self.mask.sum()),before_sq=float(before),after_sq=float(after),
            correction_sq=float(norm),dot=float(dot),mse_reduction=None if float(before)==0 else float(1-after/before),
            cosine=None if float(norm)==0 else float(dot/(norm*before).sqrt().clamp_min(1e-30)))
        assert abs(float(after-before+2*dot-norm))/max(float(before),1e-30)<1e-6
        singular=torch.linalg.svdvals(error.flatten(1))
        energy=singular.square()
        row.update(squared_singular_values=energy.cpu().tolist(),
                   cumulative_energy=(energy.cumsum(0)/energy.sum().clamp_min(1e-30)).cpu().tolist())
        if details is not None:
            basis=code.basis.double()
            projection=torch.einsum('nhd,hdr->nhr',error,basis)
            truncation=error-torch.einsum('nhr,hdr->nhd',projection,basis)
            eta=details['decoded'].double()-torch.einsum('nhd,hdr->nhr',(x-q).double(),basis)
            noise=operator(eta)[self.mask]
            strength=float(strength)
            expected=truncation.square().sum()+((1-strength)*projection-strength*noise).square().sum()
            row['orthogonal_identity_relative_gap']=float(abs(after-expected)/before.clamp_min(1e-30))
            assert row['orthogonal_identity_relative_gap']<2e-5
            values=details['values'][self.mask]
            row['saturation_fraction']=float((values.abs()>127*code.scale[None]).float().mean())
        self.rows.append(row)

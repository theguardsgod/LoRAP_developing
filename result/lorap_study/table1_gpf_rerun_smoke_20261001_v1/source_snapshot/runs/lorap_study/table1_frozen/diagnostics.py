"""Validation-only local error statistics and fixed-input capacity controls."""
import torch
from .instrument import Instrument
from .backends import predict


class Collector:
    def __init__(self,device,seed,max_nodes=4096):
        self.device,self.seed,self.max_nodes=device,seed,max_nodes
        self.rows={};self.generators={}

    @torch.no_grad()
    def add(self,i,h,context,target,delta,mask):
        if target is None:raise ValueError('Diagnostic requires a local teacher target')
        target=target.flatten(1)[mask].double();delta=delta.flatten(1)[mask].double()
        after=target-delta
        state=self.rows.setdefault(i,dict(sum_before=target.new_zeros(()),sum_after=target.new_zeros(()),
            dot=target.new_zeros(()),delta_norm=target.new_zeros(()),elements=0,nodes=0))
        state['sum_before']+=target.square().sum();state['sum_after']+=after.square().sum()
        state['dot']+=(target*delta).sum();state['delta_norm']+=delta.square().sum()
        state['elements']+=target.numel();state['nodes']+=len(target)
        generator=self.generators.setdefault(i,torch.Generator(device=self.device).manual_seed(self.seed+730000+i))
        keys=torch.rand(len(target),generator=generator,device=self.device)
        if 'keys' in state:
            keys=torch.cat((state['keys'],keys));target=torch.cat((state['before'],target));after=torch.cat((state['after'],after))
        selected=keys.topk(min(len(keys),self.max_nodes)).indices
        state.update(keys=keys[selected],before=target[selected],after=after[selected])

    def finish(self):
        result=[]
        for i,s in sorted(self.rows.items()):
            before=float(s['sum_before']);after=float(s['sum_after'])
            row=dict(layer=i,nodes=s['nodes'],elements=s['elements'],mse_before=before/s['elements'],mse_after=after/s['elements'],
                     reduction=None if before==0 else 1-after/before,
                     cosine=None if before==0 or float(s['delta_norm'])==0 else float(s['dot']/(s['sum_before']*s['delta_norm']).sqrt()),
                     spectrum_sample_nodes=len(s['before']))
            for label in ('before','after'):
                singular=torch.linalg.svdvals(s[label]);energy=singular.square();total=energy.sum()
                row[label+'_singular_values']=singular.cpu().tolist()
                row[label+'_cumulative_energy']=(energy.cumsum(0)/total if total>0 else torch.zeros_like(energy)).cpu().tolist()
            result.append(row)
        return result


@torch.no_grad()
def diagnose(model,c,b,seed,device):
    """Own-input diagnostic; all reference computation is explicitly requested."""
    collector=Collector(device,seed)
    def observe(i,h,ctx,target,delta):
        collector.add(i,h,ctx,target,delta,batch.val_mask if c['node'] else slice(None))
    for batch in b.batches('val',seed,0,device):model(batch,teacher=True,observer=observe)
    return dict(kind='own_input',scope='local pre-aggregation message quantization; fixed attention; downstream aggregate quantization retained',
                layers=collector.finish(),split='validation',max_nodes=4096)


@torch.no_grad()
def fixed_input(model,c,b,seed,device):
    """Evaluate all prompt predictors on their identical no-prompt backbone input.

    This is the causal local-error comparison. It is defined for aggregation
    predictors; input GPF/weight LoRA have different sites and are excluded.
    """
    if model.bank is None:return dict(kind='fixed_input',applicable=False,reason='No aggregation predictor')
    collector=Collector(device,seed);model.instrument.remove()
    def callback(i,h,ctx,target):
        delta=model.bank(i,h.flatten(1),ctx)
        collector.add(i,h,ctx,target,delta,batch.val_mask if c['node'] else slice(None))
        return h
    # Also remove any input GPF when evaluating the common no-prompt input.
    for handle in model.handles:handle.remove()
    model.handles=[]
    instrument=Instrument(model.backbone,c,callback)
    try:
        for batch in b.batches('val',seed,0,device):
            instrument.reset(True);predict(model.backbone,c,batch)
    finally:instrument.remove()
    return dict(kind='fixed_input',applicable=True,split='validation',layers=collector.finish(),
                scope='predictor receives identical no-prompt inputs; corrections are observed but not propagated')


def quantization_record(backbone):
    from .vendor.quantization import IntegerQuantizer
    quantizers=[]
    for name,m in backbone.named_modules():
        if isinstance(m,IntegerQuantizer):
            quantizers.append(dict(name=name,kind='affine_INT4',bits=m.num_bits,signed=m.signed,
                                   min_val=float(m.min_val) if m.min_val.numel() else None,
                                   max_val=float(m.max_val) if m.max_val.numel() else None))
        elif hasattr(m,'bit') and torch.is_tensor(m.bit) and (hasattr(m,'alpha') or hasattr(m,'gama')):
            bits=m.bit.detach().abs().round().cpu();values,counts=bits.unique(return_counts=True)
            quantizers.append(dict(name=name,kind='learned_symmetric',rounded_bit_mean=float(bits.float().mean()),
                                   parameter_entries=bits.numel(),histogram={str(int(v)):int(n) for v,n in zip(values,counts)}))
    return dict(quantizers=quantizers,statistic='rounded codebook bits by quantizer parameter entry; unused graph clusters are not traffic weighted',
                storage_note='fake quantization checkpoints contain FP32 tensors; no claim of packed INT4 artifact size',
                checkpoint_tensor_bytes=sum(t.numel()*t.element_size() for t in backbone.state_dict().values()))

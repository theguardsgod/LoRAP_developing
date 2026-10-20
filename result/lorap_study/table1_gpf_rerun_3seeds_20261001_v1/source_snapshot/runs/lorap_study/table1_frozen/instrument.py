"""Aggregation interception separates inference inputs from optional teachers.

Predictors receive exactly (quantized aggregate, detached four-value context).
The teacher lives in this harness; its tensors never enter a predictor API.
"""
import inspect
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .deterministic import stable_softmax as softmax, sum_by_index
from .vendor.quantization import get_qparams


def aggregation_layers(backbone,c):
    found=[]
    for name,m in backbone.named_modules():
        if c['framework']=='A2Q':
            if hasattr(m,'q_gat_fea') or (hasattr(m,'q_xw') and not isinstance(m.q_xw,nn.Identity)):
                found.append((name,m))
        elif hasattr(m,'mp_quantizers'):
            # Original node GIN's prompt starts after the hidden representation.
            if not (c['node'] and c['architecture']=='GIN' and name=='convs.0'):
                found.append((name,m))
    if not found:raise ValueError('No aggregation site')
    return found


@torch.no_grad()
def context_nodes(q,edges,scale,bits):
    q=q.detach().flatten(1);scale=scale.detach().clamp_min(1e-12).expand_as(q);bits=bits.detach().expand_as(q)
    bound=2**(bits-1)-1
    endpoint=(((q/scale).round().abs()==bound)&(bound>0)).float().mean(1)
    stats=torch.stack((scale.log().mean(1),bits.mean(1)/8,endpoint),1)
    dst=edges[1];degree=q.new_zeros(len(q)).index_add_(0,dst,q.new_ones(len(dst)))
    averages=sum_by_index(stats[edges[0]],dst,len(q))/degree.clamp_min(1)[:,None]
    return torch.cat((degree.log1p()[:,None],averages),1)


@torch.no_grad()
def context_edges(q,index,n,quantizer):
    lo,hi,zp,scale=get_qparams(quantizer.max_val,quantizer.min_val,quantizer.num_bits,
                              quantizer.signed,quantizer.eps,quantizer.symmetric)
    code=(q.detach().flatten(1)/scale+zp).round()
    endpoint=((code==lo)|(code==hi)).float().mean(1)
    degree=q.new_zeros(n).index_add_(0,index,q.new_ones(len(index)))
    occupancy=sum_by_index(endpoint,index,n)/degree.clamp_min(1)
    connected=(degree>0).to(q)
    return torch.stack((degree.log1p(),connected*float(torch.tensor(scale).log()),
                        connected*(quantizer.num_bits/8),occupancy),1)


class Instrument:
    def __init__(self,backbone,c,callback):
        self.c,self.callback=c,callback
        self.teacher=False;self.enabled=True;self.records={};self.handles=[];self.originals=[]
        self.layers=aggregation_layers(backbone,c)
        self.reference_calls=0
        for i,(_,layer) in enumerate(self.layers):
            self._install(i,layer)

    def reset(self,teacher=False):
        self.teacher=teacher;self.records={}

    def replace(self,obj,name,fn):
        original=getattr(obj,name);self.originals.append((obj,name,original))
        setattr(obj,name,MethodType(fn,obj));return original

    def remove(self):
        for h in self.handles:h.remove()
        for obj,name,original in reversed(self.originals):setattr(obj,name,original)
        self.handles=[];self.originals=[]
        self.callback=None

    def _install(self,i,layer):
        state={};native_prop=layer.propagate;native_aggr=layer.aggregate
        def propagate(_layer,edge_index,*args,**kwargs):
            state['edges']=edge_index
            return native_prop(edge_index,*args,**kwargs)
        self.replace(layer,'propagate',propagate)
        if self.c['framework']=='A2Q':
            quantizer=layer.q_gat_fea if hasattr(layer,'q_gat_fea') else layer.q_xw
            transposed=hasattr(layer,'q_xw')
            if hasattr(quantizer,'uniform'):
                # Expose the index already required by the real graph quantizer.
                # No second feature-derived lookup is performed by a predictor.
                if not quantizer.uniform or quantizer.is_naive:raise ValueError('Unexpected graph quantizer mapping')
                from quantize_function.get_scale_index import get_scale_index_uniform
                def mapped_quant(m,fea,edge_index):
                    if edge_index is None:raise ValueError('Graph quantizer requires edge indices')
                    index=get_scale_index_uniform(fea=fea,deg_index=edge_index,scale=m.gama,bit=m.bit)
                    m.table1_active_index=index
                    bits=fea.size(1)*m.bit[index.unique()].sum()/(8*1024)
                    result=m.quant_fea_func.apply(fea,m.gama,m.bit,m.deg_inv_sqrt,m.cal_mode,index)
                    return result,bits
                self.replace(quantizer,'forward',mapped_quant)
            def quant_hook(m,args,out):
                q=out[0] if isinstance(out,tuple) else out
                state['q']=q.T if transposed else q
                if self.teacher:state['pre']=args[0].detach().T if transposed else args[0].detach()
                else:state.pop('pre',None)
                if transposed:
                    state['scale']=m.alpha.detach().abs().T;state['bits']=m.bit.detach().abs().round().T
                elif hasattr(m,'table1_active_index'):
                    idx=m.table1_active_index
                    state['scale']=m.gama.detach()[idx].abs();state['bits']=m.bit.detach()[idx].abs().round()
                else:
                    state['scale']=m.gama.detach().abs();state['bits']=m.bit.detach().abs().round()
            self.handles.append(quantizer.register_forward_hook(quant_hook))
            if hasattr(layer,'q_gat_edge'):
                def alpha_hook(m,args,out):
                    if self.teacher:
                        value=out[0] if isinstance(out,tuple) else out
                        state['alpha']=value.detach() if isinstance(out,tuple) else value.detach().T
                self.handles.append(layer.q_gat_edge.register_forward_hook(alpha_hook))
            native_message=layer.message;signature=inspect.signature(native_message)
            def message(_layer,*args,**kwargs):
                msg=native_message(*args,**kwargs)
                if self.teacher:
                    bound=signature.bind(*args,**kwargs).arguments
                    with torch.no_grad():
                        edges=state['edges'];values=state['pre'][edges[0]]
                        if hasattr(layer,'q_gat_edge'):
                            weights=softmax(F.leaky_relu(state['alpha'],layer.negative_slope),edges[1],num_nodes=len(state['pre']))
                            values=values.reshape(-1,layer.heads,layer.out_channels)*weights[...,None]
                        elif 'norm' in bound:values=values*bound['norm'].detach().reshape(-1,1)
                        state['reference_messages']=values
                return msg
            self.replace(layer,'message',message)
            def aggregate(_layer,inputs,index,ptr=None,dim_size=None):
                # Historical custom GIN uses (inputs,index,dim_size); PyG adds ptr.
                if hasattr(layer,'__aggr_params__'):
                    size=dim_size if dim_size is not None else ptr
                    aggregate=native_aggr(inputs,index,size)
                else:
                    size=dim_size;aggregate=native_aggr(inputs,index,ptr,dim_size)
                if not self.enabled:return aggregate
                context=context_nodes(state['q'],state['edges'],state['scale'],state['bits'])
                target=None
                if self.teacher:
                    self.reference_calls+=1
                    with torch.no_grad():
                        reference=sum_by_index(state['reference_messages'],index,len(aggregate))
                        target=reference.double()-aggregate.detach().double()
                return self.callback(i,aggregate,context,target)
            self.replace(layer,'aggregate',aggregate)
        else:
            keys=('message_low','aggregate_low') if self.c['framework']=='DQ' else ('message','aggregate')
            qm,qa=(layer.mp_quantizers[k] for k in keys)
            def message_hook(m,args,out):
                state['q']=out.detach()
                if self.teacher:state['reference_messages']=args[0].detach()
                else:state.pop('reference_messages',None)
            self.handles.append(qm.register_forward_hook(message_hook))
            def aggregate(_layer,inputs,index,dim_size):
                state['index']=index;state['n']=dim_size
                result=native_aggr(inputs,index,dim_size)
                if self.teacher:
                    self.reference_calls+=1
                    with torch.no_grad():state['reference']=native_aggr(state['reference_messages'],index,dim_size)
                return result
            self.replace(layer,'aggregate',aggregate)
            def aggregate_hook(m,args,out):
                if not self.enabled:return out
                context=context_edges(state['q'],state['index'],state['n'],qm)
                target=None
                if self.teacher:
                    # Call forward directly to avoid re-entering this hook. Observer
                    # stays frozen; downstream aggregate quantization is retained.
                    with torch.no_grad():target=m.forward(state['reference']).double()-out.detach().double()
                return self.callback(i,out,context,target)
            self.handles.append(qa.register_forward_hook(aggregate_hook))

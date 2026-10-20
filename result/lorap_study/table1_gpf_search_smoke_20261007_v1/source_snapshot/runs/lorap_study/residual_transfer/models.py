"""Repository quantizers and explicit GCN/GAT weighted value aggregation."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops,remove_self_loops,degree,softmax

from quantize_function.u_quant_func_bit_debug import QLinear,u_quant_xw,u_quant_fea,u_quant_weight
from ..models import build_model,sum_messages
from ..mechanism import NoPrompt


def loop_edges(edges,n):
    return add_self_loops(remove_self_loops(edges)[0],num_nodes=n)[0]


class Operator:
    """Fixed node operator for each head, including the GIN self contribution."""
    def __init__(self,edges,weights=None,eps=None):
        self.edges,self.weights,self.eps=edges,weights,eps

    def __call__(self,x):
        # x: N x heads x channels. Weighted sums are FP32/FP64, not INT32.
        if self.weights is None:
            return sum_messages(x,self.edges)+(1+self.eps)*x
        w=self.weights.to(x.dtype)
        out=x.new_zeros(x.shape)
        return out.index_add(0,self.edges[1],x[self.edges[0]]*w[...,None])

    def detached(self):
        return Operator(self.edges,None if self.weights is None else self.weights.detach(),
                        None if self.eps is None else self.eps.detach())

    def codes(self,codes,scale):
        if self.weights is not None:
            # Gather the actual INT8 edge payload before receiver-side decoding.
            edge_codes=codes[self.edges[0]]
            messages=(edge_codes.to(scale.dtype)*scale[None])*self.weights.to(scale.dtype)[...,None]
            return torch.zeros_like(codes,dtype=scale.dtype).index_add(0,self.edges[1],messages)
        if self.edges.size(1)*127>torch.iinfo(torch.int32).max:
            raise OverflowError('INT32 bound exceeded')
        out=torch.zeros_like(codes,dtype=torch.int32)
        out.index_add_(0,self.edges[1],codes[self.edges[0]].int())
        return (out.to(scale.dtype)+(1+self.eps)*codes.to(scale.dtype))*scale[None]


class GCNLayer(nn.Module):
    def __init__(self,inputs,outputs,n,first):
        super().__init__()
        self.lin=QLinear(inputs,outputs,n,4,all_positive=first,quant_fea=not first,
            para_dict=dict(alpha_init=.1,alpha_std=.2,gama_init=.01,gama_std=.01))
        self.lin.fea_quant.study_unsigned=first
        self.q_xw=u_quant_xw(1,outputs,4,alpha_init=.01,alpha_std=.01)

    def forward(self,x,edges,callback,index):
        x=self.lin(x)
        edges=loop_edges(edges,len(x))
        inv=degree(edges[1],len(x),dtype=x.dtype).pow(-.5)
        weights=(inv[edges[0]]*inv[edges[1]])[:,None]
        q=callback.quantize(index,x,self.q_xw) if callback else self.q_xw(x.T).T
        operator=Operator(edges,weights)
        out=callback(index,x[:,None],q[:,None],operator,self.q_xw) if callback else operator(q[:,None])
        return out[:,0]


class GATLayer(nn.Module):
    def __init__(self,inputs,outputs,heads,n,first,dropout):
        super().__init__()
        self.heads,self.width,self.dropout=heads,outputs,dropout
        self.lin=QLinear(inputs,heads*outputs,n,4,all_positive=not first,quant_fea=not first)
        self.lin.fea_quant.study_unsigned=not first
        self.att=nn.Parameter(torch.empty(1,heads,2*outputs))
        self.bias=nn.Parameter(torch.zeros(heads*outputs))
        self.q_gat_attn=u_quant_weight(1,heads,4)
        self.q_gat_fea=u_quant_fea(n,4)
        self.q_gat_edge=u_quant_weight(1,heads,4)
        glorot(self.att)

    def attention(self,q,edges):
        att=self.q_gat_attn(self.att.squeeze(0))[None]
        pair=torch.cat((q[edges[1]],q[edges[0]]),dim=-1)
        alpha=(pair*att).sum(-1)
        alpha=self.q_gat_edge(alpha.T).T
        alpha=softmax(F.leaky_relu(alpha,.2),edges[1],num_nodes=len(q))
        return F.dropout(alpha,self.dropout,training=self.training)

    def forward(self,x,edges,callback,index):
        x=self.lin(x)
        q=callback.quantize(index,x,self.q_gat_fea) if callback else self.q_gat_fea(x)
        x,q=x.view(-1,self.heads,self.width),q.view(-1,self.heads,self.width)
        edges=loop_edges(edges,len(x))
        operator=Operator(edges,self.attention(q,edges))
        out=callback(index,x,q,operator,self.q_gat_fea) if callback else operator(q)
        return out.flatten(1)+self.bias


class NodeNetwork(nn.Module):
    def __init__(self,c,data):
        super().__init__()
        self.architecture,self.dropout=c['architecture'],c['dropout']
        n,inputs,classes=data.num_nodes,data.num_features,int(data.y.max())+1
        if self.architecture=='GCN':
            self.layers=nn.ModuleList([GCNLayer(inputs,c['hidden'],n,True),GCNLayer(c['hidden'],classes,n,False)])
        else:
            self.layers=nn.ModuleList([GATLayer(inputs,c['hidden'],c['heads'],n,True,c['attention_dropout']),
                GATLayer(c['hidden']*c['heads'],classes,1,n,False,c['attention_dropout'])])
        for m in self.modules():
            if isinstance(m,nn.Linear):
                glorot(m.weight)

    def forward(self,data,callback=None):
        x=data.x
        if self.architecture=='GAT':
            x=F.dropout(x,self.dropout,training=self.training)
        x=self.layers[0](x,data.edge_index,callback,0)
        x=F.dropout(F.relu(x),self.dropout,training=self.training)
        return self.layers[1](x,data.edge_index,callback,1)


def build(c,data,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if c['architecture']=='GIN':
            model=build_model('CiteSeer','original',seed,data,dropout=c['dropout'],width=c['hidden'],bit=c['bit'])
            model.prompt_aggr=NoPrompt()
        else:
            model=NodeNetwork(c,data)
    for name,p in model.named_parameters():
        if name.endswith('.bit') and '.fea_quant.' not in name:
            p.requires_grad_(False)
    return model


def message_quantizers(model,c):
    if c['architecture']=='GIN':
        return [model.conv2.q_xw]
    return [m.q_xw if c['architecture']=='GCN' else m.q_gat_fea for m in model.layers]


def forward(model,c,data,callback=None):
    if data.x.device.type!='cuda':
        raise RuntimeError('Model computation requires CUDA')
    if c['architecture']!='GIN':
        return model(data,callback)
    if callback is None:
        return model(data)[0]
    x,_=model.conv1(data.x,data.edge_index,data.x.new_zeros(()))
    x=F.dropout(x,model.dropout,training=model.training)
    conv=model.conv2
    q=callback.quantize(0,x,conv.q_xw)
    operator=Operator(remove_self_loops(data.edge_index)[0],eps=conv.eps)
    out=callback(0,x[:,None],q[:,None],operator,conv.q_xw)[:,0]
    return conv.mlp(out)


def storage_proxy(model,c):
    total=next(model.parameters()).new_zeros(())
    for name,p in model.named_parameters():
        if '.fea_quant.bit' in name:
            total=total+c['hidden']*p.abs().sum()/(8*1024)
        elif c['architecture']=='GAT' and name.endswith('q_gat_fea.bit'):
            total=total+c['hidden']*c['heads']*p.abs().sum()/(8*1024)
    return total


def optimizer(model,c):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('.bit'):
            kind,lr,decay='bits',c['lr_bit'],0.
        elif '.fea_quant.gama' in name:
            kind,lr,decay='features',c['lr_feature'],0.
        elif '.weight_quant.alpha' in name:
            kind,lr,decay='weights_scale',c['lr_weight'],0.
        elif 'q_xw.alpha' in name or 'q_gat_fea.gama' in name:
            kind,lr,decay='messages',c['lr_message'],0.
        elif 'q_gat_' in name:
            kind,lr,decay='attention_scale',c['lr_attention'],0.
        else:
            kind,lr,decay='backbone',c['lr'],c['weight_decay']
        groups.setdefault(kind,dict(params=[],lr=lr,weight_decay=decay,name=kind))['params'].append(p)
    return torch.optim.Adam(list(groups.values()))


@torch.no_grad()
def project_quantizers(model,c):
    # Signed surrogate b=1 is the zero-valued quantizer (qmax=0); b=0 would
    # create qmax=-0.5 in the inherited function and is not valid zero-bit coding.
    for name,p in model.named_parameters():
        if name.endswith('.bit') and p.requires_grad:
            p.clamp_(*c['trainable_quantizer_bit_bounds'])

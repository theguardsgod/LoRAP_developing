"""Native quantization operators with explicit, valid task heads.

Architecture repairs are listed in BACKEND_NOTES.md. No historical training
entrypoint is imported or executed, and no experiment uses external telemetry.
"""
import importlib
from types import SimpleNamespace, MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .deterministic import mean_pool as global_mean_pool, stable_softmax, install as install_reductions
from torch_geometric.nn.inits import glorot
from .vendor.quantization import IntegerQuantizer
from .vendor.linear_quantized import LinearQuantized
from .vendor.baseline_quant import GINConvQuant, GCNConvQuant, GATConvQuant
from .vendor.multi_quant import GINConvMultiQuant, GCNConvMultiQuant, GATConvMultiQuant


def factories(c, positive=False):
    def q(signed=True):
        return lambda: IntegerQuantizer(4,signed=signed,use_momentum=c['use_momentum'],
                                        use_ste=c['use_ste'],percentile=c['percentile'],sample=None)
    layer={k:q(not positive if k=='inputs' else True) for k in ('inputs','weights','features','norm','attention','alpha')}
    if c['framework']=='DQ':
        mp={k+'_'+s:(nn.Identity if s=='high' else q()) for k in ('message','aggregate','update') for s in ('low','high')}
        layer={**layer,**{k+'_'+s:(nn.Identity if s=='high' else f) for k,f in layer.items() for s in ('low','high')}}
    else:
        mp={k:q() for k in ('message','aggregate','update_q')}
    return layer,mp


class ResettableSequential(nn.Sequential):
    def reset_parameters(self):
        for m in self:
            if hasattr(m,'reset_parameters'): m.reset_parameters()


class DQNetwork(nn.Module):
    def __init__(self,c,b):
        super().__init__()
        self.c=c
        # Historical graph heads used .5 independently of the node dropout key.
        self.head_dropout=c.get('head_dropout', .5)
        if not 0. <= self.head_dropout < 1.:
            raise ValueError('head_dropout must be in [0, 1)')
        dq=c['framework']=='DQ'
        cls={'GIN':GINConvMultiQuant if dq else GINConvQuant,
             'GCN':GCNConvMultiQuant if dq else GCNConvQuant,
             'GAT':GATConvMultiQuant if dq else GATConvQuant}[c['architecture']]
        width=c['width']*(c['heads'] if c['architecture']=='GAT' else 1)
        lq,mq=factories(c)
        pos,_=factories(c,positive=True)
        self.convs=nn.ModuleList(); self.norms=nn.ModuleList()
        self.embedding=None
        if not c['node']:
            if c['regression']: self.embedding=nn.Embedding(28,width)
            elif c['architecture']=='GIN' and dq: self.embedding=LinearQuantized(b.features,width,layer_quantizers=lq)
            else: self.embedding=nn.Linear(b.features,width)
        for i in range(c['layers']):
            inputs=b.features if c['node'] and i==0 else width
            outputs=b.classes if c['node'] and i==c['layers']-1 else width
            if c['architecture']=='GIN':
                if c['node']:
                    block=[LinearQuantized(inputs,outputs,layer_quantizers=pos)]
                    if i==0:block.append(nn.ReLU())
                else:
                    # The local historical QAT graph GIN quantizes message/aggregate/update;
                    # its dense MLP stays floating point. DQ also quantizes dense layers.
                    lin=lambda p: LinearQuantized(width,width,layer_quantizers=p) if dq else nn.Linear(width,width)
                    block=[lin(lq),nn.ReLU(),lin(pos),nn.ReLU(),nn.BatchNorm1d(width)]
                conv=cls(ResettableSequential(*block),train_eps=True,mp_quantizers=mq)
            elif c['architecture']=='GCN':
                conv=cls(inputs,outputs,layer_quantizers=lq,mp_quantizers=mq)
            else:
                heads=1 if (c['node'] and i==c['layers']-1) or (not c['node'] and i==c['layers']-1) else c['heads']
                conv=cls(inputs,outputs//heads,heads=heads,concat=True,
                         dropout=.1 if c['node'] else 0.,layer_quantizers=lq,mp_quantizers=mq)
            self.convs.append(conv)
            self.norms.append(nn.Identity() if c['node'] or c['architecture']=='GIN' else nn.BatchNorm1d(outputs))
        if not c['node']:
            dense=lambda a,z,p: nn.Linear(a,z) if c['architecture']=='GIN' and not dq else LinearQuantized(a,z,layer_quantizers=p)
            self.lin1=dense(width,width,lq); self.lin2=dense(width,b.classes,pos)
        if 'batchnorm_momentum' in c:
            momentum=c['batchnorm_momentum']
            if not 0. < momentum <= 1.:
                raise ValueError('batchnorm_momentum must be in (0, 1]')
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d): module.momentum=momentum

    def forward(self,data):
        c=self.c
        if c['node']: x=data.x
        elif c['regression']: x=self.embedding(data.x.reshape(-1))
        else: x=self.embedding(torch.cat((data.x,data.pos),1))
        mask=(torch.bernoulli(data.prob_mask).bool() if self.training else
              torch.zeros(len(x),dtype=torch.bool,device=x.device)) if c['framework']=='DQ' else None
        if c['node'] and c['architecture']=='GIN': x=F.dropout(x,.5,training=self.training)
        residual=x
        for i,(conv,norm) in enumerate(zip(self.convs,self.norms)):
            old=x
            x=conv(x,data.edge_index,mask) if c['framework']=='DQ' else conv(x,data.edge_index)
            if c['node']:
                if i<c['layers']-1:
                    if c['architecture']!='GIN': x=F.relu(x)
                    x=F.dropout(x,.5,training=self.training)
            elif c['architecture']=='GIN':
                x=x+residual; residual=old
            else:
                x=F.relu(norm(x))+old
        if not c['node']:
            x=global_mean_pool(x,data.batch)
            x=F.dropout(F.relu(self.lin1(x)),self.head_dropout,training=self.training)
            x=self.lin2(x)
        return x


def native_module(name):
    module=importlib.import_module('.vendor.'+name,__package__)
    module.global_mean_pool=global_mean_pool;module.softmax=stable_softmax
    return module


def _a2q_node_gat_forward(self,data):
    x=F.dropout(data.x,p=self.drop_out,training=self.training)
    x=self.conv1(x,data.edge_index,first_layer=True)
    x=F.dropout(F.relu(x),p=self.drop_out,training=self.training)
    return self.conv2(x,data.edge_index,first_layer=False)


def _zinc_gat_forward(self,data):
    x=self.embeding(data.x.flatten())
    budget=x.new_zeros(1)
    for conv in self.convs: x,budget=conv(x,data.edge_index,budget)
    x=global_mean_pool(x,data.batch)
    x,_,budget=self.lin1(x,data.edge_index,budget)
    x,_,budget=self.lin2(F.relu(x),data.edge_index,budget)
    return x,budget


def build(c,b,seed):
    from .vendor import baseline_quant,multi_quant
    baseline_quant.softmax=stable_softmax;multi_quant.softmax=stable_softmax
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if c['framework']!='A2Q': model=DQNetwork(c,b)
        elif c['node']:
            m=native_module('a2q_node_gat' if c['architecture']=='GAT' else 'a2q_node')
            m.dataset=SimpleNamespace(data=b.data,num_node_features=b.features,num_classes=b.classes)
            if c['architecture']=='GIN':
                model=m.GIN(m.dataset,c['layers'],c['width'],4,is_q=True,drop_out=c['dropout'])
            elif c['architecture']=='GCN':
                model=m.GCN(c['width'],4,is_q=True,drop_out=c['dropout'])
            else:
                model=m.GAT(c['width'],4,is_q=True,heads=c['heads'],drop_out=c['dropout'],drop_attn=c['dropout'])
                model.forward=MethodType(_a2q_node_gat_forward,model)
            for layer in model.modules():
                if isinstance(layer,nn.Linear):glorot(layer.weight)
        else:
            arch=c['architecture'].lower()
            source=f"a2q_{'zinc' if c['regression'] and arch!='gat' else 'graph'}_{arch}"
            m=native_module(source)
            common=dict(dataset=b.train,num_layers=c['layers'],hidden_units=c['width'],bit=4,
                        is_q=True,num_deg=c['num_deg'],uniform=True,init=c['init'])
            if arch=='gin':model=m.qGIN(**common)
            elif arch=='gcn':model=m.GCN(**common,add_self_loops=False)
            else:
                # Construct graph attention width h*heads and a one-head final layer,
                # including on ZINC where the old script silently discarded heads.
                class Metadata:
                    num_classes=b.classes
                    def __getitem__(self,index):
                        return SimpleNamespace(x=torch.empty(1,1),pos=torch.empty(1,2)) if c['regression'] else b.train[index]
                common['dataset']=Metadata()
                model=m.GAT(**common,heads=c['heads'],out_dim=c['width']*c['heads'],add_self_loops=False)
                if c['regression']:
                    model.embeding=nn.Embedding(28,c['width']*c['heads'])
                    model.forward=MethodType(_zinc_gat_forward,model)
        if c['framework']=='A2Q':
            for name,p in model.named_parameters():
                if name.endswith('.bit') and 'fea_quant' not in name:
                    p.requires_grad_(False)
                # qGATConv's historical unused .weight is never in forward.
                if c['architecture']=='GAT' and name.endswith('.weight'):
                    owner=model.get_submodule(name.rsplit('.',1)[0])
                    if hasattr(owner,'q_gat_attn'):p.requires_grad_(False)
    install_reductions(model)
    return model


def predict(model,c,data):
    if data.x.device.type!='cuda':raise RuntimeError('GPU model execution required')
    out=model(data)
    if isinstance(out,tuple):return out[0],out[1].squeeze()
    if c['framework']=='A2Q':
        budget=sum(c['width']*(c['heads'] if 'q_gat_fea' in n else 1)*p.abs().sum()/(8*1024)
                   for n,p in model.named_parameters() if n.endswith('.bit') and ('fea_quant' in n or 'q_gat_fea' in n))
    else:budget=out.new_zeros(())
    return out,budget


def optimizer(model,c):
    groups={}
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        decay=0.
        if c['framework']!='A2Q': kind,lr,decay='backbone',c['lr'],c['weight_decay']
        elif n.endswith('.bit'):kind,lr='bits',c['lr_bit']
        elif 'fea_quant.gama' in n:kind,lr='feature_scale',c['lr_feature']
        elif 'weight_quant.alpha' in n:kind,lr='weight_scale',c['lr_weight']
        elif 'q_xw.alpha' in n or 'q_gat_fea.gama' in n:kind,lr='message_scale',c['lr_message']
        elif 'q_gat_attn.alpha' in n:kind,lr='attention_scale',c['lr_attention']
        elif 'q_gat_edge.alpha' in n:kind,lr='edge_scale',c['lr_edge']
        else:kind,lr,decay='backbone',c['lr'],c['weight_decay']
        groups.setdefault(kind,dict(params=[],lr=lr,weight_decay=decay,name=kind))['params'].append(p)
    return torch.optim.Adam(list(groups.values()))


@torch.no_grad()
def project(model,c):
    if c['framework']=='A2Q':
        for n,p in model.named_parameters():
            if p.requires_grad and n.endswith('.bit'):p.clamp_(*c['bit_bounds'])


def load_state(model,state):
    # Legacy observer buffers start empty and acquire scalar shape on first call.
    for n,m in model.named_modules():
        if isinstance(m,IntegerQuantizer):
            for key in ('min_val','max_val'):
                value=state[(n+'.' if n else '')+key]
                setattr(m,key,torch.empty_like(value,device=next(model.parameters()).device))
    model.load_state_dict(state,strict=True)

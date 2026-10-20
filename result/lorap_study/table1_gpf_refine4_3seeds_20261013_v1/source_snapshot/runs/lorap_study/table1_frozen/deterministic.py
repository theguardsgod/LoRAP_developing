"""Stable CSR reductions: atomic summation noise can cross INT4 thresholds."""
from types import MethodType
import torch
from torch_scatter import segment_csr


def sum_by_index(src,index,n):
    if not len(index):return src.new_zeros((n,*src.shape[1:]))
    order=torch.argsort(index,stable=True)
    counts=torch.bincount(index,minlength=n)
    ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    return segment_csr(src[order],ptr,reduce='sum')


def mean_pool(x,batch,size=None):
    n=int(batch.max())+1 if size is None else size
    counts=torch.bincount(batch,minlength=n).clamp_min(1)
    return sum_by_index(x,batch,n)/counts[:,None]


def stable_softmax(src,index=None,ptr=None,num_nodes=None,dim=0):
    if dim!=0 or index is None:raise ValueError('This study uses indexed node-dimension attention')
    n=int(index.max())+1 if num_nodes is None else num_nodes
    order=torch.argsort(index,stable=True);counts=torch.bincount(index,minlength=n)
    pointer=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    maximum=segment_csr(src.detach()[order],pointer,reduce='max')
    exp=(src-maximum[index]).exp()
    denominator=segment_csr(exp[order],pointer,reduce='sum')
    return exp/(denominator[index]+1e-16)


def install(model):
    for layer in model.modules():
        if hasattr(layer,'propagate') and hasattr(layer,'aggregate') and hasattr(layer,'message'):
            custom=hasattr(layer,'__aggr_params__')
            def aggregate(self,inputs,index,ptr=None,dim_size=None,custom=custom):
                n=dim_size if dim_size is not None else ptr
                if n is None:n=int(index.max())+1
                return sum_by_index(inputs,index,n)
            layer.aggregate=MethodType(aggregate,layer)

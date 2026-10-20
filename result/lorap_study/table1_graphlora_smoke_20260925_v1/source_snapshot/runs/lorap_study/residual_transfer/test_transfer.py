import copy
import os
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from ..full_run import require_cuda
from ..models import state_digest
from ..residual_adapt.run import update_best,verify_selection
from .config import configuration
from .models import build,forward,Operator,loop_edges,optimizer,storage_proxy,project_quantizers
from .channel import calibrate,GatedModel,Diagnostics,mixed_quantize,quantize,metadata


def graph(device):
    generator=torch.Generator(device=device).manual_seed(914099)
    n=20
    x=torch.rand(n,6,generator=generator,device=device)*2
    src=torch.arange(n-2,device=device)
    edges=torch.stack((torch.cat((src,src.roll(1))),torch.cat((src.roll(1),src))))
    masks={s:torch.zeros(n,dtype=torch.bool,device=device) for s in ('train','val','test')}
    masks['train'][:12]=True;masks['val'][12:16]=True;masks['test'][16:]=True
    return Data(x=x,y=torch.arange(n,device=device)%3,edge_index=edges,
                **{s+'_mask':m for s,m in masks.items()})


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        cls.device=require_cuda()

    def model(self,architecture):
        c=configuration('Cora',architecture)
        c.update(hidden=8,heads=2)
        d=graph(self.device)
        b=build(c,d,41).to(self.device).eval()
        return c,d,b

    def test_weighted_operator_matches_dense_and_isolated_nodes(self):
        d=graph(self.device)
        edges=loop_edges(d.edge_index,len(d.x))
        weights=torch.rand(edges.size(1),2,device=self.device)
        x=torch.randn(len(d.x),2,4,device=self.device)
        actual=Operator(edges,weights)(x)
        expected=[]
        for h in range(2):
            matrix=torch.zeros(len(x),len(x),device=self.device)
            matrix.index_put_((edges[1],edges[0]),weights[:,h],accumulate=True)
            expected.append(matrix@x[:,h])
        torch.testing.assert_close(actual,torch.stack(expected,1),rtol=1e-6,atol=1e-6)
        self.assertTrue(torch.isfinite(actual[-2:]).all())
        codes=torch.randint(-127,128,(len(x),2,2),device=self.device,dtype=torch.int8)
        scale=torch.rand(2,2,device=self.device)
        op=Operator(edges,weights)
        self.assertTrue(torch.equal(op.codes(codes,scale),op(codes.float()*scale[None])))

    def test_gcn_matches_independent_normalized_dense_reference(self):
        c,d,b=self.model('GCN')
        layer=b.layers[0]
        h=layer.lin(d.x)
        q=layer.q_xw(h.T).T
        edges=loop_edges(d.edge_index,len(h))
        adjacency=torch.zeros(len(h),len(h),device=self.device)
        adjacency.index_put_((edges[1],edges[0]),torch.ones(edges.size(1),device=self.device),accumulate=True)
        inverse=adjacency.sum(1).rsqrt()
        expected=(inverse[:,None]*adjacency*inverse[None])@q
        torch.testing.assert_close(layer(d.x,d.edge_index,None,0),expected,atol=1e-6,rtol=1e-6)

    def test_gat_matches_independent_edge_softmax_reference(self):
        c,d,b=self.model('GAT')
        layer=b.layers[0]
        h=layer.lin(d.x)
        q=layer.q_gat_fea(h).view(len(h),layer.heads,layer.width)
        edges=loop_edges(d.edge_index,len(h))
        att=layer.q_gat_attn(layer.att[0])
        logits=(torch.cat((q[edges[1]],q[edges[0]]),-1)*att[None]).sum(-1)
        logits=F.leaky_relu(layer.q_gat_edge(logits.T).T,.2)
        expected=torch.zeros_like(q)
        for node in range(len(h)):
            mask=edges[1]==node
            probabilities=logits[mask].softmax(0)
            expected[node]=(probabilities[...,None]*q[edges[0,mask]]).sum(0)
        torch.testing.assert_close(layer(d.x,d.edge_index,None,0),expected.flatten(1)+layer.bias,atol=1e-6,rtol=1e-6)

    def test_zero_gate_equivalence_and_exact_diagnostic_switch(self):
        for architecture in ('GIN','GCN','GAT'):
            with self.subTest(architecture=architecture):
                c,d,b=self.model(architecture)
                cal,_=calibrate(b,c,d)
                wrapped=GatedModel(b,c,cal).to(self.device)
                with torch.no_grad():
                    original=forward(b,c,d)
                    zero=wrapped(d)
                    self.assertTrue(torch.equal(original,zero))
                    wrapped.gates.fill_(.37)
                    plain=wrapped(d)
                    collector=Diagnostics(d.val_mask)
                    wrapped.observer=collector
                    self.assertTrue(torch.equal(plain,wrapped(d)))
                    self.assertEqual(len(collector.rows),len(cal))

    def test_gate_gradients_and_frozen_backbone(self):
        for architecture in ('GIN','GCN','GAT'):
            with self.subTest(architecture=architecture):
                c,d,b=self.model(architecture)
                cal,_=calibrate(b,c,d)
                wrapped=GatedModel(b,c,cal).to(self.device)
                frozen=state_digest(b,'all')
                opt=torch.optim.Adam([wrapped.gates],lr=.01)
                loss=F.cross_entropy(wrapped(d)[d.train_mask],d.y[d.train_mask])
                loss.backward()
                self.assertIsNotNone(wrapped.gates.grad)
                self.assertTrue(torch.isfinite(wrapped.gates.grad).all())
                self.assertGreater(float(wrapped.gates.grad.abs().sum()),0)
                self.assertTrue(all(p.grad is None for p in b.parameters()))
                opt.step();wrapped.project()
                self.assertEqual(frozen,state_digest(b,'all'))
                self.assertTrue(((wrapped.gates>=0)&(wrapped.gates<=1)).all())

    def test_real_baseline_backward_finite(self):
        for architecture in ('GIN','GCN','GAT'):
            with self.subTest(architecture=architecture):
                c,d,b=self.model(architecture)
                b.train();opt=optimizer(b,c)
                logits=forward(b,c,d)
                loss=F.cross_entropy(logits[d.train_mask],d.y[d.train_mask])+c['a_loss']*F.relu(storage_proxy(b,c)-c['a_storage']).square()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None))
                opt.step()

    def test_calibration_training_only_and_bit_budget_match(self):
        for architecture in ('GIN','GCN','GAT'):
            with self.subTest(architecture=architecture):
                c,d,b=self.model(architecture)
                a,manifest=calibrate(b,c,d)
                d.y[~d.train_mask]=d.y[~d.train_mask].roll(1)
                again,_=calibrate(b,c,d)
                for first,second,row in zip(a,again,manifest):
                    for name in first:
                        self.assertTrue(torch.equal(first[name],second[name]))
                    self.assertEqual(int(first['increments'].sum()),row['extra_message_bits_per_node'])
                    self.assertTrue(all(bool(d.train_mask[k]) for k in row['basis_training_node_ids']))
                wrapped=GatedModel(b,c,a)
                wrapped.mode='higher'
                self.assertTrue(torch.isfinite(wrapped(d)).all())

    def test_unit_gate_identity_and_quantizer_bypass(self):
        for architecture in ('GIN','GCN','GAT'):
            with self.subTest(architecture=architecture):
                c,d,b=self.model(architecture)
                cal,_=calibrate(b,c,d)
                wrapped=GatedModel(b,c,cal)
                with torch.no_grad():
                    wrapped.mode='unit'
                    collector=Diagnostics(d.val_mask)
                    wrapped.observer=collector
                    wrapped(d)
                    self.assertTrue(all(r['orthogonal_identity_relative_gap']<2e-5 for r in collector.rows))
                    wrapped.observer=None
                    wrapped.bypass_values=True
                    a=wrapped(d)
                    wrapped.mode='none'
                    self.assertTrue(torch.equal(a,wrapped(d)))

    def test_zero_bit_increment_matches_existing_quantizer(self):
        from .models import message_quantizers
        for architecture in ('GIN','GCN','GAT'):
            c,d,b=self.model(architecture)
            quantizer=message_quantizers(b,c)[0]
            width=c['hidden']*c['heads'] if architecture=='GAT' else c['hidden']
            x=torch.randn(len(d.x),width,device=self.device)
            q=quantize(x,quantizer)
            self.assertTrue(torch.equal(q,mixed_quantize(x,quantizer,torch.zeros(width,device=self.device))))

    def test_selection_epoch_zero_ties_and_immutable_state(self):
        model=torch.nn.Linear(2,2,device=self.device)
        initial=update_best(None,1.,0,model)
        self.assertIs(update_best(initial,1.,1,model),initial)
        with torch.no_grad():model.weight.add_(1)
        self.assertFalse(torch.equal(initial['model']['weight'],model.weight.detach().cpu()))
        last=dict(epoch=1,history=[dict(epoch=0,validation=dict(loss=1.)),dict(epoch=1,validation=dict(loss=1.))],best=initial)
        verify_selection(last,initial)

    def test_zero_reference_error_has_no_reduction_ratio(self):
        x=torch.zeros(4,1,3,device=self.device)
        edges=torch.empty(2,0,dtype=torch.long,device=self.device)
        collector=Diagnostics(torch.ones(4,dtype=torch.bool,device=self.device))
        collector(0,x,x,Operator(edges,eps=x.new_zeros(())),None,None,None,0.)
        self.assertIsNone(collector.rows[0]['mse_reduction'])
        self.assertIsNone(collector.rows[0]['cosine'])

    def test_quantizer_projection_prevents_invalid_zero_bit(self):
        c,d,model=self.model('GCN')
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name.endswith('.bit') and p.requires_grad:
                    p.fill_(.1);p[0]=18
        project_quantizers(model,c)
        for name,p in model.named_parameters():
            if name.endswith('.bit'):
                self.assertTrue(((p>=1)&(p<=16)).all())
        self.assertTrue(torch.isfinite(forward(model,c,d)).all())


if __name__=='__main__':
    unittest.main()

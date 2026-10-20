"""Check trainable gate behavior, frozen state, BN intervention and selection."""

import copy
import os
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from ..models import state_digest
from ..residual_channel.experiment import forward as fixed_forward
from .model import AdaptedGIN, digest_subset, is_bn_stat, is_quantizer, optimizer_for, recalibrate_bn
from .run import LocalDiagnostics, epoch_order, update_best, verify_selection


class Linear(nn.Linear):
    def forward(self,x,edges,budget):
        return F.linear(x,self.weight,self.bias),edges,budget


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn=nn.BatchNorm1d(4)

    def forward(self,x,edges,budget):
        return self.bn(x),budget


class InputPrompt(nn.Module):
    def add(self,x):
        return x


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        from quantize_function.u_quant_gc_bit_debug import u_quant_xw
        self.q_xw=u_quant_xw(1,4,4)
        self.eps=nn.Parameter(torch.zeros(1))
        self.mlp=MLP()
        with torch.no_grad():
            self.q_xw.alpha.fill_(.1)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_node=InputPrompt()
        self.embedding=Linear(4,4,bias=False)
        self.convs=nn.ModuleList([Conv() for _ in range(4)])
        self.lin1=Linear(4,4,bias=False)
        self.lin2=Linear(4,2,bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(4))
            self.lin1.weight.copy_(torch.eye(4))
            self.lin2.weight.copy_(torch.tensor([[.2,.4,.1,.3],[-.3,-.2,.2,.1]]))


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.set_num_threads(2)
        cls.device=torch.device("cuda:0")

    def setUp(self):
        torch.manual_seed(71)
        self.base=Tiny().to(self.device).eval()
        code=dict(basis=torch.eye(4,device=self.device)[:,:2],scale=torch.full((2,),.01,device=self.device))
        self.cal=[dict(graph=code,channels={"higher_16":torch.tensor([0,1],device=self.device)}) for _ in range(4)]
        graph=Data(x=torch.rand(8,2)+.25,pos=torch.rand(8,2)+.25,
                   edge_index=torch.tensor([[0,1,2,3,4,5],[1,2,3,0,5,4]]),y=torch.tensor([0]))
        self.batch=Batch.from_data_list([graph,graph.clone()]).to(self.device)

    def test_zero_initialization_and_unit_gate_reference(self):
        m=AdaptedGIN(self.base,self.cal,"gate_only").to(self.device)
        with torch.no_grad():
            torch.testing.assert_close(m(self.batch),fixed_forward(self.base,self.batch),rtol=0,atol=0)
            m.gates.fill_(1)
            torch.testing.assert_close(m(self.batch),fixed_forward(self.base,self.batch,"graph",self.cal),rtol=0,atol=0)

    def test_gate_gradient_at_zero_and_projection(self):
        m=AdaptedGIN(self.base,self.cal,"gate_only").to(self.device)
        before=state_digest(m.backbone,"all")
        codes=digest_subset(m,lambda n:n.startswith("codes."))
        m.training_mode()
        opt=optimizer_for(m,dict(gate_lr=.01,backbone_lr=.0001))
        loss=F.cross_entropy(m(self.batch),self.batch.y)
        loss.backward()
        self.assertTrue(torch.isfinite(m.gates.grad).all())
        self.assertGreater(float(m.gates.grad.abs().max()),0.)
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        opt.step()
        m.project_gates()
        self.assertTrue(((m.gates>=0)&(m.gates<=1)).all())
        self.assertGreater(float(m.gates.max()),0.)
        self.assertEqual(before,state_digest(m.backbone,"all"))
        self.assertEqual(codes,digest_subset(m,lambda n:n.startswith("codes.")))

    def test_finetuning_preserves_quantizers_and_codes(self):
        m=AdaptedGIN(self.base,self.cal,"gate_ft").to(self.device)
        frozen=digest_subset(m,is_quantizer)
        code=digest_subset(m,lambda n:n.startswith("codes."))
        before=m.backbone.lin2.weight.detach().clone()
        m.training_mode()
        opt=optimizer_for(m,dict(gate_lr=.01,backbone_lr=.0001))
        F.cross_entropy(m(self.batch),self.batch.y).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        opt.step()
        m.project_gates()
        self.assertEqual(frozen,digest_subset(m,is_quantizer))
        self.assertEqual(code,digest_subset(m,lambda n:n.startswith("codes.")))
        self.assertFalse(torch.equal(before,m.backbone.lin2.weight))

    def test_bn_changes_only_statistics_and_uses_node_moments(self):
        m=AdaptedGIN(self.base,self.cal,"gate_only").to(self.device)
        fixed=digest_subset(m,lambda n:not is_bn_stat(n))
        stats=digest_subset(m,is_bn_stat)
        records=recalibrate_bn(m,[self.batch],self.device)
        self.assertEqual(fixed,digest_subset(m,lambda n:not is_bn_stat(n)))
        self.assertNotEqual(stats,digest_subset(m,is_bn_stat))
        self.assertEqual(len(records),4)
        self.assertTrue(all(r["nodes"]==16 for r in records))
        x=torch.tensor([[1.,2.,3.,4.],[3.,4.,7.,10.]],device=self.device)
        # Independent check of the moment formula used by population calibration.
        variance=(x.double().square().sum(0)-x.double().sum(0).square()/len(x))/(len(x)-1)
        torch.testing.assert_close(variance,x.double().var(0,unbiased=True))

    def test_diagnostics_and_bypass_invariance(self):
        m=AdaptedGIN(self.base,self.cal,"gate_only").to(self.device)
        with torch.no_grad():
            m.gates.copy_(torch.tensor([.1,.2,.3,.4],device=self.device))
            torch.testing.assert_close(m(self.batch),m(self.batch,LocalDiagnostics()),rtol=0,atol=0)
            bypass=m(self.batch,bypass_messages=True)
            m.gates.zero_()
            torch.testing.assert_close(bypass,m(self.batch,bypass_messages=True),rtol=0,atol=0)

    def test_epoch_zero_fallback_ties_and_snapshot(self):
        m=nn.Linear(1,1).to(self.device)
        best=update_best(None,1.,0,m)
        original=best["model"]["weight"].clone()
        with torch.no_grad():
            m.weight.add_(1)
        self.assertTrue(torch.equal(original,best["model"]["weight"]))
        self.assertIs(best,update_best(best,1.,1,m))
        self.assertIs(best,update_best(best,2.,2,m))
        with self.assertRaises(FloatingPointError):
            update_best(best,float("nan"),3,m)
        last=dict(epoch=2,history=[dict(epoch=i,validation=dict(loss=l)) for i,l in enumerate((1.,1.,2.))],best=best)
        verify_selection(last,best)
        changed=copy.deepcopy(best)
        changed["model"]["weight"].add_(.001)
        with self.assertRaises(AssertionError):
            verify_selection(last,changed)

    def test_paired_epoch_order(self):
        self.assertEqual(epoch_order(100,41,1),epoch_order(100,41,1))
        self.assertNotEqual(epoch_order(100,41,1),epoch_order(100,41,2))
        self.assertEqual(sorted(epoch_order(100,41,1)),list(range(100)))


if __name__=="__main__":
    unittest.main()

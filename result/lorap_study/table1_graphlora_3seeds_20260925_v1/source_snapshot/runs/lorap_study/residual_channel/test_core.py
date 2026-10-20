"""CUDA-only numerical and mechanism invariants, using unittest."""

import unittest
from unittest.mock import patch
import os

import torch

from .core import (canonical_basis, encode, error_terms, graph_operator,
                   propagate_codes, residual_correction, select_channels, shuffle_codes,
                   higher_precision, selected_precision)
from .experiment import Reservoir, calibrate, forward, Diagnostics


class ResidualChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; tests must not silently skip or fall back")
        cls.device = torch.device("cuda:0")
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(812)
        self.r = torch.randn(13, 7, device=self.device)
        self.b = torch.linalg.qr(torch.randn(7, 2, device=self.device)).Q
        self.edges = torch.tensor([[0,1,1,2,5,7,8,8],[1,0,0,3,6,8,7,8]], device=self.device)
        self.eps = torch.tensor([.3], device=self.device)
        self.scale = torch.tensor([.03,.07], device=self.device)

    def test_zero_residual(self):
        correction, details = residual_correction(torch.zeros_like(self.r), self.edges, self.eps, self.b,self.scale)
        self.assertEqual(int(correction.count_nonzero()), 0)
        self.assertEqual(int(details["codes"].count_nonzero()), 0)

    def test_isolated_nodes_and_real_self_coefficient(self):
        empty = torch.empty(2,0,device=self.device,dtype=torch.long)
        correction, d = residual_correction(self.r, empty,self.eps,self.b,self.scale)
        torch.testing.assert_close(correction,(1+self.eps)*d["decoded"]@self.b.T)

    def test_multiedges_selfloop_removal_and_permutation(self):
        x = torch.tensor([[1.],[2.],[4.]],device=self.device)
        e = torch.tensor([[0,0,1,2],[1,1,0,2]],device=self.device)
        actual = graph_operator(x,e,torch.tensor([.5],device=self.device))
        torch.testing.assert_close(actual,torch.tensor([[3.5],[5.],[6.]],device=self.device))
        torch.testing.assert_close(actual,graph_operator(x,e[:,[3,1,0,2]],torch.tensor([.5],device=self.device)))

    def test_int8_rounding_saturation_int32_sum(self):
        r = torch.tensor([[.5,-.5],[1000,-1000],[1.49,-1.49]],device=self.device)
        codes,_ = encode(r,torch.eye(2,device=self.device),torch.ones(2,device=self.device))
        self.assertEqual(codes.dtype,torch.int8)
        self.assertEqual(codes.cpu().tolist(),[[1,-1],[127,-127],[1,-1]])
        codes = torch.full((1001,2),127,dtype=torch.int8,device=self.device)
        edges = torch.stack((torch.arange(1,1001,device=self.device),torch.zeros(1000,device=self.device,dtype=torch.long)))
        value = propagate_codes(codes,edges,torch.zeros(1,device=self.device),torch.ones(2,device=self.device))
        self.assertEqual(value[0,0].item(),127127)

    def test_exact_decomposition_and_float_upper_bound(self):
        for floating in (False,True):
            correction,d = residual_correction(self.r,self.edges,self.eps,self.b,self.scale,floating=floating)
            terms = error_terms(self.r,correction,self.edges,self.eps,self.b,d["decoded"])
            self.assertLess(terms["identity_relative_gap"].item(),1e-6)
            self.assertEqual(terms["after_sq"]<terms["before_sq"],terms["side_quantization_sq"]<terms["captured_sq"])
            if floating:
                self.assertLess(terms["side_quantization_sq"].item(),1e-10)

    def test_graph_basis_objective(self):
        e = graph_operator(self.r,self.edges,self.eps)
        basis, spectrum = canonical_basis(e)
        torch.testing.assert_close(basis.T@basis,torch.eye(2,device=self.device))
        error = (e-e@basis@basis.T).double().square().sum()
        torch.testing.assert_close(error,spectrum[2:].sum(),atol=2e-5,rtol=2e-6)

    def test_shuffle_preserves_each_graph_multiset(self):
        codes = torch.arange(26,device=self.device,dtype=torch.int8).reshape(13,2)
        batch = torch.tensor([0]*5+[1]*8,device=self.device)
        g = torch.Generator(device=self.device).manual_seed(44)
        out = shuffle_codes(codes,batch,g)
        for mask in (batch==0,batch==1):
            torch.testing.assert_close(out[mask].sort(dim=0).values,codes[mask].sort(dim=0).values)

    def test_no_padding_in_channel_budget(self):
        gains = torch.tensor([-1.,3.,0.,2.,-5.],device=self.device)
        self.assertEqual(select_channels(gains,16).cpu().tolist(),[1,3])
        self.assertEqual(select_channels(gains,1).cpu().tolist(),[1])

    def test_node_sample_replay_and_limit(self):
        a,b = Reservoir(7,self.device,10),Reservoir(7,self.device,10)
        for _ in range(4):
            a.add(error=self.r)
            b.add(error=self.r*2)
        self.assertEqual(len(a.ids),10)
        torch.testing.assert_close(a.ids,b.ids)
        torch.testing.assert_close(a.values["error"]*2,b.values["error"])

    def test_selected_precision_preserves_other_channels_and_range(self):
        from quantize_function.u_quant_gc_bit_debug import u_quant_xw
        q = u_quant_xw(1,7,4).to(self.device)
        with torch.no_grad():
            q.alpha.fill_(.2)
        indices = torch.tensor([1,4],device=self.device)
        actual = selected_precision(self.r,q,indices)
        expected = q(self.r.T).T
        expected[:,indices] = higher_precision(self.r,q)[:,indices]
        torch.testing.assert_close(actual,expected)
        self.assertLessEqual(actual.abs().max().item(),1.400001)

    def test_frozen_model_and_diagnostic_switch(self):
        from torch_geometric.data import Data, Batch
        from ..full_models import build_full_backbone
        from ..models import state_digest
        data = Data(x=torch.randn(13,1),pos=torch.randn(13,2),
                    edge_index=self.edges.cpu(),y=torch.tensor([1]))
        config = dict(task="graph",hidden_units=8,layers=4,bit=4,num_deg=20,
                      init="uniform",num_prompts=20,rank=2,classes=3)
        model = build_full_backbone(config,41,{"train":[data]}).to(self.device).eval().requires_grad_(False)
        digest = state_digest(model,"all")
        calibration,_ = calibrate(model,[data,data.clone()],self.device)
        batch = Batch.from_data_list([data,data.clone()]).to(self.device)
        with torch.no_grad():
            original,_ = model(batch)
        with patch("runs.lorap_study.residual_channel.experiment.error_terms",side_effect=AssertionError("Unexpected reference branch")):
            actual = forward(model,batch)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual,original,atol=1e-5,rtol=1e-5)
        for mode in ("none","graph","higher_16"):
            a = forward(model,batch,mode,calibration)
            b = forward(model,batch,mode,calibration,Diagnostics(calibration,self.device))
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(forward(model,batch,"bypass",calibration),
                                   forward(model,batch,"bypass_side",calibration))
        self.assertEqual(digest,state_digest(model,"all"))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()

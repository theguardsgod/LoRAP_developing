import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from ..full_run import require_cuda,cpu_tree
from ..models import state_digest
from ..residual_transfer.config import configuration as backbone_configuration
from ..residual_transfer.models import build,forward,Operator
from ..residual_transfer.test_transfer import graph
from ..residual_adapt.run import update_best,verify_selection
from .config import configuration
from .model import QuantOnlyModel,Predictor,calibrate,context_from_quantized,local_target
from .diagnostics import Diagnostics
from .run import export,inference_audit


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
        cls.device=require_cuda()

    def case(self,architecture='GCN',kind='error'):
        d=graph(self.device)
        bc=backbone_configuration('Cora',architecture);bc.update(hidden=8,heads=2)
        c=configuration('Cora',architecture,True)
        b=build(bc,d,41).to(self.device).eval()
        stats,scales,records=calibrate(b,bc,c,d)
        model=QuantOnlyModel(b,bc,c,stats,41,kind).to(self.device)
        return d,bc,c,b,stats,scales,records,model

    def test_zero_initialization_and_paired_parameters(self):
        for architecture in ('GIN','GCN','GAT'):
            d,bc,c,b,stats,scales,records,model=self.case(architecture)
            task=QuantOnlyModel(b,bc,c,stats,41,'task').to(self.device)
            mlp=QuantOnlyModel(b,bc,c,stats,41,'mlp').to(self.device)
            self.assertEqual(state_digest(task.predictors,'all'),state_digest(model.predictors,'all'))
            with torch.no_grad():
                original=forward(b,bc,d)
                for wrapped in (model,task,mlp): self.assertTrue(torch.equal(original,wrapped(d)))

    def test_parameter_match_for_all_formal_widths(self):
        for width in (3,6,7,16,64):
            c=configuration('Cora','GCN')
            stats=dict(feature_mean=torch.zeros(width,device=self.device),feature_std=torch.ones(width,device=self.device),
                context_mean=torch.zeros(4,device=self.device),context_std=torch.ones(4,device=self.device),output_scale=torch.ones((),device=self.device))
            low=Predictor(stats,c,'error').to(self.device);mlp=Predictor(stats,c,'mlp').to(self.device)
            self.assertLessEqual(abs(low.parameter_count-mlp.parameter_count)/low.parameter_count,.05)

    def test_gradients_finite_frozen_backbone_and_basis_learning(self):
        for architecture in ('GIN','GCN','GAT'):
            for kind in ('task','error','mlp'):
                d,bc,c,b,stats,scales,records,model=self.case(architecture,kind)
                frozen=state_digest(b,'all');initial=state_digest(model.predictors,'all')
                opt=torch.optim.Adam(model.predictors.parameters(),lr=.001)
                for _ in range(2):
                    opt.zero_grad(set_to_none=True)
                    if kind=='error': logits,aux=model.supervised_forward(d,d.train_mask,scales)
                    else: logits,aux=model(d),0.
                    loss=F.cross_entropy(logits[d.train_mask],d.y[d.train_mask])+.1*aux
                    loss.backward()
                    gradients=[p.grad for p in model.predictors.parameters() if p.grad is not None]
                    self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
                    self.assertTrue(all(p.grad is None for p in b.parameters()))
                    opt.step()
                self.assertNotEqual(initial,state_digest(model.predictors,'all'))
                self.assertEqual(frozen,state_digest(b,'all'))
                self.assertTrue(torch.isfinite(model(d)).all())

    def test_teacher_never_called_during_prediction(self):
        for architecture in ('GIN','GCN','GAT'):
            d,bc,c,b,stats,scales,records,model=self.case(architecture)
            with patch('runs.lorap_study.quant_predict.model.local_target',side_effect=RuntimeError('forbidden')):
                self.assertTrue(torch.isfinite(model(d)).all())
                with self.assertRaises(RuntimeError): model.supervised_forward(d,d.train_mask,scales)

    def test_export_poisoned_reference_and_unlabeled_inputs(self):
        for architecture in ('GIN','GCN','GAT'):
            d,bc,c,b,stats,scales,records,model=self.case(architecture)
            with torch.no_grad():
                for p in model.predictors: p.right.normal_(std=.01)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'inference.pt'
                export(model,c,d,41,'error',path)
                audit=inference_audit(model,d,path,self.device)
                self.assertTrue(all(audit.values()))
                saved=torch.load(path,map_location='cpu',weights_only=False)
                self.assertNotIn('training_loss_scales',saved)

    def test_isolated_nodes_and_endpoint_is_not_clipping(self):
        q=torch.tensor([[[0.,7.]],[[7.,-7.]],[[1.,2.]]],device=self.device)
        edges=torch.tensor([[0,1],[1,0]],device=self.device)
        ctx=context_from_quantized(q,edges,torch.ones(1,2,device=self.device),torch.full((1,2),4.,device=self.device))
        torch.testing.assert_close(ctx[:,3],torch.tensor([1.,.5,0.],device=self.device))
        self.assertTrue(torch.equal(ctx[2],torch.zeros(4,device=self.device)))
        zero=context_from_quantized(q*0,edges,torch.ones(1,2,device=self.device),torch.ones(1,2,device=self.device))
        self.assertTrue(torch.equal(zero[:,3],torch.zeros(3,device=self.device)))
        self.assertFalse(ctx.requires_grad)

    def test_local_target_matches_independent_dense_operator(self):
        d=graph(self.device);n=len(d.x)
        x=torch.randn(n,2,4,device=self.device,requires_grad=True);q=x.detach().round()
        weights=torch.rand(d.num_edges,2,device=self.device,requires_grad=True)
        target=local_target(x,q,Operator(d.edge_index,weights))
        matrices=torch.zeros(2,n,n,device=self.device,dtype=torch.double)
        for h in range(2):
            matrices[h].index_put_((d.edge_index[1],d.edge_index[0]),weights[:,h].detach().double(),accumulate=True)
        expected=torch.stack([matrices[h]@(x.detach().double()-q.double())[:,h] for h in range(2)],1).flatten(1)
        torch.testing.assert_close(target,expected,atol=1e-12,rtol=1e-12)
        self.assertFalse(target.requires_grad)

    def test_teacher_loss_uses_training_targets_only(self):
        d,bc,c,b,stats,scales,records,model=self.case()
        _,before=model.supervised_forward(d,d.train_mask,scales)
        def altered(x,q,op):
            value=local_target(x,q,op)
            value[~d.train_mask]+=1e6
            return value
        with patch('runs.lorap_study.quant_predict.model.local_target',side_effect=altered):
            _,after=model.supervised_forward(d,d.train_mask,scales)
        self.assertTrue(torch.equal(before,after))

    def test_inference_calibration_does_not_use_reference_or_labels(self):
        d,bc,c,b,stats,scales,records,model=self.case()
        changed=d.clone();changed.y=(changed.y+1)%3
        def altered(x,q,op): return local_target(x,q,op)*10
        with patch('runs.lorap_study.quant_predict.model.local_target',side_effect=altered):
            again,other,_=calibrate(b,bc,c,changed)
        for a,z in zip(stats,again):
            for name in a: self.assertTrue(torch.equal(a[name],z[name]))
        self.assertFalse(torch.equal(scales,other))
        for record in records: self.assertTrue(all(bool(d.train_mask[i]) for i in record['training_node_ids']))

    def test_diagnostics_identity_and_no_output_change(self):
        for architecture in ('GIN','GCN','GAT'):
            d,bc,c,b,stats,scales,records,model=self.case(architecture)
            with torch.no_grad():
                for p in model.predictors: p.right.normal_(std=.01)
                collector=Diagnostics(d.val_mask,c)
                self.assertTrue(torch.equal(model(d),model.diagnose(d,collector)))
                self.assertTrue(all(r['identity_relative_gap']<1e-8 for r in collector.rows))
        collector=Diagnostics(d.val_mask,c)
        collector(0,torch.zeros(len(d.x),3,device=self.device),torch.zeros(len(d.x),3,device=self.device))
        self.assertIsNone(collector.rows[0]['mse_reduction'])

    def test_validation_selection_earliest_tie(self):
        model=torch.nn.Linear(2,2).to(self.device)
        history=[];best=None
        for epoch,loss in enumerate((2.,1.,1.,3.)):
            best=update_best(best,loss,epoch,model)
            history.append(dict(epoch=epoch,validation=dict(loss=loss)))
        last=dict(epoch=3,history=history,best=copy.deepcopy(best))
        verify_selection(last,best)
        self.assertEqual(best['epoch'],1)


if __name__=='__main__': unittest.main()

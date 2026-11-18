import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import portable as p
from .test_portable import Bundle, GraphBundle, cpu_predict, gm, final_evaluation
from ..table1_frozen.registry import CELLS


class GatRepairTests(unittest.TestCase):
    def test_actual_optimizer_groups_scale_without_changing_initialization(self):
        self.assertTrue(torch.cuda.is_available())
        for cell,scale in [('QAT_ZINC_GAT',.02),('A2Q_CIFAR10_GAT',.1)]:
            with tempfile.TemporaryDirectory() as temp, patch.object(p,'snapshot'):
                root=Path(temp)/'study'
                study=p.prepare(root,root/'data',[cell],[41],variants=['graphlora'],
                    gradient_clip_norm=1.,stability_epochs=30,learning_rate_scale=scale)
                job=study['jobs'][0];original=CELLS[cell]
                for name,value in original.items():
                    if name=='lr' or name.startswith('lr_'):
                        self.assertAlmostEqual(job['config'][name],value*scale)
                # CIFAR10 exposes all A2Q optimizer groups, using synthetic graphs.
                if cell=='A2Q_CIFAR10_GAT':
                    bundle=GraphBundle(job['config'])
                    model,initial=p.model_for(study,job,bundle,'cuda:0')
                    try:
                        opt=p.make_optimizer(model,job['config'])
                        original_opt=p.make_optimizer(model,original)
                        self.assertFalse(opt.state)
                        self.assertGreater(len(opt.param_groups),3)
                        for group,old in zip(opt.param_groups,original_opt.param_groups):
                            self.assertEqual(group['name'],old['name'])
                            self.assertAlmostEqual(group['lr'],old['lr']*scale)
                        self.assertEqual(p.digest(model.backbone),initial)
                    finally:p.close(model)
                self.assertEqual(study['variants'],['graphlora'])
                self.assertEqual(len(study['jobs']),1)
                self.assertFalse(list(root.glob('*/seed_*/none')))

    def test_single_method_screening_then_full_training(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(p,'snapshot'), \
             patch.object(p,'Bundle',Bundle), patch.object(p,'predict',cpu_predict), \
             patch.object(gm,'predict',cpu_predict), patch.object(final_evaluation,'predict',cpu_predict):
            root=Path(temp)/'study'
            study=p.prepare(root,root/'data',['QAT_Cora_GCN'],[41],variants=['graphlora'],
                learning_rate_scale=.1,gradient_clip_norm=1.,stability_epochs=2)
            job=study['jobs'][0];job['epochs']=3;p.write_json(root/'study.json',study)
            p.fit(root,study,job,'cpu')
            self.assertFalse((root/job['directory']/'done.json').exists())
            with self.assertRaises(RuntimeError):p.test_manifest(root,study)
            p.release_stability_gate(root,study)
            p.fit(root,study,job,'cpu')
            p.finish(root,study,'cpu')
            with patch.object(p,'model_for',side_effect=AssertionError('Repeated inference')):
                p.finish(root,study,'cpu')
            self.assertEqual(p.read(root/'completion_audit.json')['tests'],1)
            for scale in [0,-1,float('nan')]:
                with self.assertRaises(ValueError):
                    p.prepare(Path(temp)/'bad',root/'data',learning_rate_scale=scale)


if __name__=='__main__':unittest.main()

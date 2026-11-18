"""Raw-gradient rejection, actual clipping, and full-data screening resume."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import portable as p
from . import model as gm
from .test_portable import Bundle, cpu_predict
from ..table1_frozen.common import write_json
from ..table1_frozen import finish as final_evaluation


class StabilizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_clip_bounds_gradient_and_adam_moment(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[3., 4.]])
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        raw, after, clipped = p.guard_and_clip(model, 1e6, 1.)
        self.assertEqual(raw, 5.)
        self.assertTrue(clipped)
        self.assertLessEqual(after, 1.000001)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[.6, .8]]))
        opt.step()
        self.assertLessEqual(opt.state[model.weight]['exp_avg_sq'].max().item(), .001)

    def test_raw_explosion_is_not_hidden_by_clipping(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[2e6, 0.]])
        before = model.weight.detach().clone()
        opt = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(FloatingPointError, 'exceeds limit') as caught:
            p.guard_and_clip(model, 1e6, 1.)
            opt.step()
        self.assertEqual(caught.exception.gradient_diagnostics[0]['name'], 'weight')
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertFalse(opt.state)
        self.assertEqual(model.weight.grad[0, 0].item(), 2e6)

    def test_registration_applies_same_recipe_to_both_variants(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(p, 'snapshot'):
            study = p.prepare(Path(temp)/'new', Path(temp)/'data', ['QAT_MNIST_GAT'], [41,42,43],
                              learning_rate=.001, gradient_clip_norm=1., stability_epochs=30)
            self.assertEqual(len(study['jobs']), 6)
            self.assertTrue(all(j['config']['lr']==.001 and j['epochs']==200 for j in study['jobs']))
            self.assertEqual(study['hyperparameters']['gradient_norm_limit'], 1e6)
            self.assertEqual(study['hyperparameters']['stability_epochs'], 30)
            self.assertFalse(study['controls_reused'])
            changed = copy.deepcopy(study); changed['hyperparameters']['gradient_clip_norm'] = 2.
            self.assertNotEqual(p.job_key(study, study['jobs'][0]), p.job_key(changed, changed['jobs'][0]))

    def test_invalid_recipe_is_rejected_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(p, 'snapshot'):
            root=Path(temp)/'invalid'
            for kwargs in [dict(learning_rate=float('nan')), dict(learning_rate=0),
                           dict(gradient_clip_norm=float('inf')), dict(gradient_clip_norm=2e6),
                           dict(stability_epochs=0), dict(stability_epochs=200)]:
                with self.assertRaises(ValueError):
                    p.prepare(root,Path(temp)/'data',['QAT_MNIST_GAT'],[41],**kwargs)
                self.assertFalse(root.exists())

    def test_screening_resume_matches_continuous_and_test_is_gated(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(p,'snapshot'), \
             patch.object(p,'Bundle',Bundle), patch.object(p,'predict',cpu_predict), \
             patch.object(gm,'predict',cpu_predict), patch.object(final_evaluation,'predict',cpu_predict):
            root=Path(temp)/'staged'
            study=p.prepare(root,Path(temp)/'data',['QAT_Cora_GCN'],[41],
                            learning_rate=.001,gradient_clip_norm=.1,stability_epochs=1)
            for j in study['jobs']:j['epochs']=3
            write_json(root/'study.json',study)
            for j in study['jobs']:p.fit(root,study,j,'cpu')
            self.assertFalse(list(root.glob('*/seed_*/*/done.json')))
            with self.assertRaisesRegex(RuntimeError,'Stability'):
                p.test_manifest(root,study)
            failed=root/study['jobs'][0]['directory']/'failure.json'
            write_json(failed,dict(reason='test sentinel'))
            with self.assertRaisesRegex(RuntimeError,'failure'):
                p.release_stability_gate(root,study)
            failed.unlink()
            gate=p.release_stability_gate(root,study)
            self.assertTrue(gate['passed']);self.assertEqual(gate['tests'],0)
            self.assertEqual(len(gate['jobs']),2)
            with self.assertRaisesRegex(RuntimeError,'All paired'):
                p.test_manifest(root,study)
            changed=copy.deepcopy(study);changed['hyperparameters']['gradient_clip_norm']=.2
            with self.assertRaisesRegex(ValueError,'gate'):
                p.release_stability_gate(root,changed)
            for j in study['jobs']:p.fit(root,study,j,'cpu')
            continuous_root=Path(temp)/'continuous'
            continuous=p.prepare(continuous_root,Path(temp)/'data',['QAT_Cora_GCN'],[41],
                                 learning_rate=.001,gradient_clip_norm=.1)
            for j in continuous['jobs']:
                j['epochs']=3;p.fit(continuous_root,continuous,j,'cpu')
                expected=p.read(continuous_root/j['directory']/'done.json')
                actual=p.read(root/j['directory']/'done.json')
                self.assertEqual(expected['state_sha256'],actual['state_sha256'])
                self.assertEqual(expected['validation'],actual['validation'])
            p.finish(root,study,'cpu')
            with patch.object(p,'model_for',side_effect=AssertionError('Repeated test inference')):
                p.finish(root,study,'cpu')
            self.assertTrue(p.read(root/'completion_audit.json')['passed'])


if __name__=='__main__':
    unittest.main()

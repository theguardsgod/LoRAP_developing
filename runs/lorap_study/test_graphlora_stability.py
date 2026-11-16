"""Regression coverage for quantized zero vectors and fresh-study enforcement."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from .graphlora import (DEFAULT_NORM_FLOOR, OBJECTIVE_VERSION, checked_gradient_norm,
                       contrastive_loss, normalize_embeddings)
from .table1_graphlora import model as table_model
from .table1_graphlora import run
from .table1_frozen.common import config_hash, write_json
from .table1_frozen.registry import CELLS


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_normalization_has_bounded_jacobian_at_zero_and_near_zero(self):
        for row in ([0., 0., 0.], [1e-15, -1e-15, 0.], [.001, -.002, .003], [1., -2., 3.]):
            x = torch.tensor(row, dtype=torch.double, requires_grad=True)
            jacobian = torch.autograd.functional.jacobian(normalize_embeddings, x)
            self.assertTrue(torch.isfinite(jacobian).all())
            self.assertLessEqual(torch.linalg.matrix_norm(jacobian, 2).item(), 1 / DEFAULT_NORM_FLOOR + 1e-10)
            if x.norm() > DEFAULT_NORM_FLOOR:
                torch.testing.assert_close(normalize_embeddings(x), F.normalize(x, dim=-1), rtol=0, atol=0)
        for value in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                normalize_embeddings(torch.ones(2, 3), value)

    def test_mixed_zero_views_do_not_create_exploding_contrastive_gradients(self):
        torch.manual_seed(17)
        a = torch.randn(12, 5)
        b = torch.randn(12, 5)
        a[:8] = 0
        b[:4] = 0
        a[8] = 1e-15
        a.requires_grad_(); b.requires_grad_()
        loss = contrastive_loss(a, b, torch.arange(12) % 3)
        grads = torch.autograd.grad(loss, (a, b))
        self.assertTrue(torch.isfinite(loss))
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertLess(grad.norm().item(), 100.)
        # The same real failure condition used to pass finite-value checks.
        legacy = contrastive_loss(a, b, torch.arange(12) % 3, norm_floor=1e-12)
        legacy_grads = torch.autograd.grad(legacy, (a, b))
        self.assertGreater(max(g.norm().item() for g in legacy_grads), 1e8)

    def test_all_zero_and_singleton_views_have_finite_values_and_gradients(self):
        for size in (1, 7):
            a = torch.zeros(size, 4, requires_grad=True)
            b = torch.zeros(size, 4, requires_grad=True)
            loss = contrastive_loss(a, b)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertTrue(torch.isfinite(b.grad).all())

    def test_contrastive_autocast_uses_float32_similarities(self):
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(23)
            a = torch.randn(6, 4).to(dtype)
            a[:3] = 0
            b = torch.randn(6, 4).to(dtype)
            a.requires_grad_(); b.requires_grad_()
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = contrastive_loss(a, b)
            expected = contrastive_loss(a.float(), b.float())
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.backward()
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertTrue(torch.isfinite(b.grad).all())

    def test_regression_structure_handles_zero_latents(self):
        torch.manual_seed(29)
        nodes = torch.randn(8, 4)
        nodes[:5] = 0
        nodes.requires_grad_()
        edges = torch.tensor([[0, 1, 2, 5, 6], [1, 0, 3, 6, 7]])
        loss = table_model.regression_structure(nodes, edges, torch.tensor([0] * 4 + [1] * 4))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(nodes.grad).all())
        self.assertLess(nodes.grad.norm().item(), 100.)

    def test_disabled_auxiliary_objectives_are_not_evaluated(self):
        logits = torch.randn(6, 2, requires_grad=True)
        batch = Data(y=torch.tensor([0, 1, 0, 1, 0, 1]),
                     train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool))
        model = nn.Module()
        model.frozen = False
        model.last_views = (None, None)
        c = dict(node=True, regression=False, a_loss=0., a_storage=1.)
        hp = dict(run.HP, contrast_weight=0., structure_weight=0.)
        with patch.object(table_model, 'predict', return_value=(logits, logits.new_zeros(()))), \
             patch.object(table_model, 'contrastive_loss', side_effect=AssertionError('Disabled contrast ran')), \
             patch.object(table_model, 'structure_loss', side_effect=AssertionError('Disabled structure ran')):
            actual = table_model.objective(model, c, batch, hp)[0]
        expected = F.cross_entropy(logits[:2], batch.y[:2])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_finite_gradient_explosion_is_rejected_before_adam(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters())
        for p in model.parameters():
            p.grad = torch.full_like(p, 1e9)
        with self.assertRaisesRegex(FloatingPointError, 'optimizer was not stepped'):
            checked_gradient_norm(model)
        self.assertEqual(len(optimizer.state), 0)
        for p in model.parameters():
            p.grad.fill_(.1)
        self.assertLess(checked_gradient_norm(model).item(), 1.)

    def test_old_objective_is_rejected_before_any_fit_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'fresh initialization'):
                run.fit(root, dict(hyperparameters={}), dict(directory='old'), 'cpu')
            self.assertEqual(list(root.iterdir()), [])
        repaired = dict(objective_version=OBJECTIVE_VERSION, hyperparameters=dict(run.HP))
        run.require_current_objective(repaired)
        invalid = copy.deepcopy(repaired)
        invalid['hyperparameters']['normalization_floor'] = 0
        with self.assertRaises(ValueError):
            run.require_current_objective(invalid)

    def test_active_source_must_match_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            active = root / 'active'
            snapshot = root / 'source_snapshot'
            active.mkdir(); snapshot.mkdir()
            (active / 'fixture.py').write_text('new loss\n')
            (snapshot / 'fixture.py').write_text('old loss\n')
            write_json(root / 'source_manifest.json', {'fixture.py': hashlib.sha256(b'old loss\n').hexdigest()})
            with patch.object(run, 'REPO', active):
                with self.assertRaisesRegex(ValueError, 'Active source differs'):
                    run.verify_sources(root)

    def test_prepare_records_repaired_objective_and_relocated_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / 'reference'
            c = dict(CELLS['QAT_Cora_GCN'], seed=41, subset_limit=None, stage='baseline')
            baseline = reference / c['cell'] / 'seed_41' / 'baseline'
            baseline.mkdir(parents=True)
            (baseline / 'best.pt').write_bytes(b'checkpoint')
            sha = hashlib.sha256(b'checkpoint').hexdigest()
            write_json(baseline / 'config.json', c)
            write_json(baseline / 'done.json', dict(full_split=True, config_hash=config_hash(c)))
            write_json(baseline / 'test.json', dict(completed_test_evaluations=1, checkpoint_file_sha256=sha))
            with patch.object(run, 'snapshot'):
                study = run.prepare(root / 'new', reference, [c['cell']], [41], smoke=True,
                                    normalization_floor=.2, data_roots={'Cora': root / 'data'})
            self.assertEqual(study['objective_version'], OBJECTIVE_VERSION)
            self.assertEqual(study['hyperparameters']['normalization_floor'], .2)
            self.assertEqual(study['jobs'][0]['config']['data_root'], str(root / 'data'))
            self.assertEqual(study['jobs'][0]['reference']['config']['data_root'], c['data_root'])


if __name__ == '__main__':
    unittest.main()

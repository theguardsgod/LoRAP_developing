"""Expanded regularization, historical equivalence and exact AdamW resume."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data, Batch

from ..table1_frozen import backends
from ..table1_frozen.registry import CELLS
from . import run, report, refine, refine4_space, test_pipeline


class SyntheticGraphs:
    def __init__(self, c):
        self.c, self.features, self.classes = c, 5, 3
        self.graphs = []
        for label, n in enumerate((3, 5, 7)):
            index = torch.arange(n)
            self.graphs.append(Data(x=torch.linspace(.1, 1., n * 3).reshape(n, 3),
                pos=torch.stack((index / n, index.flip(0) / n), 1),
                edge_index=torch.stack((index, index.roll(1))), y=torch.tensor([label]),
                prob_mask=torch.full((n,), .2)))

    def batches(self, split, seed, epoch, device, limit=None, allow_test=False):
        if split == 'test' and not allow_test:
            raise RuntimeError('Synthetic test gate is locked')
        yield Batch.from_data_list(self.graphs).to(device)

    def manifest(self):
        return dict(synthetic=True, train_graphs=3, val_graphs=3)


class ExpandedSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        cls.device = 'cuda:0'
        torch.cuda.set_device(0)
        torch.set_num_threads(4)

    def config(self, **overrides):
        return dict(CELLS['QAT_CIFAR10_GCN'], **dict(width=8, layers=2, **overrides))

    def test_defaults_match_archived_backend_in_training_and_eval(self):
        previous = run.REPO / 'result/lorap_study/table1_gpf_refine3_3seeds_20261013_v1'
        filename = previous / 'source_snapshot/runs/lorap_study/table1_frozen/backends.py'
        spec = importlib.util.spec_from_file_location(
            'runs.lorap_study.table1_frozen._historical_backend', filename)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for framework in ('QAT', 'DQ'):
            with self.subTest(framework=framework):
                c = self.config(framework=framework)
                bundle = SyntheticGraphs(c)
                batch = next(bundle.batches('train', 41, 0, self.device))
                old = legacy.build(c, bundle, 41).to(self.device)
                new = backends.build(c, bundle, 41).to(self.device)
                self.assertEqual(run.digest(old), run.digest(new))
                for model in (old, new):
                    model.train()
                    torch.cuda.manual_seed(917)
                    out = model(batch)
                    out.square().mean().backward()
                    if model is old:
                        expected = out.detach().clone()
                    else:
                        self.assertTrue(torch.equal(expected, out))
                self.assertEqual(run.digest(old), run.digest(new))
                for a, b in zip(old.parameters(), new.parameters()):
                    if a.grad is not None:
                        self.assertTrue(torch.equal(a.grad, b.grad))
                old.eval(); new.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(old(batch), new(batch)))

    def test_graph_dropout_and_batchnorm_settings_take_effect(self):
        c = self.config(head_dropout=.1, batchnorm_momentum=.01)
        bundle = SyntheticGraphs(c)
        batch = next(bundle.batches('train', 41, 0, self.device))
        model = backends.build(c, bundle, 41).to(self.device).train()
        captured = []
        handle = model.norms[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        native_dropout = backends.F.dropout
        with patch.object(backends.F, 'dropout', wraps=native_dropout) as dropout:
            model(batch)
            self.assertEqual(dropout.call_count, 1)
            self.assertEqual(dropout.call_args.args[1], .1)
            self.assertTrue(dropout.call_args.kwargs['training'])
        handle.remove()
        torch.testing.assert_close(model.norms[0].running_mean,
                                   .01 * captured[0].mean(0), rtol=1e-5, atol=1e-7)
        model.eval()
        with torch.no_grad():
            torch.cuda.manual_seed(1); a = model(batch)
            torch.cuda.manual_seed(2); b = model(batch)
        self.assertTrue(torch.equal(a, b))
        restored = backends.build(c, bundle, 41).to(self.device).eval()
        backends.load_state(restored, model.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(a, restored(batch)))

    def test_adamw_uses_decoupled_decay_and_preserves_groups(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(1, 1, bias=False).to(self.device)
        model.prompts = torch.nn.Linear(1, 1, bias=False).to(self.device)
        c = self.config(lr=.1, weight_decay=.01)
        hp = dict(prompt_lr=.2, prompt_weight_decay=0., optimizer='adamw')
        opt = run.make_search_optimizer(model, c, hp)
        self.assertIsInstance(opt, torch.optim.AdamW)
        ids = [id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(2.); p.grad = torch.zeros_like(p)
        opt.step()
        self.assertAlmostEqual(model.backbone.weight.item(), 2. * (1. - .1 * .01), places=6)
        self.assertEqual(model.prompts.weight.item(), 2.)

    def test_adamw_graph_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run, 'Bundle', SyntheticGraphs), patch.object(report, 'generate'):
            helper = test_pipeline.PipelineTests()
            root = Path(directory)
            study = helper.fixture(root / 'resume', seeds=(41,), count=1)
            job = study['jobs'][0]
            c = self.config(head_dropout=.1, batchnorm_momentum=.01)
            job['config'] = c
            job['reference']['done']['initial_sha256'] = run.digest(backends.build(c, SyntheticGraphs(c), 41))
            job['hyperparameters'].update(optimizer='adamw', lr_scheduler=dict(name='cosine'))
            helper.register(root / 'resume', study)
            run.fit(root / 'resume', study, job, self.device, 1)
            run.fit(root / 'resume', study, job, self.device, 2)
            other = copy.deepcopy(study)
            (root / 'continuous' / job['directory']).mkdir(parents=True)
            helper.register(root / 'continuous', other)
            run.fit(root / 'continuous', other, other['jobs'][0], self.device, 2)
            a = torch.load(root / 'resume' / job['directory'] / 'last.pt', weights_only=False)
            b = torch.load(root / 'continuous' / job['directory'] / 'last.pt', weights_only=False)
            for key in ('model', 'optimizer', 'scheduler', 'rng', 'best'):
                helper.assert_tree_equal(a[key], b[key])

    def test_round_four_registration_and_initialization(self):
        previous = run.REPO / 'result/lorap_study/table1_gpf_refine3_3seeds_20261013_v1'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'round4'
            study = refine.prepare(root, previous, round_number=4)
            run.check_registration(study); run.verify_sources(root)
            self.assertEqual(study['cells'], refine4_space.CELLS)
            self.assertEqual(study['promoted_new'], 4)
            self.assertEqual(study['final_tests'], 3)
            self.assertIn('300', study['comparison_note'])
            reused = [j for j in study['jobs'] if j.get('reuse_from')]
            self.assertEqual(len(reused), 3)
            for job in reused:
                old = run.read(Path(job['reuse_from']['directory']) / 'config.json')['job']
                self.assertEqual(job['config'], old['config'])
                self.assertEqual(job['hyperparameters'], old['hyperparameters'])
                self.assertEqual(job['epochs'], old['epochs'])
            pending = run.tasks(root, study)
            self.assertEqual(len(pending), 24)
            self.assertTrue(all(cap == 80 and j['epochs'] == 300 and j['seed'] == 41 for j, cap in pending))
            incumbent = reused[0]
            options = study['candidates_by_cell'][study['cells'][0]]
            fingerprints = set()
            initial = run.digest(backends.build(incumbent['config'], SyntheticGraphs(incumbent['config']), 41))
            allowed = {'head_dropout', 'batchnorm_momentum', 'batch_size', 'lr', 'weight_decay'}
            for candidate in options:
                c = candidate['config']
                self.assertLessEqual({k for k in c if c[k] != incumbent['config'].get(k)}, allowed)
                self.assertEqual(run.digest(backends.build(c, SyntheticGraphs(c), 41)), initial)
                fingerprints.add(run.config_hash(dict(config=c, hp=candidate['hp'], epochs=candidate.get('epochs', 200))))
            self.assertEqual(len(fingerprints), 25)
            status = run.read(root / 'status.json')
            self.assertEqual(status['new_screens_expected'], 24)
            self.assertEqual(status['validation_fits_expected'], 15)
            self.assertEqual(status['reused_incumbent_fits'], 3)
            self.assertFalse(list(root.glob('*/candidate_*/seed_*/test.json')))


if __name__ == '__main__':
    unittest.main()

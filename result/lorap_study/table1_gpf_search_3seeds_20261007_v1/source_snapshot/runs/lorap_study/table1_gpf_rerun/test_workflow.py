"""Resume equivalence, source/control integrity, and a synthetic test gate."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from ..table1_frozen.backends import build, load_state, predict
from ..table1_frozen.common import cpu_tree, digest, write_json
from ..table1_frozen.registry import CELLS
from ..table1_frozen.test_final_flow import SyntheticBundle as BaseBundle
from . import run
from .model import GPFLoRAP, make_optimizer, objective


class SyntheticBundle(BaseBundle):
    def manifest(self):
        return dict(synthetic=True, nodes=6)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('GPU workflow tests require CUDA')
        cls.device = os.environ.get('TABLE1_DEVICE', 'cuda:0')
        torch.cuda.set_device(cls.device)
        torch.set_num_threads(4)

    def fixture(self, root, seeds=(41, 42, 43)):
        c = dict(CELLS['QAT_Cora_GIN'], cell='QAT_Cora_GIN', baseline_epochs=2)
        bundle = SyntheticBundle(c)
        jobs = []
        for seed in seeds:
            base = build(c, bundle, seed)
            job = dict(cell=c['cell'], seed=seed, config=c, epochs=2,
                       directory=f"{c['cell']}/seed_{seed}",
                       reference=dict(done=dict(initial_sha256=digest(base)), test=dict(metric=50.)))
            (root / job['directory']).mkdir(parents=True)
            jobs.append(job)
        study = dict(jobs=jobs, protocol='joint', hyperparameters=run.HP, subset_limit=None,
                     smoke=False, training_jobs=len(jobs), cells=[c['cell']], seeds=list(seeds))
        write_json(root / 'study.json', study)
        return study

    def test_paired_training_rng(self):
        from ..table1_graphlora import run as graph_run
        with tempfile.TemporaryDirectory() as temp:
            study = self.fixture(Path(temp), seeds=(41,))
            job = study['jobs'][0]
            b = SyntheticBundle(job['config'])
            graph, _, _ = graph_run.create_model(dict(study, hyperparameters=graph_run.HP), job, b, self.device)
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            graph.close()
            prompt, _, _ = run.create_model(study, job, b, self.device)
            self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
            self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
            prompt.close()

    def test_dq_prompt_covers_both_precisions(self):
        c = dict(CELLS['DQ_Cora_GCN'])
        b = SyntheticBundle(c)
        batch = next(b.batches('train', 41, 0, self.device))
        batch.prob_mask = torch.tensor([0, 1, 0, 1, 0, 1], device=self.device).float()
        model = GPFLoRAP(build(c, b, 41).to(self.device), c, 41, batch).train()
        prompt = model.prompts['aggregation'][0]
        seen = []
        native = prompt.add

        def capture(value):
            seen.append(value.detach())
            return native(value)

        with patch.object(prompt, 'add', capture):
            loss = objective(model, c, batch, run.HP)[0]
            loss.backward()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].size(0), batch.num_nodes)
        self.assertTrue(torch.isfinite(loss))
        model.close()

    def test_resume_and_once_only_final_test(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run, 'Bundle', SyntheticBundle):
            root = Path(temp)
            study = self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'All registered'):
                run.lock_tests(root, study)
            first = study['jobs'][0]
            native_write = run.write_json

            def stop_after_epoch(path, value):
                native_write(path, value)
                if Path(path).name == 'progress.json' and value['epoch'] == 1:
                    (root / 'STOP').touch()

            with patch.object(run, 'write_json', stop_after_epoch):
                run.fit(root, study, first, self.device)
            self.assertFalse((root / first['directory'] / 'done.json').exists())
            (root / 'STOP').unlink()
            run.fit(root, study, first, self.device)
            resumed = run.read(root / first['directory'] / 'done.json')
            continuous_root = root / 'continuous'
            continuous_root.mkdir()
            continuous_study = self.fixture(continuous_root, seeds=(41,))
            run.fit(continuous_root, continuous_study, continuous_study['jobs'][0], self.device)
            continuous = run.read(continuous_root / first['directory'] / 'done.json')
            self.assertEqual(resumed['state_sha256'], continuous['state_sha256'])
            self.assertEqual(resumed['validation'], continuous['validation'])
            for job in study['jobs'][1:]:
                run.fit(root, study, job, self.device)
            smoke = dict(study, smoke=True)
            with self.assertRaisesRegex(RuntimeError, 'Smoke'):
                run.lock_tests(root, smoke)
            run.finish(root, study, self.device)
            self.assertEqual(len(list(root.glob('*/seed_*/test.json'))), 3)
            self.assertTrue(run.read(root / 'completion_audit.json')['passed'])
            with patch.object(run, 'create_model', side_effect=AssertionError('Unexpected repeated inference')):
                run.finish(root, study, self.device)
            changed = copy.deepcopy(study)
            changed['hyperparameters']['rank'] = 3
            with self.assertRaisesRegex(ValueError, 'configuration'):
                run.lock_tests(root, changed)



if __name__ == "__main__":
    unittest.main()

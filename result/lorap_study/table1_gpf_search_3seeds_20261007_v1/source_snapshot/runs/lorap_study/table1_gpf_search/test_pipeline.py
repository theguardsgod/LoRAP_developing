"""Synthetic CUDA checks for search decisions, exact resume and final gates.

Only synthetic train/validation/test graphs are used; benchmark test data are
never opened by these tests.
"""
import copy
import fcntl
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..table1_frozen.backends import build
from ..table1_frozen.common import config_hash, digest, write_json
from ..table1_frozen.registry import CELLS
from ..table1_gpf_rerun.test_workflow import SyntheticBundle
from . import report, run, space


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Synthetic pipeline tests require CUDA')
        cls.device = os.environ.get('TABLE1_DEVICE', 'cuda:0')
        torch.cuda.set_device(cls.device)
        torch.set_num_threads(4)

    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(run, 'Bundle', SyntheticBundle).start()
        patch.object(report, 'generate', return_value=None).start()

    def fixture(self, root, *, seeds=(41, 42, 43), count=3, screen=1):
        root.mkdir(parents=True, exist_ok=True)
        original = dict(CELLS['QAT_Cora_GIN'], baseline_epochs=2)
        options = space.candidates()[:count]
        jobs = []
        for candidate in options:
            for seed in seeds:
                config, hp = space.instantiate(candidate, original)
                base = build(config, SyntheticBundle(config), seed)
                job = dict(cell=config['cell'], candidate=candidate['id'], seed=seed,
                    config=config, hyperparameters=hp, epochs=2, screen_epochs=screen,
                    directory=f"{config['cell']}/candidate_{candidate['id']}/seed_{seed}",
                    reference=dict(done=dict(initial_sha256=digest(base))))
                jobs.append(job)
                (root / job['directory']).mkdir(parents=True)
        study = dict(version=1, jobs=jobs, protocol='joint', subset_limit=None,
            smoke=False, cells=[original['cell']], seeds=list(seeds), search_seed=41,
            candidates=options, promote=min(2, count), final_tests=len(seeds))
        self.register(root, study)
        return study

    @staticmethod
    def register(root, study):
        study.pop('registration_sha256', None)
        study['registration_sha256'] = config_hash(study)
        write_json(root / 'study.json', study)

    def assert_tree_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(type(first), type(second))
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(first, second)

    def put_record(self, root, study, job, filename, loss):
        checkpoint = root / job['directory'] / ('screen_best.pt' if filename == 'screen.json' else 'best.pt')
        checkpoint.write_bytes(b'synthetic planner checkpoint')
        record = dict(config_hash=config_hash(run.fit_config(study, job)),
            validation=dict(loss=loss, metric=0.), selected_epoch=1,
            epochs=job['screen_epochs'] if filename == 'screen.json' else job['epochs'],
            checkpoint_file_sha256=run.sha(checkpoint), full_split=True,
            test_evaluations=0, state_sha256='synthetic planner only')
        write_json(root / job['directory'] / filename, record)

    def test_validation_promotion_ties_and_immutable_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.fixture(root)
            first = [j for j in study['jobs'] if j['seed'] == 41]
            self.assertEqual(len(run.tasks(root, study)), 3)
            for job, loss in zip(first, (.4, .2, .2)):
                self.put_record(root, study, job, 'screen.json', loss)
                # Unrelated test-like records must never influence selection.
                write_json(root / job['directory'] / 'test.json', dict(metric=1000 if job['candidate'] == 'c00' else -1000))
            pending = run.tasks(root, study)
            self.assertEqual([j['candidate'] for j, _ in pending], ['c01', 'c02'])
            self.assertTrue(all(cap == 2 for _, cap in pending))
            for job, _ in pending:
                self.put_record(root, study, job, 'done.json', .15)
            pending = run.tasks(root, study)
            self.assertEqual([(j['candidate'], j['seed']) for j, _ in pending], [('c01', 42), ('c01', 43)])
            self.assertEqual(run.read(root / study['cells'][0] / 'selection.json')['candidate'], 'c01')
            run.tasks(root, study)
            self.put_record(root, study, first[0], 'screen.json', .01)
            with self.assertRaisesRegex(ValueError, 'Locked validation decision'):
                run.tasks(root, study)
            self.put_record(root, study, first[0], 'screen.json', .4)
            self.put_record(root, study, first[2], 'done.json', .01)
            with self.assertRaisesRegex(ValueError, 'Locked validation decision'):
                run.tasks(root, study)

    def test_resume_matches_continuous_model_optimizer_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.fixture(root / 'resumed', seeds=(41,), count=1)
            resumed_root, job = root / 'resumed', study['jobs'][0]
            run.fit(resumed_root, study, job, self.device, until=1)
            screen_before = run.read(resumed_root / job['directory'] / 'screen.json')
            screen_sha = run.sha(resumed_root / job['directory'] / 'screen_best.pt')
            self.assertFalse((resumed_root / job['directory'] / 'done.json').exists())
            run.fit(resumed_root, study, job, self.device, until=2)
            other = self.fixture(root / 'continuous', seeds=(41,), count=1)
            run.fit(root / 'continuous', other, other['jobs'][0], self.device, until=2)
            resumed = torch.load(resumed_root / job['directory'] / 'last.pt', map_location='cpu', weights_only=False)
            continuous = torch.load(root / 'continuous' / job['directory'] / 'last.pt', map_location='cpu', weights_only=False)
            for key in ('config_hash', 'epoch', 'model', 'optimizer', 'scheduler', 'rng', 'best'):
                self.assert_tree_equal(resumed[key], continuous[key])
            strip_seconds = lambda rows: [{k: v for k, v in row.items() if k != 'seconds'} for row in rows]
            self.assert_tree_equal(strip_seconds(resumed['history']), strip_seconds(continuous['history']))
            self.assertEqual(screen_before, run.read(resumed_root / job['directory'] / 'screen.json'))
            self.assertEqual(screen_sha, run.sha(resumed_root / job['directory'] / 'screen_best.pt'))

    def test_resume_repairs_missing_screen_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.fixture(root, seeds=(41,), count=1)
            job = study['jobs'][0]
            run.fit(root, study, job, self.device, until=1)
            path = root / job['directory']
            expected = run.read(path / 'screen.json')
            # Model the interruption between durable last.pt and screen.json.
            (path / 'screen.json').unlink()
            (path / 'screen_best.pt').unlink()
            run.fit(root, study, job, self.device, until=1)
            self.assertEqual(expected, run.read(path / 'screen.json'))
            self.assertEqual(expected['checkpoint_file_sha256'], run.sha(path / 'screen_best.pt'))

    def test_worker_repairs_node_full_screen_without_done(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.fixture(root, count=1, screen=2)
            study['smoke'] = True
            self.register(root, study)
            job = study['jobs'][0]
            native_write = run.write_json

            def interrupt_after_screen(path, value):
                native_write(path, value)
                if Path(path).name == 'screen.json':
                    raise RuntimeError('Synthetic interruption after screen')

            with patch.object(run, 'write_json', interrupt_after_screen):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic interruption'):
                    run.fit(root, study, job, self.device, until=2)
            self.assertTrue((root / job['directory'] / 'screen.json').exists())
            self.assertFalse((root / job['directory'] / 'done.json').exists())
            write_json(root / 'source_manifest.json', {})
            with patch.object(run, '__file__', str(root / 'source_snapshot' / 'synthetic_runner.py')), \
                 patch.object(run.time, 'sleep', side_effect=AssertionError('Stalled completion marker')):
                run.worker(root, self.device)
            self.assertTrue((root / job['directory'] / 'done.json').exists())
            audit = run.read(root / 'completion_audit.json')
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['tests'], 0)
            self.assertEqual(len(list(root.glob('*/candidate_*/seed_*/done.json'))), 3)

    def test_numerical_rejections_keep_confirmation_seeds_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.fixture(root)
            first = [j for j in study['jobs'] if j['seed'] == 41]
            write_json(root / first[0]['directory'] / 'failure.json', dict(
                config_hash=config_hash(run.fit_config(study, first[0])), numerical_rejection=True))
            for job in first[1:]:
                self.put_record(root, study, job, 'screen.json', .2)
            pending = run.tasks(root, study)
            self.assertEqual([j['candidate'] for j, _ in pending], ['c01', 'c02'])
            write_json(root / first[1]['directory'] / 'failure.json', dict(
                config_hash=config_hash(run.fit_config(study, first[1])), numerical_rejection=True))
            self.put_record(root, study, first[2], 'done.json', .15)
            pending = run.tasks(root, study)
            self.assertEqual([(j['candidate'], j['seed']) for j, _ in pending], [('c02', 42), ('c02', 43)])
            write_json(root / pending[0][0]['directory'] / 'failure.json', dict(numerical_rejection=False))
            self.assertEqual(len(run.tasks(root, study)), 2)
            with self.assertRaisesRegex(RuntimeError, 'three-seed fits'):
                run.lock_tests(root, study)

    def test_three_seed_gate_once_only_and_lock_contention(self):
        from ..table1_frozen import finish as final_module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'selections'):
                run.lock_tests(root, study)
            # Execute only the planner's requested phases, never rejected seeds.
            while True:
                pending = run.tasks(root, study)
                if not pending:
                    break
                if (root / 'SELECTION_LOCK.json').exists():
                    with self.assertRaisesRegex(RuntimeError, 'three-seed fits'):
                        run.lock_tests(root, study)
                job, cap = pending[0]
                run.fit(root, study, job, self.device, until=cap)
            smoke = dict(study, smoke=True)
            with self.assertRaisesRegex(RuntimeError, 'Smoke'):
                run.lock_tests(root, smoke)
            with open(root / '.test.lock', 'a') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(run, 'lock_tests', side_effect=AssertionError('Concurrent test gate entered')):
                    run.finish(root, study, self.device)
            run.finish(root, study, self.device)
            tests = list(root.glob('*/candidate_*/seed_*/test.json'))
            self.assertEqual(len(tests), 3)
            self.assertTrue(run.read(root / 'completion_audit.json')['passed'])
            self.assertTrue(all(run.read(p)['completed_test_evaluations'] == 1 for p in tests))
            with patch.object(run, 'create_model', side_effect=AssertionError('Repeated model creation')), \
                 patch.object(final_module, 'predict', side_effect=AssertionError('Repeated test inference')):
                run.finish(root, study, self.device)
            broken = run.read(tests[0])
            broken['completed_test_evaluations'] = 2
            write_json(tests[0], broken)
            with self.assertRaises(ValueError):
                run.finish(root, study, self.device)

    def test_registration_saved_config_source_and_unknown_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.fixture(root)
            run.check_registration(study)
            mutated = copy.deepcopy(study)
            mutated['jobs'][0]['hyperparameters']['rank'] = 7
            with self.assertRaisesRegex(ValueError, 'Registered search settings'):
                run.check_registration(mutated)
            job = study['jobs'][0]
            self.put_record(root, study, job, 'screen.json', .2)
            mutated_job = copy.deepcopy(job)
            mutated_job['hyperparameters']['prompt_lr'] *= 2
            with self.assertRaisesRegex(ValueError, 'configuration'):
                run.checked_record(root, study, mutated_job, 'screen.json')
            archived = root / 'source_snapshot' / 'synthetic.py'
            archived.parent.mkdir()
            archived.write_text('a = 1\n')
            write_json(root / 'source_manifest.json', {'synthetic.py': run.sha(archived)})
            run.verify_sources(root)
            archived.write_text('a = 2\n')
            with self.assertRaisesRegex(ValueError, 'Archived source changed'):
                run.verify_sources(root)
            cell = study['cells'][0]
            selected = dict(candidate='not_registered', validation_loss=0., ranking=[], criterion='validation_loss', seed=41)
            write_json(root / cell / 'selection.json', selected)
            cells = {cell: selected}
            write_json(root / 'SELECTION_LOCK.json', dict(cells=cells, sha256=config_hash(cells)))
            with self.assertRaises(ValueError):
                run.lock_tests(root, study)


if __name__ == '__main__':
    unittest.main()

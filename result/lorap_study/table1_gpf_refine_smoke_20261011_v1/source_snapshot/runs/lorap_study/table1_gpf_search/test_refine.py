"""Mean-seed selection, cached test reuse, and new scheduler resume checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..table1_frozen.common import write_json
from ..table1_gpf_rerun.test_workflow import SyntheticBundle
from . import run, report, refine_space
from . import test_pipeline


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        cls.device = 'cuda:0'

    def fixture(self, root):
        helper = test_pipeline.PipelineTests()
        study = helper.fixture(root)
        study.update(selection_protocol='mean_three', selection_seeds=[41, 42, 43],
                     protected_candidate='c00', promoted_new=1, promote=2)
        helper.register(root, study)
        return study, helper

    def test_selection_uses_all_three_seeds_and_retains_incumbent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, helper = self.fixture(root)
            for job in study['jobs']:
                if job['seed'] == 41:
                    helper.put_record(root, study, job, 'screen.json', {'c00': .2, 'c01': .05, 'c02': .3}[job['candidate']])
            pending = run.tasks(root, study)
            self.assertEqual(len(pending), 6)
            self.assertEqual({j['candidate'] for j, _ in pending}, {'c00', 'c01'})
            for job, _ in pending:
                if job['candidate'] == 'c01' and job['seed'] == 43:
                    continue
                loss = .2 if job['candidate'] == 'c00' else (.05 if job['seed'] == 41 else .5)
                helper.put_record(root, study, job, 'done.json', loss)
            self.assertEqual(len(run.tasks(root, study)), 1)
            self.assertFalse((root / 'SELECTION_LOCK.json').exists())
            remaining, _ = run.tasks(root, study)[0]
            helper.put_record(root, study, remaining, 'done.json', .5)
            self.assertEqual(run.tasks(root, study), [])
            selected = run.read(root / study['cells'][0] / 'selection.json')
            self.assertEqual(selected['candidate'], 'c00')
            self.assertAlmostEqual(selected['validation_loss'], .2)
            self.assertEqual(len(selected['per_seed_validation']), 3)

    def test_cosine_and_plateau_resume(self):
        for spec in (dict(name='cosine'), dict(name='plateau', patience=1, factor=.5)):
            with self.subTest(spec=spec), tempfile.TemporaryDirectory() as directory, \
                 patch.object(run, 'Bundle', SyntheticBundle), patch.object(report, 'generate'):
                root = Path(directory)
                helper = test_pipeline.PipelineTests()
                study = helper.fixture(root / 'resume', seeds=(41,), count=1)
                study['jobs'][0]['hyperparameters']['lr_scheduler'] = spec
                helper.register(root / 'resume', study)
                job = study['jobs'][0]
                run.fit(root / 'resume', study, job, self.device, 1)
                run.fit(root / 'resume', study, job, self.device, 2)
                other = copy.deepcopy(study)
                other_root = root / 'continuous'
                (other_root / job['directory']).mkdir(parents=True)
                helper.register(other_root, other)
                run.fit(other_root, other, other['jobs'][0], self.device, 2)
                a = torch.load(root / 'resume' / job['directory'] / 'last.pt', weights_only=False)
                b = torch.load(other_root / job['directory'] / 'last.pt', weights_only=False)
                for key in ('model', 'optimizer', 'scheduler', 'rng', 'best'):
                    helper.assert_tree_equal(a[key], b[key])

    def test_cached_incumbent_test_is_not_executed_again(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(report, 'generate'):
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            old = dict(cell='x', seed=41, metric=75., completed_test_evaluations=1,
                       checkpoint_file_sha256='checkpoint', test_lock_sha256='old-lock')
            write_json(source / 'test.json', old)
            job = dict(directory='cached', reuse_from=dict(directory=str(source),
                       test_file_sha256=run.sha(source / 'test.json')))
            (root / job['directory']).mkdir()
            case = dict(directory='cached', method='gpf_lorap_tuned', checkpoint_file_sha256='checkpoint')
            manifest = dict(cases=[case], sha256='new-lock')
            with patch.object(run, 'lock_tests', return_value=([job], manifest)), \
                 patch.object(run, 'create_model', side_effect=AssertionError('Cached model inference repeated')):
                run.finish(root, dict(seeds=[41], cells=['x']), self.device)
                run.finish(root, dict(seeds=[41], cells=['x']), self.device)
            result = run.read(root / 'cached/test.json')
            self.assertEqual(result['metric'], old['metric'])
            self.assertTrue(result['reused_previous_test'])
            self.assertEqual(result['original_test_lock_sha256'], 'old-lock')
            audit = run.read(root / 'completion_audit.json')
            self.assertEqual(audit['new_test_evaluations'], 0)
            self.assertEqual(audit['reused_test_results'], 1)

    def test_space_preserves_architecture_and_quantization_budget(self):
        import json
        previous = Path(__file__).resolve().parents[3] / 'result/lorap_study/table1_gpf_search_3seeds_20261007_v1'
        study = json.loads((previous / 'study.json').read_text())
        selections = json.loads((previous / 'SELECTION_LOCK.json').read_text())['cells']
        allowed = {'lr', 'weight_decay', 'lr_feature', 'lr_message', 'lr_weight', 'lr_bit', 'lr_attention', 'lr_edge'}
        for cell in refine_space.CELLS:
            old = next(j for j in study['jobs'] if j['cell'] == cell and j['seed'] == 41 and j['candidate'] == selections[cell]['candidate'])
            options = refine_space.candidates(cell, old['config'], old['hyperparameters'])
            self.assertEqual(len(options), 13)
            self.assertEqual(options[0]['config'], old['config'])
            for candidate in options:
                self.assertLessEqual({k for k in old['config'] if old['config'][k] != candidate['config'][k]}, allowed)


if __name__ == '__main__':
    unittest.main()

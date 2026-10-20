"""GPU checkpoint resume, both-mode release barrier, validation-only pilot."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import graphlora_run as runner
from .run import write_json
from .test_lora import fixture


@unittest.skipUnless(torch.cuda.is_available(), 'Host CUDA required')
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_epoch_boundary_stop_and_resume(self):
        config = runner.study_config('a2q', [41], 3, 'pilot')
        data, device = fixture('MNIST'), torch.device('cuda:0')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = runner.fit(root/'full/a2q', config, 'graphlora', 41, data, None, device)
            original = runner.train_graph_epoch
            def stop_after_epoch(*args, **kwargs):
                result = original(*args, **kwargs)
                (root/'resumed/STOP').touch()
                return result
            with patch.object(runner, 'train_graph_epoch', side_effect=stop_after_epoch):
                with self.assertRaises(runner.StopRequested):
                    runner.fit(root/'resumed/a2q', config, 'graphlora', 41, data, None, device)
            directory = root/'resumed/a2q/MNIST/graphlora/seed_41'
            self.assertEqual(torch.load(directory/'last.pt', weights_only=False)['epoch'], 1)
            (root/'resumed/STOP').unlink()
            actual = runner.fit(root/'resumed/a2q', config, 'graphlora', 41, data, None, device)
            self.assertEqual(expected['best_epoch'], actual['best_epoch'])
            self.assertAlmostEqual(expected['validation']['loss'], actual['validation']['loss'], places=5)
            a = torch.load(root/'full/a2q/MNIST/graphlora/seed_41/last.pt', weights_only=False)
            b = torch.load(directory/'last.pt', weights_only=False)
            for name, tensor in a['model'].items():
                torch.testing.assert_close(tensor, b['model'][name], rtol=1e-4, atol=1e-5)
            for name, value in a['rng']['loaders'].items():self.assertTrue(torch.equal(value, b['rng']['loaders'][name]))
            self.assertEqual(actual['test_evaluations'], 0)

    def test_full_worker_tests_only_after_all_paired_fits_and_only_once(self):
        data = fixture('MNIST')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = [runner.study_config(m, [41], 1, 'full') for m in ('fixed4', 'a2q')]
            write_json(root/'study.json', dict(configs=configs, profile='full', device='cuda:0', threads=2))
            events = []; original = runner.evaluate
            def observed(*args, **kwargs):
                split = kwargs.get('split', args[4] if len(args)>4 else None)
                if split == 'test':
                    self.assertEqual(len(list(root.glob('*/MNIST/*/seed_41/result.json'))), 4)
                    self.assertTrue((root/'TEST_RELEASE.json').exists())
                events.append(split)
                return original(*args, **kwargs)
            with patch.object(runner, 'study_data', return_value=(data, {'fixture':True})), \
                 patch.object(runner, 'evaluate', side_effect=observed):
                runner.worker(root)
            self.assertEqual(events.count('test'), 4)
            self.assertEqual(events[-4:], ['test']*4)
            self.assertTrue(runner.read(root/'completion_audit.json')['passed'])
            with patch.object(runner, 'study_data', return_value=(data, {'fixture':True})), \
                 patch.object(runner, 'evaluate', side_effect=AssertionError('Repeated evaluation')):
                runner.worker(root)

    def test_pilot_has_no_test_release(self):
        from .graphlora_report import generate_report
        data = fixture('MNIST')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = [runner.study_config('fixed4', [41], 1, 'pilot')]
            write_json(root/'study.json', dict(configs=configs, profile='pilot', device='cuda:0', threads=2))
            with patch.object(runner, 'study_data', return_value=(data, {'fixture':True})), \
                 patch.object(runner, 'test_once', side_effect=AssertionError('Pilot test leakage')):
                runner.worker(root)
            self.assertFalse((root/'TEST_RELEASE.json').exists())
            generate_report(root)
            status = runner.read(root/'status.json')
            self.assertTrue(status['complete']); self.assertEqual(status['tests_completed'], 0)
            self.assertIn('TBD', (root/'table_v_graphlora_rows.tex').read_text())
            self.assertEqual(runner.read(root/'fixed4/pairing_seed_41.json')['passed'], True)


if __name__=='__main__':unittest.main()

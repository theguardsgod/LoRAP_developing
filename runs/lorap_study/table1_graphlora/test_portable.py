"""Portable registration, paired initialization, resume, and test-barrier tests."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

from ..table1_frozen.test_final_flow import SyntheticBundle
from ..table1_frozen import finish as final_evaluation
from ..table1_frozen.common import write_json
from . import model as gm
from . import portable as p
from ..test_lora import fixture


class Bundle(SyntheticBundle):
    def manifest(self):
        return dict(synthetic=True, split='fixed-six-nodes')


class GraphBundle:
    def __init__(self, config):
        self.train = fixture('MNIST')['train']
        self.features, self.classes = 3, 2

    def batches(self, split, seed, epoch, device, limit=None, allow_test=False):
        if split == 'test' and not allow_test:
            raise RuntimeError('Synthetic test gate is locked')
        yield Batch.from_data_list(self.train).to(device)

    def manifest(self):
        return dict(synthetic=True, split='fixed-two-graphs')


def cpu_predict(model, config, batch):
    out = model(batch)
    return out, out.new_zeros(())


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def register(self, root, smoke=False):
        with patch.object(p, 'snapshot'):
            return p.prepare(root, root / 'datasets', ['QAT_Cora_GCN'], [41], smoke)

    def test_self_contained_plan_and_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'study'
            study = self.register(root)
            self.assertEqual(len(study['jobs']), 2)
            self.assertFalse(study['controls_reused'])
            self.assertEqual({j['variant'] for j in study['jobs']}, {'none', 'graphlora'})
            self.assertNotIn('reference', study['jobs'][0])
            self.assertEqual(study['jobs'][0]['config']['data_root'], str(root / 'datasets/cora'))
            with self.assertRaisesRegex(RuntimeError, 'All paired'):
                p.test_manifest(root, study)
            smoke_root = Path(temp) / 'smoke'
            smoke = self.register(smoke_root, smoke=True)
            with self.assertRaisesRegex(RuntimeError, 'Smoke'):
                p.test_manifest(smoke_root, smoke)

    def test_cpu_rng_roundtrip(self):
        from ..table1_frozen.common import seed_all, rng_state, restore_rng
        seed_all(41)
        state = rng_state('cpu')
        expected = torch.rand(8)
        restore_rng(state, 'cpu')
        torch.testing.assert_close(torch.rand(8), expected, rtol=0, atol=0)
        self.assertIsNone(state['cuda'])

    def test_focal_mnist_backends_train_paired_on_synthetic_graphs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(p, 'Bundle', GraphBundle), \
             patch.object(p, 'predict', cpu_predict), patch.object(gm, 'predict', cpu_predict), \
             patch.object(p, 'snapshot'):
            root = Path(temp) / 'graphs'
            study = p.prepare(root, root / 'datasets', ['QAT_MNIST_GCN', 'QAT_MNIST_GAT'], [41], smoke=True)
            for job in study['jobs']:
                p.fit(root, study, job, 'cpu')
            for cell in study['cells']:
                done = [p.read(root / j['directory'] / 'done.json') for j in study['jobs'] if j['cell'] == cell]
                self.assertEqual(done[0]['initial_backbone_sha256'], done[1]['initial_backbone_sha256'])
            self.assertFalse((root / 'TEST_LOCK.json').exists())

    def test_resume_pairs_and_once_only_test(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(p, 'Bundle', Bundle), \
             patch.object(p, 'predict', cpu_predict), patch.object(gm, 'predict', cpu_predict), \
             patch.object(final_evaluation, 'predict', cpu_predict):
            root = Path(temp) / 'study'
            study = self.register(root)
            for job in study['jobs']:
                job['epochs'] = 3
            write_json(root / 'study.json', study)
            none_job, adapter_job = study['jobs']
            p.fit(root, study, none_job, 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'All paired'):
                p.test_manifest(root, study)
            original_write = p.write_json

            def pause(path, value):
                original_write(path, value)
                if Path(path).name == 'progress.json' and value['epoch'] == 1:
                    (root / 'STOP').touch()

            with patch.object(p, 'write_json', side_effect=pause):
                p.fit(root, study, adapter_job, 'cpu')
            self.assertFalse((root / adapter_job['directory'] / 'done.json').exists())
            (root / 'STOP').unlink()
            p.fit(root, study, adapter_job, 'cpu')
            done = p.read(root / adapter_job['directory'] / 'done.json')
            other = Path(temp) / 'continuous'
            continuous = self.register(other)
            continuous['jobs'][1]['epochs'] = 3
            p.fit(other, continuous, continuous['jobs'][1], 'cpu')
            expected = p.read(other / continuous['jobs'][1]['directory'] / 'done.json')
            self.assertEqual(done['state_sha256'], expected['state_sha256'])
            p.test_manifest(root, study)
            p.finish(root, study, 'cpu')
            self.assertEqual(len(list(root.glob('*/seed_*/*/test.json'))), 2)
            with patch.object(p, 'model_for', side_effect=AssertionError('Repeated test inference')):
                p.finish(root, study, 'cpu')
            self.assertTrue(p.read(root / 'completion_audit.json')['passed'])
            p.generate(root, study)
            self.assertEqual(len(p.read(root / 'summary.json')), 2)
            changed = copy.deepcopy(study)
            changed['hyperparameters']['normalization_floor'] = .2
            with self.assertRaisesRegex(ValueError, 'registration'):
                p.test_manifest(root, changed)
            donepath = root / adapter_job['directory'] / 'done.json'
            corrupted = p.read(donepath)
            corrupted['initial_backbone_sha256'] = 'mismatched'
            write_json(donepath, corrupted)
            with self.assertRaisesRegex(ValueError, 'initialization'):
                p.test_manifest(root, study)


if __name__ == '__main__':
    unittest.main()

"""PubMed recipe and paired A2Q/GPF-LoRAP workflow checks."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from . import pubmed_gpf_lorap as pubmed
from .table1_graphlora import portable as engine
from .table1_gpf_rerun import model as gpf
from .graphlora import checked_gradient_norm
from .table1_frozen.data import Bundle


class SyntheticPubMed:
    def __init__(self, config):
        self.features, self.classes = 5, 3
        ids = torch.arange(12)
        self.data = Data(x=torch.arange(60, dtype=torch.float32).reshape(12, 5) / 60,
            edge_index=torch.cat([torch.stack([ids, (ids + 1) % 12]), torch.stack([(ids + 1) % 12, ids])], dim=1),
            y=ids % 3, train_mask=ids < 6, val_mask=(ids >= 6) & (ids < 9), test_mask=ids >= 9)

    def batches(self, split, seed, epoch, device, limit=None, allow_test=False):
        if split == 'test' and not allow_test:
            raise RuntimeError('Synthetic test split is locked')
        yield self.data.clone().to(device)

    def manifest(self):
        return dict(synthetic=True, nodes=12, classes=3, split='fixed')


def cpu_predict(model, config, batch):
    out = model(batch)
    if isinstance(out, tuple):
        return out[0], out[1].squeeze()
    budget = sum(config['width'] * (config['heads'] if 'q_gat_fea' in n else 1) * p.abs().sum() / (8 * 1024)
                 for n, p in model.named_parameters() if n.endswith('.bit') and ('fea_quant' in n or 'q_gat_fea' in n))
    return out, budget


class PubMedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_requested_pubmed_cells_and_recipe(self):
        configs = pubmed.configurations()
        self.assertEqual(set(configs), {'A2Q_PubMed_GIN', 'A2Q_PubMed_GCN', 'A2Q_PubMed_GAT'})
        for c in configs.values():
            self.assertEqual((c['dataset'], c['framework']), ('PubMed', 'A2Q'))
            self.assertIsNone(c['paper_gpf_lorap'])
        self.assertEqual(configs['A2Q_PubMed_GAT']['attention_dropout'], .3)
        with tempfile.TemporaryDirectory() as temp, patch.object(engine, 'snapshot'):
            study = pubmed.prepare(Path(temp) / 'study', Path(temp) / 'data')
            self.assertEqual(len(study['jobs']), 18)
            self.assertEqual(study['variants'], ['none', 'gpf_lorap'])
            self.assertEqual(study['seeds'], [41, 42, 43])
            self.assertEqual(study['protocol'], 'joint')

    def test_all_three_a2q_backends_train_and_pair(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(engine, 'snapshot'), \
             patch.object(engine, 'Bundle', SyntheticPubMed), patch.object(engine, 'predict', cpu_predict), \
             patch.object(gpf, 'predict', cpu_predict):
            root = Path(temp) / 'smoke'
            study = pubmed.prepare(root, root / 'data', [41], smoke=True)
            for job in study['jobs']:
                engine.fit(root, study, job, 'cpu')
            for cell in study['cells']:
                done = [engine.read(root / job['directory'] / 'done.json') for job in study['jobs'] if job['cell'] == cell]
                self.assertEqual(done[0]['initial_backbone_sha256'], done[1]['initial_backbone_sha256'])
                self.assertGreater(done[1]['prompt_parameters'], 0)
            engine.generate(root, study)
            rows = pubmed.report(root)
            self.assertTrue(all(r['table_value'] == 'TBD' for r in rows))
            self.assertFalse((root / 'TEST_LOCK.json').exists())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA allocation required')
class PubMedCudaTests(unittest.TestCase):
    def test_three_real_backends_have_live_prompt_and_backbone_gradients(self):
        torch.set_num_threads(4)
        device = 'cuda:0'
        root = Path(os.environ.get('PUBMED_DATA_ROOT', 'data')).resolve()
        with tempfile.TemporaryDirectory() as temp, patch.object(engine, 'snapshot'):
            study = pubmed.prepare(Path(temp) / 'study', root, [41], smoke=True)
            for job in [j for j in study['jobs'] if j['variant'] == 'gpf_lorap']:
                bundle = Bundle(job['config'])
                model, _ = engine.model_for(study, job, bundle, device)
                try:
                    model.train()
                    batch = next(bundle.batches('train', 41, 0, device))
                    loss = gpf.objective(model, job['config'], batch, study['hyperparameters'])[0]
                    loss.backward()
                    checked_gradient_norm(model, study['hyperparameters']['gradient_norm_limit'])
                    for group in ('input', 'aggregation'):
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                            for p in model.prompts[group].parameters()))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in model.backbone.parameters() if p.requires_grad))
                    print('PubMed GPU backend passed:', job['cell'], flush=True)
                finally:
                    model.close()
                del model, batch, bundle
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()

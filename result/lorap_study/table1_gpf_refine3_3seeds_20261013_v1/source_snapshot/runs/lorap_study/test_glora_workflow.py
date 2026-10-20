"""Exercise real GPU training resume, compact audit, and the test-release barrier."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import glora_run as runner
from .glora import backbone_digest, build_glora, glora_optimizer
from .lora import baseline_config
from .run import write_json
from .test_lora import fixture


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class WorkflowTests(unittest.TestCase):
    def test_resume_export_audit_and_once_only_test(self):
        torch.set_num_threads(2)
        config=dict(baseline_config('MNIST'),quantization_mode='a2q',epochs=3,seeds=[41],glora_rank=2,
            search_population=4,search_generations=2,search_elite=2,search_mutation=.2,max_adapter_parameters=None)
        data=fixture('MNIST');device=torch.device('cuda:0')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            full,resumed=root/'full',root/'resumed'
            model=build_glora(config,41,data)
            optimizer=glora_optimizer(model,config)
            groups=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay'],
                parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups if g['name']!='glora']
            for output in (full,resumed):
                reference=output/'reference/a2q/seed_41'
                write_json(reference/'initialization.json',dict(common_backbone_sha256=backbone_digest(model)))
                write_json(reference/'optimizer_groups.json',groups)
            expected=runner.fit_supernet(full,config,41,data,device)
            original=runner.train_gpu_epoch;calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('interrupted after committed epoch')
                return original(*args,**kwargs)
            with patch.object(runner,'train_gpu_epoch',side_effect=interrupted):
                with self.assertRaises(RuntimeError):runner.fit_supernet(resumed,config,41,data,device)
            actual=runner.fit_supernet(resumed,config,41,data,device)
            self.assertEqual(actual['best_epoch'],expected['best_epoch'])
            self.assertEqual(actual['reference_validation_loss'],expected['reference_validation_loss'])
            full_last=torch.load(runner.directory_for(full,config,41)/'last.pt',weights_only=False)
            resumed_last=torch.load(runner.directory_for(resumed,config,41)/'last.pt',weights_only=False)
            self.assertEqual(full_last['glora_rng'],resumed_last['glora_rng'])
            for name,value in full_last['model'].items():
                torch.testing.assert_close(value,resumed_last['model'][name],rtol=0,atol=0)
            result=runner.search_and_export(resumed,config,41,data,device)
            self.assertEqual(result['test_evaluations'],0)
            self.assertEqual(result['search_evaluations'],8)
            directory=runner.directory_for(resumed,config,41)
            runner.verify_compact(directory,config,41,data)
            with self.assertRaises(FileNotFoundError):runner.test_once(resumed,config,41,data,device)
            with self.assertRaises(FileNotFoundError):
                runner.release_tests(resumed,[config,dict(config,quantization_mode='fixed4',seeds=[42])],data)
            self.assertFalse((resumed/'test_release.json').exists())
            runner.release_tests(resumed,[config],data)
            tested=runner.test_once(resumed,config,41,data,device)
            self.assertEqual(tested['test_evaluations'],1)
            with patch.object(runner,'evaluate',side_effect=AssertionError('Repeated test')):
                self.assertEqual(runner.test_once(resumed,config,41,data,device),tested)
            saved=torch.load(directory/'compact.pt',weights_only=False)
            saved['model']['embedding.weight'].add_(.1)
            torch.save(saved,directory/'compact.pt')
            with self.assertRaises(AssertionError):runner.verify_compact(directory,config,41,data)


if __name__=='__main__':unittest.main()

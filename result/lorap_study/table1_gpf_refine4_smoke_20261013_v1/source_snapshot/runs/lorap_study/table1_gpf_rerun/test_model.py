"""GPU checks on every selected backend; only training data are used."""
import unittest

import torch
from torch.nn import functional as F

from .model import GPFLoRAP, counts, make_optimizer, objective
from .run import HP
from ..table1_frozen.backends import build, load_state, predict
from ..table1_frozen.common import cpu_tree, digest, seed_all
from ..table1_frozen.data import Bundle
from ..table1_frozen.registry import CELLS
from ..table1_frozen.vendor.quantization import IntegerQuantizer

TARGETS = ['QAT_CiteSeer_GIN', 'QAT_MNIST_GIN', 'DQ_Cora_GIN', 'DQ_MNIST_GIN',
           'QAT_CIFAR10_GCN', 'DQ_ZINC_GCN', 'DQ_CiteSeer_GAT', 'DQ_MNIST_GAT',
           'DQ_CIFAR10_GAT', 'A2Q_ZINC_GAT']


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Real-data model tests require CUDA')
        cls.device = 'cuda:0'
        torch.cuda.set_device(cls.device)
        torch.set_num_threads(4)

    def test_all_target_backends(self):
        for name in TARGETS:
            with self.subTest(cell=name):
                print('checking', name, flush=True)
                c = dict(CELLS[name], batch_size=2)
                b = Bundle(c)
                seed_all(41)
                base = build(c, b, 41).to(self.device)
                batch = next(b.batches('train', 41, 0, self.device, limit=2))
                with torch.no_grad():
                    base.train()
                    predict(base, c, batch)
                base.eval()
                calls, handles = {}, []
                for key, module in base.named_modules():
                    if isinstance(module, IntegerQuantizer) or hasattr(module, 'quant_fea_func') or hasattr(module, 'quant_weight_func'):
                        def called(m, a, o, key=key):
                            calls[key] = calls.get(key, 0) + 1
                        handles.append(module.register_forward_hook(called))
                with torch.no_grad():
                    reference, budget = predict(base, c, batch)
                expected_calls = dict(calls)
                original = digest(base)
                cpu_rng = torch.get_rng_state().clone()
                gpu_rng = torch.cuda.get_rng_state().clone()
                model = GPFLoRAP(base, c, 41, batch)
                self.assertEqual(digest(base), original)
                self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
                initial_prompts = cpu_tree(model.prompts.state_dict())
                with torch.no_grad():
                    model.prompts['input'].p_list.zero_()
                    for prompt in model.prompts['aggregation']:
                        prompt.P_left.zero_()
                calls.clear()
                model.eval()
                with torch.no_grad():
                    actual, actual_budget = predict(model, c, batch)
                self.assertTrue(torch.equal(reference, actual), name)
                self.assertTrue(torch.equal(budget, actual_budget), name)
                self.assertEqual(calls, expected_calls)
                model.prompts.load_state_dict(initial_prompts)
                self.assertGreater(counts(model)['prompt_parameters'], 0)
                opt = make_optimizer(model, c)
                before = cpu_tree(base.state_dict())
                model.train()
                loss, _, _, _ = objective(model, c, batch, HP)
                loss.backward()
                for group in ('input', 'aggregation'):
                    grads = [p.grad for p in model.prompts[group].parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertTrue(any(g.abs().sum() > 0 for g in grads), (name, group))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in base.parameters()))
                opt.step()
                self.assertTrue(any(not torch.equal(before[n], p.detach().cpu()) for n, p in base.named_parameters()))
                model.eval()
                saved = cpu_tree(model.state_dict())
                with torch.no_grad():
                    before_pred = predict(model, c, batch)[0]
                for handle in handles:
                    handle.remove()
                native = build(c, b, 41).to(self.device)
                restored = GPFLoRAP(native, c, 41, batch)
                load_state(restored, saved)
                restored.eval()
                anonymous = batch.clone()
                anonymous.y = None
                with torch.no_grad():
                    after = predict(restored, c, anonymous)[0]
                self.assertTrue(torch.equal(before_pred, after), name)
                model.close()
                restored.close()
                del model, restored, native, base, b, batch, saved
                torch.cuda.empty_cache()

    def test_original_prompt_formula(self):
        from prompt.GPF import GPF_plus_LoRA
        p = GPF_plus_LoRA(6, 20, 2).cuda()
        x = torch.randn(7, 6, device=self.device)
        expected = x + F.softmax(F.linear(x, p.a.weight, p.a.bias), dim=-1) @ (p.P_left @ p.P_right)
        self.assertTrue(torch.equal(p.add(x), expected))
        self.assertTrue(torch.allclose(p.add(x)[:3], p.add(x[:3]), atol=1e-6))


if __name__ == '__main__':
    unittest.main()

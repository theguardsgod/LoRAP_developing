"""Real-data GPU compatibility checks; no benchmark test set is read."""
import os
import unittest

import torch
from ..table1_frozen.backends import build, load_state, predict
from ..table1_frozen.common import cpu_tree, digest, seed_all
from ..table1_frozen.data import Bundle
from ..table1_frozen.registry import CELLS
from ..table1_frozen.vendor.quantization import IntegerQuantizer
from .model import GraphLoRA, counts, make_optimizer, objective, Branch
from .run import HP, DEFAULT_SEEDS


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('GPU tests require CUDA')
        cls.device = os.environ.get('TABLE1_DEVICE', 'cuda:0')
        torch.cuda.set_device(cls.device)
        torch.set_num_threads(4)

    def test_default_three_seeds(self):
        self.assertEqual(DEFAULT_SEEDS, [41, 42, 43])
        self.assertEqual(len(CELLS) * len(DEFAULT_SEEDS), 135)

    def test_all_backends_zero_gradient_and_roundtrip(self):
        names = os.environ.get('TABLE1_CELLS', '').split() or list(CELLS)
        for name in names:
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
                state = cpu_tree(base.state_dict())
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
                model = GraphLoRA(base, c, 41, batch)
                self.assertEqual(digest(base), original)
                calls.clear()
                model.eval()
                with torch.no_grad():
                    actual, actual_budget = predict(model, c, batch)
                self.assertTrue(torch.equal(reference, actual), name)
                self.assertTrue(torch.equal(budget, actual_budget), name)
                self.assertEqual(calls, expected_calls)
                self.assertEqual(counts(model)['prompt_parameters'], 0)
                opt = make_optimizer(model, c)
                for step in range(2):
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    loss, _, _, _ = objective(model, c, batch, HP)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    grads = [p.grad for p in model.graphlora.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertTrue(any(g.abs().sum() > 0 for g in grads), name)
                    opt.step()
                model.eval()
                saved = cpu_tree(model.state_dict())
                with torch.no_grad():
                    before = predict(model, c, batch)[0]
                for h in handles:
                    h.remove()
                native = build(c, b, 41).to(self.device)
                restored = GraphLoRA(native, c, 41, batch)
                load_state(restored, saved)
                restored.eval()
                anonymous = batch.clone()
                anonymous.y = None
                with torch.no_grad():
                    after = predict(restored, c, anonymous)[0]
                self.assertTrue(torch.equal(before, after), name)
                model.close()
                restored.close()
                del model, restored, native, base, b, batch, anonymous, saved, state
                torch.cuda.empty_cache()

    def test_isolated_nodes_and_independent_graphs(self):
        x = torch.arange(28, device=self.device, dtype=torch.float32).reshape(7, 4) / 20
        edges = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]], device=self.device)
        from ..table1_frozen.deterministic import sum_by_index
        aggregate = sum_by_index(x[edges[0]], edges[1], len(x))
        for arch in ('GIN', 'GCN', 'GAT'):
            branch = Branch(arch, 4, 4, 2, 1).to(self.device).eval()
            with torch.no_grad():
                for name, p in branch.named_parameters():
                    if name.endswith('.up'):
                        p.normal_(0, .1)
                combined = branch(x, aggregate, edges)
                separate = branch(x[:3], aggregate[:3], edges[:, :2])
                self.assertTrue(torch.allclose(combined[:3], separate, atol=1e-6))
                isolated = branch(x[:1], x[:1] * 0, edges[:, :0])
                self.assertTrue(torch.isfinite(isolated).all())


if __name__ == '__main__':
    unittest.main()

"""Regression checks for causal controls; no real test-set data is used."""

import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

from .mechanism import (CONTROLS, PRECISIONS, bypass_quantizers, decompose,
                        intervene, make_control, quantizer_leaves, vector_metrics)
from .models import state_digest


def tiny_data():
    generator = torch.Generator().manual_seed(7)
    return Data(x=torch.rand(8, 5, generator=generator),
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 1], [1, 0, 3, 2, 5, 4, 0, 2]]),
                y=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
                train_mask=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
                val_mask=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool))


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_capacity_initialization_and_gradients(self):
        data = tiny_data()
        backbone = set()
        prompt = set()
        for precision in PRECISIONS:
            for control in CONTROLS:
                model = make_control(data, 41, precision, control)
                backbone.add(state_digest(model))
                if control == "original":
                    prompt.add(state_digest(model, "prompt_base"))
                expected = 0 if control == "none" else 412
                self.assertEqual(sum(p.numel() for p in model.prompt_aggr.parameters()), expected)
                with bypass_quantizers(model) if precision == "fp32" else torch.enable_grad():
                    logits, _ = model(data)
                    nn.functional.cross_entropy(logits, data.y).backward()
                self.assertTrue(torch.isfinite(logits).all())
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                if precision == "fp32":
                    for _, module in quantizer_leaves(model):
                        self.assertTrue(all(p.grad is None and not p.requires_grad for p in module.parameters()))
        self.assertEqual(len(backbone), 1)
        self.assertEqual(len(prompt), 1)

    def test_fp32_bypasses_quantizer_computation_and_restores(self):
        data = tiny_data()
        model = make_control(data, 41, "qat", "original").eval()
        leaves = quantizer_leaves(model)
        self.assertEqual(len(leaves), 5)
        before = model(data)[0]
        # Raising forward confirms bypass does not merely discard quantized output.
        with patch.object(leaves[0][1], "forward", side_effect=RuntimeError("executed")):
            with bypass_quantizers(model):
                out = model(data)[0]
                self.assertTrue(torch.isfinite(out).all())
                with bypass_quantizers(model):
                    torch.testing.assert_close(model(data)[0], out, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, "executed"):
                model(data)
        torch.testing.assert_close(model(data)[0], before, rtol=0, atol=0)

    def test_full_fp32_equals_explicit_gin(self):
        data = tiny_data()
        model = make_control(data, 41, "fp32", "original").eval()
        from .models import sum_messages
        from torch_geometric.utils import remove_self_loops
        edges, _ = remove_self_loops(data.edge_index)
        x = data.x
        for index, layer in enumerate((model.conv1, model.conv2)):
            agg = sum_messages(x, edges)
            if index:
                agg = model.prompt_aggr.add(agg)
            linear = layer.mlp[0]
            x = torch.relu(nn.functional.linear((1 + layer.eps) * x + agg, linear.weight, linear.bias))
        with bypass_quantizers(model):
            torch.testing.assert_close(model(data)[0], x, rtol=0, atol=0)

    def test_interventions_baseline_restoration_and_self_error(self):
        data = tiny_data()
        model = make_control(data, 41, "qat", "original").eval()
        baseline = model(data)[0]
        digest = state_digest(model, "all")
        rows = []
        with torch.no_grad(), intervene(model, "learned", records=rows):
            torch.testing.assert_close(model(data, node_mask=data.val_mask)[0], baseline, rtol=0, atol=0)
        self.assertNotEqual(rows[0]["neighbor"]["before_frobenius"],
                            rows[0]["full_update"]["before_frobenius"])
        for mode in ("off", "signflip", "shuffle", "mean", "parallel", "orthogonal"):
            with torch.no_grad(), intervene(model, mode):
                self.assertTrue(torch.isfinite(model(data)[0]).all())
        with self.assertRaises(ValueError):
            with intervene(model):
                raise ValueError("restore")
        torch.testing.assert_close(model(data)[0], baseline, rtol=0, atol=0)
        self.assertEqual(state_digest(model, "all"), digest)

    def test_local_zero_error_and_off_equal_identity_quantizer(self):
        data = tiny_data()
        model = make_control(data, 41, "qat", "original").eval()
        rows = []
        with torch.no_grad(), intervene(model, strength=0, records=rows):
            actual = model(data)[0]
        with patch.object(model.conv2.q_xw, "forward", side_effect=lambda x: x):
            expected = model(data)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(rows[0]["full_update"]["before_frobenius"], 0)

    def test_projection_and_error_metrics(self):
        error = torch.tensor([[3., 4.], [0., 0.], [-2., 5.]])
        delta = torch.tensor([[2., 1.], [1., 2.], [-1., -1.]])
        parallel, orthogonal = decompose(delta, error)
        torch.testing.assert_close(parallel + orthogonal, delta)
        torch.testing.assert_close((orthogonal * error).sum(1), torch.zeros(3), atol=1e-6, rtol=0)
        metrics = vector_metrics(error, error)
        self.assertEqual(metrics["norm_reduction"], 1)
        self.assertAlmostEqual(metrics["cosine"], 1)
        self.assertEqual(vector_metrics(error, torch.zeros_like(error))["norm_reduction"], 0)
        self.assertEqual(vector_metrics(error, -error)["norm_reduction"], -1)
        self.assertIsNone(vector_metrics(torch.zeros_like(error), delta)["norm_reduction"])

    def test_new_runner_selects_validation_and_tests_once(self):
        from .config import dataset_config
        from .mechanism import run_control
        config = dataset_config("CiteSeer", "pilot")
        config["epochs"] = 3
        epochs, calls = [], []

        def train(model, *args):
            epoch = len(epochs) + 1
            epochs.append(epoch)
            with torch.no_grad():
                model.conv2.eps.fill_(epoch)
            return dict(loss=1., accuracy=.5, legacy_budget_proxy=1.)

        def evaluate_mock(model, data, config, device, split):
            epoch = int(model.conv2.eps.item())
            calls.append((split, epoch))
            return dict(loss=3. if epoch == 1 else 1., accuracy=.5,
                        examples=4, legacy_budget_proxy=1.)

        with tempfile.TemporaryDirectory() as directory:
            with patch("runs.lorap_study.mechanism.train_epoch", side_effect=train), \
                    patch("runs.lorap_study.mechanism.evaluate", side_effect=evaluate_mock):
                result, _ = run_control(config, tiny_data(), 41, "fp32", "none", Path(directory) / "run")
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual([call for call in calls if call[0] == "test"], [("test", 2)])
        self.assertEqual(calls[:3], [("val", 1), ("val", 2), ("val", 3)])

    def test_interaction_sign_and_pairing(self):
        from .mechanism_report import paired_effects
        results = []
        # QAT: +10 pp, +0.2 CE reduction; FP32: +4 pp, +0.1 CE reduction.
        for precision in PRECISIONS:
            for control in CONTROLS:
                extra = control != "none"
                metrics = dict(accuracy=.5 + (0.1 if precision == "qat" else .04) * extra,
                               loss=1. - (.2 if precision == "qat" else .1) * extra)
                results.append(dict(precision=precision, control=control, seed=41,
                                    validation=metrics, test=metrics))
        for effect in paired_effects(list(reversed(results))):
            self.assertAlmostEqual(effect["interaction_accuracy_pp"], 6.)
            self.assertAlmostEqual(effect["interaction_loss"], .1)


if __name__ == "__main__":
    unittest.main()

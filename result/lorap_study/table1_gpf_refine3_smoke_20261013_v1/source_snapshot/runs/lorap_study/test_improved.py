"""Checks for initialization, calibration, rank allocation and held-out selection."""

import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import torch

from .improved import (FullUpdateDiagnostics, StablePromptBank, allocate_ranks,
                       bare_model, calibrate, install_prompt)
from .improve_run import choose_candidate
from .models import build_model, state_digest, sum_messages
from .prompts import parameter_count
from .test_study import graph_data, node_data


def fake_calibration(width, layers):
    return [dict(context_mean=torch.zeros(4), context_scale=torch.ones(4),
                 rms_floor=torch.tensor(.01), basis=torch.eye(width),
                 normalized_energy=torch.arange(width, 0, -1).float()) for _ in range(layers)]


class ImprovedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_start_and_gradients(self):
        for spectral in (False, True):
            bank = StablePromptBank(16, 20, [2], fake_calibration(16, 1), 41, spectral)
            x = torch.randn(12, 16, requires_grad=True)
            c = torch.randn(12, 4, requires_grad=True)
            target = torch.randn_like(x)
            torch.testing.assert_close(bank.add(x, c), x, rtol=0, atol=0)
            self.assertGreater(float(bank.factors[0].P_right.square().sum()), 0)
            optimizer = torch.optim.Adam(bank.parameters(), lr=.01)
            for _ in range(3):
                optimizer.zero_grad()
                loss = (bank.add(x, c) - target).square().mean()
                loss.backward()
                self.assertIsNone(c.grad)
                self.assertGreater(float(bank.factors[0].P_left.grad.abs().sum()), 0)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters()))
                optimizer.step()
            self.assertTrue(torch.isfinite(bank.add(torch.zeros_like(x), torch.full_like(c, 1e8))).all())
            self.assertEqual(parameter_count(bank), 497)

    def test_rank_budget_and_equal_capacity(self):
        infos = fake_calibration(16, 2)
        infos[0]["normalized_energy"] = torch.tensor([10., 9., 8., .1])
        infos[1]["normalized_energy"] = torch.tensor([10., 1., .1, .01])
        ranks = allocate_ranks(infos, 4)
        self.assertEqual(ranks, [3, 1])
        uniform = StablePromptBank(16, 20, [2, 2], infos, 41)
        adaptive = StablePromptBank(16, 20, ranks, infos, 41)
        self.assertEqual(parameter_count(uniform), parameter_count(adaptive))
        self.assertEqual(len(list(uniform.parameters())), len({id(p) for p in uniform.parameters()}))
        with self.assertRaises(ValueError):
            allocate_ranks(infos, 1)
        for info in infos:
            info["normalized_energy"].zero_()
        self.assertEqual(allocate_ranks(infos, 4), [2, 2])

    def test_calibration_is_training_only_bounded_and_nonmutating(self):
        data = node_data()
        model = bare_model("CiteSeer", 41, data)
        initial = state_digest(model, "all")
        rng = torch.random.get_rng_state().clone()
        cal = calibrate(model, "CiteSeer", data, max_nodes=2)
        self.assertEqual(cal[0]["seen_nodes"], int(data.train_mask.sum()))
        self.assertEqual(cal[0]["sampled_nodes"], 2)
        self.assertEqual(state_digest(model, "all"), initial)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        changed = data.clone()
        changed.y[~changed.train_mask] = 99
        again = calibrate(model, "CiteSeer", changed, max_nodes=2)
        for key in ("basis", "context_mean", "normalized_energy"):
            torch.testing.assert_close(cal[0][key], again[0][key], rtol=0, atol=0)

    def test_zero_start_preserves_actual_models_and_backbone_gradients(self):
        for name, data in (("CiteSeer", node_data()), ("MNIST", graph_data())):
            original = bare_model(name, 41, data if name == "CiteSeer" else None).eval()
            before = original(data)[0]
            for variant in ("zero_init", "stable_random", "stable_spectral", "adaptive_spectral"):
                model = bare_model(name, 41, data if name == "CiteSeer" else None).eval()
                cal = fake_calibration(16 if name == "CiteSeer" else 110, 1 if name == "CiteSeer" else 4)
                install_prompt(model, name, variant, 41, cal)
                torch.testing.assert_close(model(data)[0], before, rtol=0, atol=0)
                self.assertEqual(state_digest(model), state_digest(original))
            if name == "CiteSeer":
                original(data)[0].square().mean().backward()
                model(data)[0].square().mean().backward()
                for (left_name, left), (right_name, right) in zip(original.named_parameters(), model.named_parameters()):
                    if left_name.startswith("prompt_"):
                        break
                    self.assertEqual(left_name, right_name)
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)

    def test_mnist_legacy_layout_parity_and_train_calibration(self):
        data = graph_data()
        legacy = build_model("MNIST", "original", 41).eval()
        model = bare_model("MNIST", 41)
        cal = calibrate(model, "MNIST", {"train": data.to_data_list()}, max_nodes=8, max_graphs=1)
        self.assertEqual(len(cal), 4)
        self.assertTrue(all(info["seen_nodes"] == 5 for info in cal))
        install_prompt(model, "MNIST", "original", 41, cal)
        model.eval()
        self.assertEqual(state_digest(model, "all"), state_digest(legacy, "all"))
        a, b = model(data), legacy(data)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)

    def test_no_reference_or_svd_in_normal_forward(self):
        data = node_data()
        model = bare_model("CiteSeer", 41, data).eval()
        install_prompt(model, "CiteSeer", "stable_spectral", 41, fake_calibration(16, 1))
        with patch("runs.lorap_study.improved.sum_messages", wraps=sum_messages) as agg, \
                patch("torch.linalg.svd", side_effect=AssertionError("runtime SVD")):
            before = model(data)[0]
            self.assertEqual(agg.call_count, 0)
            with FullUpdateDiagnostics(model, data.val_mask):
                after = model(data)[0]
            self.assertEqual(agg.call_count, 2)
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_candidate_selection_ignores_test_and_accuracy(self):
        results = [dict(variant="a", validation=dict(loss=1., accuracy=.9), test=dict(accuracy=1.)),
                   dict(variant="b", validation=dict(loss=.5, accuracy=.2), test=dict(accuracy=0.)),
                   dict(variant="c", validation=dict(loss=.5, accuracy=.8), test=dict(accuracy=.9))]
        self.assertEqual(choose_candidate(results), "b")

    def test_multibatch_error_aggregation_uses_squared_norms(self):
        from .improve_report import merged_error
        batches = [dict(full_update=dict(before_frobenius=3., after_frobenius=0., prompt_norm=3., cosine=1.)),
                   dict(full_update=dict(before_frobenius=4., after_frobenius=4., prompt_norm=0., cosine=None))]
        # Global before=5, after=4; averaging batch reductions would give 0.5.
        self.assertAlmostEqual(merged_error(batches)["reduction"], .2)
        self.assertAlmostEqual(merged_error(batches)["cosine"], .6)

    def test_fit_withholds_test_then_evaluates_selected_checkpoint_once(self):
        from .config import dataset_config
        from .improve_run import fit, final_test
        config = dataset_config("CiteSeer", "pilot")
        config["epochs"] = 3
        seen, steps = [], []

        def train(model, *args):
            epoch = len(steps) + 1
            steps.append(epoch)
            with torch.no_grad():
                model.conv2.eps.fill_(epoch)
            return dict(loss=1., accuracy=.5, legacy_budget_proxy=1.)

        def eval_mock(model, data, config, device, split, diagnostics=None):
            epoch = int(model.conv2.eps.item())
            seen.append((split, epoch))
            return dict(loss=2. if epoch == 1 else 1., accuracy=.5, examples=3, legacy_budget_proxy=1.)

        data = node_data()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fit"
            with patch("runs.lorap_study.improve_run.train_epoch", side_effect=train), \
                    patch("runs.lorap_study.improve_run.evaluate", side_effect=eval_mock):
                model, result = fit(config, "zero_init", 41, data, fake_calibration(16, 1), directory)
                self.assertEqual(result["best_epoch"], 2)
                self.assertTrue(all(split == "val" for split, _ in seen))
                self.assertEqual(result["test_evaluations"], 0)
                final_test(model, result, config, data, directory)
                with self.assertRaises(ValueError):
                    final_test(model, result, config, data, directory)
        self.assertEqual([r for r in seen if r[0] == "test"], [("test", 2)])


if __name__ == "__main__":
    unittest.main()

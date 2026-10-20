"""Tests for the locked uniform-rank recipe and its component controls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .diagnostics import AggregationDiagnostics
from .improved import StablePromptBank, bare_model
from .models import model_counts, state_digest, sum_messages
from .prompts import quantization_context
from .test_improved import fake_calibration
from .test_study import graph_data, node_data
from .uniform import VARIANTS, UniformPromptBank, calibrate_uniform, install_uniform
from .uniform_run import final_test, fit_uniform, main, primary_config


class UniformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_main_matches_previous_random_bank_after_updates(self):
        cal = fake_calibration(16, 1)
        old = StablePromptBank(16, 20, [2], cal, 44, spectral=False)
        new = UniformPromptBank(16, 1, cal, 44)
        x, c = torch.randn(8, 16), torch.randn(8, 4)
        target = torch.randn_like(x)
        opts = [torch.optim.Adam(m.parameters(), lr=.01) for m in (old, new)]
        for _ in range(3):
            torch.testing.assert_close(old.add(x, c), new.add(x, c), rtol=0, atol=0)
            for model, opt in zip((old, new), opts):
                opt.zero_grad()
                (model.add(x, c)-target).square().mean().backward()
                opt.step()
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)

    def test_zero_start_finite_gradients_and_detached_context(self):
        for variant in VARIANTS[2:]:
            bank = UniformPromptBank(16, 1, fake_calibration(16, 1), 44, variant)
            x = torch.randn(8, 16, requires_grad=True)
            c = torch.randn(8, 4, requires_grad=True)
            if variant != "nonzero_start":
                self.assertTrue(torch.equal(bank.add(x, c), x))
            else:
                self.assertFalse(torch.equal(bank.add(x, c), x))
            opt = torch.optim.Adam(bank.parameters(), lr=.01)
            for _ in range(3):
                opt.zero_grad()
                bank.add(x, c).square().mean().backward()
                self.assertIsNone(c.grad)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters()))
                self.assertGreater(float(bank.factors[0].P_left.grad.abs().sum()), 0)
                opt.step()
            self.assertTrue(torch.isfinite(bank.add(torch.zeros_like(x), torch.full_like(c, 1e8))).all())

    def test_backbones_ranks_capacity_and_paired_initialization(self):
        for dataset, width, layers, counts in (("MNIST", 110, 4, [0, 2480, 3345, 3345, 3345, 3260, 3345]),
                                               ("CiteSeer", 16, 1, [0, 412, 497, 497, 497, 412, 497])):
            data = node_data()
            models = []
            for variant, expected in zip(VARIANTS, counts):
                model = bare_model(dataset, 44, data if dataset == "CiteSeer" else None)
                ranks = install_uniform(model, dataset, 44, fake_calibration(width, layers), variant)
                self.assertEqual(ranks, [2]*layers if variant != "none" else [])
                self.assertEqual(model_counts(model)["aggregate_prompt_parameters"], expected)
                models.append(model)
            self.assertEqual(len({state_digest(m) for m in models}), 1)
            for index in (3, 4, 5):
                self.assertTrue(torch.equal(models[2].prompt_aggr.a.weight, models[index].prompt_aggr.a.weight))
                for a, b in zip(models[2].prompt_aggr.factors, models[index].prompt_aggr.factors):
                    self.assertTrue(torch.equal(a.P_right, b.P_right))

    def test_standardization_and_clipping_change_only_condition_path(self):
        cal = fake_calibration(16, 1)
        cal[0]["context_mean"] = torch.tensor([1., -4., .5, .1])
        cal[0]["context_scale"] = torch.tensor([2., 1., .125, .1])
        standard = UniformPromptBank(16, 1, cal, 44)
        raw = UniformPromptBank(16, 1, cal, 44, "raw_conditions")
        with torch.no_grad():
            standard.factors[0].P_left.normal_()
            standard.condition.weight.normal_()
            standard.gate.weight.normal_()
        raw.load_state_dict(standard.state_dict())
        x, c = torch.randn(9, 16), torch.randn(9, 4)
        z = ((c-cal[0]["context_mean"])/cal[0]["context_scale"]).clamp(-5, 5)
        torch.testing.assert_close(standard.add(x, c), raw.add(x, z), rtol=0, atol=0)
        self.assertFalse(torch.allclose(standard.add(x, c), raw.add(x, c)))

    def test_no_conditions_skips_context_and_diagnostics_are_optional(self):
        data = graph_data()
        model = bare_model("MNIST", 44)
        install_uniform(model, "MNIST", 44, fake_calibration(110, 4), "no_conditions")
        model.eval()
        with torch.no_grad(), patch("runs.lorap_study.models.quantization_context", side_effect=AssertionError("context executed")), patch("runs.lorap_study.models.sum_messages", wraps=sum_messages) as agg:
            plain = model(data)[0]
            self.assertEqual(agg.call_count, 4)
        with torch.no_grad(), patch("runs.lorap_study.models.sum_messages", wraps=sum_messages) as agg:
            diagnostic = model(data, diagnostics=AggregationDiagnostics())[0]
            self.assertEqual(agg.call_count, 8)
        self.assertTrue(torch.equal(plain, diagnostic))

    def test_reference_free_calibration_matches_spectral_statistics(self):
        for dataset, data in (("CiteSeer", node_data()), ("MNIST", {"train": graph_data().to_data_list()})):
            model = bare_model(dataset, 44, data if dataset == "CiteSeer" else None)
            before = state_digest(model, "all")
            rng = torch.random.get_rng_state().clone()
            with patch("torch.linalg.svd", side_effect=AssertionError("SVD executed")):
                stats = calibrate_uniform(model, dataset, data, 44, max_nodes=3, max_graphs=2)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(before, state_digest(model, "all"))
            spectral = calibrate_uniform(model, dataset, data, 44, True, 3, 2)
            for a, b in zip(stats, spectral):
                for field in ("context_mean", "context_scale", "rms_floor"):
                    torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)
                self.assertEqual(a["sampled_nodes"], 3)
                self.assertEqual(a["split"], "train")
                self.assertFalse(a["fp_reference_used"])
                self.assertNotIn("basis", a)

    def test_training_withholds_test_and_selects_earliest_minimum(self):
        data = node_data()
        config = dict(primary_config("CiteSeer"), epochs=3, diagnostics=False)
        cal = calibrate_uniform(bare_model("CiteSeer", 44, data), "CiteSeer", data, 44)
        calls = []
        def evaluation(model, data, config, device, split, collector=None):
            calls.append(split)
            loss = [2., 1., 1., 1., 1.][len(calls)-1]
            return dict(loss=loss, accuracy=.5, mean_budget=10.)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"trial"
            with patch("runs.lorap_study.uniform_run.evaluate", side_effect=evaluation), patch("runs.lorap_study.uniform_run.FullUpdateDiagnostics", side_effect=AssertionError("reference executed")):
                model, result = fit_uniform(config, "main", 44, data, cal, directory)
                self.assertEqual(calls, ["val"]*4)
                self.assertEqual(result["best_epoch"], 2)
                self.assertIsNone(result["test"])
                final_test(model, result, config, data, directory)
                self.assertEqual(calls[-1], "test")
                self.assertEqual(result["test_evaluations"], 1)
                with self.assertRaises(ValueError):
                    final_test(model, result, config, data, directory)

    def test_full_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = primary_config("MNIST", full=True)
            path = Path(temporary)/"input.json"
            path.write_text(json.dumps(config))
            output = Path(temporary)/"output"
            main(["--config", str(path), "--output-dir", str(output), "--write-configs-only"])
            self.assertEqual(json.loads((output/"protocol.json").read_text())["config"], config)
            self.assertEqual(config["seeds"], list(range(41, 51)))
            self.assertEqual(config["epochs"], 200)
            self.assertIsNone(config["subset_sizes"])


if __name__ == "__main__":
    unittest.main()

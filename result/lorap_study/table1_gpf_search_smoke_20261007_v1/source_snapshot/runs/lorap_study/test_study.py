"""Run: python -m unittest runs.lorap_study.test_study -v"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data

from .config import dataset_config, VARIANTS
from .data import stratified_indices
from .diagnostics import AggregationDiagnostics, QuantizationObserver
from .models import (AggregationLayer, build_model, model_counts, optimizer_for, state_digest,
                     sum_messages)
from .prompts import ConditionedLoRAP, LoRAP, make_prompt, parameter_count, quantization_context
from .run import BestCheckpoint, run_one


def node_data():
    return Data(x=torch.rand(6, 5), y=torch.tensor([0, 1, 2, 0, 1, 2]),
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 2], [1, 2, 3, 4, 5, 0, 2]]),
                train_mask=torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
                val_mask=torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool),
                test_mask=torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool))


def graph_data():
    graphs = [Data(x=torch.rand(5, 1), pos=torch.rand(5, 2), y=torch.tensor([label]),
                   edge_index=torch.tensor([[0, 1, 2, 3, 4, 2], [1, 2, 3, 4, 0, 2]]))
              for label in (0, 1)]
    return Batch.from_data_list(graphs)


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_conditioned_initialization_and_gradients(self):
        original = make_prompt("original", 16, 20, 2, 41)
        conditioned = make_prompt("conditioned", 16, 20, 2, 41)
        x = torch.randn(7, 16, requires_grad=True)
        context = torch.randn(7, 4, requires_grad=True)
        self.assertTrue(torch.equal(original.add(x), conditioned.add(x, context)))
        conditioned.add(x, context).square().mean().backward()
        self.assertIsNone(context.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(conditioned.condition.weight.grad.abs().sum(), 0)
        self.assertGreater(conditioned.gate.weight.grad.abs().sum(), 0)
        with self.assertRaises(ValueError):
            conditioned.add(x)

    def test_context_uses_actual_direction_and_handles_isolated_nodes(self):
        quantizer = SimpleNamespace(alpha=torch.tensor([[0.5], [0.25]], requires_grad=True),
                                    bit=torch.tensor([[3.], [4.]], requires_grad=True))
        # Source 0 saturated on both channels; source 1 on neither.
        xq = torch.tensor([[1.5, 1.75], [0., 0.], [0.5, 0.25], [0., 0.]])
        edges = torch.tensor([[0, 1, 0], [2, 2, 1]])
        context = quantization_context(xq, edges, quantizer)
        torch.testing.assert_close(context[:, 0], torch.tensor([0., 1., 2., 0.]).log1p())
        torch.testing.assert_close(context[:, 3], torch.tensor([0., 1., 0.5, 0.]))
        torch.testing.assert_close(context[:, 1], torch.full((4,), torch.tensor([0.5, 0.25]).log().mean()))
        torch.testing.assert_close(context[:, 2], torch.full((4,), 3.5 / 8))
        empty = quantization_context(xq, torch.empty((2, 0), dtype=torch.long), quantizer)
        self.assertTrue(torch.isfinite(empty).all())
        self.assertEqual(float(empty[:, 3].sum()), 0)
        self.assertFalse(context.requires_grad)

    def test_parameter_matching(self):
        for width, expected_hidden in ((16, 15), (110, 11)):
            conditioned = make_prompt("conditioned", width, 20, 2, 41)
            mlp = make_prompt("mlp", width, 20, 2, 41)
            self.assertEqual(mlp.hidden_width, expected_hidden)
            self.assertLessEqual(abs(parameter_count(mlp) / parameter_count(conditioned) - 1), 0.05)
            self.assertTrue(all(p.requires_grad for p in mlp.parameters()))

    def test_paired_backbones_and_prompt_bases(self):
        for dataset in ("MNIST", "CiteSeer"):
            models = [build_model(dataset, variant, 41, node_data()) for variant in VARIANTS]
            self.assertEqual(len({state_digest(m) for m in models}), 1)
            self.assertEqual(state_digest(models[0], "prompt_base"), state_digest(models[1], "prompt_base"))
            self.assertNotEqual(state_digest(models[0]), state_digest(build_model(dataset, "original", 42, node_data())))

    def test_forward_backward_all_variants(self):
        for dataset, data in (("MNIST", graph_data()), ("CiteSeer", node_data())):
            for variant in VARIANTS:
                with self.subTest(dataset=dataset, variant=variant):
                    model = build_model(dataset, variant, 41, data)
                    optimizer = optimizer_for(model, dataset_config(dataset, "pilot"))
                    logits, budget = model(data)
                    loss = torch.nn.functional.cross_entropy(logits, data.y) + 1e-5 * budget.square()
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                    optimizer.step()
                    self.assertTrue(torch.isfinite(model(data)[0]).all())
                    prompt_ids = {id(p) for name, p in model.named_parameters() if name.startswith("prompt_")}
                    actual = {id(p) for group in optimizer.param_groups if group["name"] == "prompts" for p in group["params"]}
                    self.assertEqual(prompt_ids, actual)

    def test_diagnostics_does_not_change_prediction_or_run_when_disabled(self):
        data = graph_data()
        model = build_model("MNIST", "conditioned", 41).eval()
        with torch.no_grad(), patch("runs.lorap_study.models.sum_messages", wraps=sum_messages) as aggregate:
            plain = model(data)[0]
            self.assertEqual(aggregate.call_count, 4)
        collector = AggregationDiagnostics()
        with torch.no_grad(), patch("runs.lorap_study.models.sum_messages", wraps=sum_messages) as aggregate:
            instrumented = model(data, diagnostics=collector)[0]
            self.assertEqual(aggregate.call_count, 8)
        self.assertTrue(torch.equal(plain, instrumented))
        self.assertEqual(len(collector.result()["layers"]), 4)

    def test_diagnostic_math_sampling_and_mask(self):
        collector = AggregationDiagnostics(max_nodes=2)
        reference = torch.tensor([[3., 0.], [0., 4.], [1000., 1000.]])
        zero = torch.zeros_like(reference)
        mask = torch.tensor([True, True, False])
        collector.add("test", reference, zero, reference / 2, mask)
        result = collector.result()["layers"]["test"]
        self.assertEqual(result["nodes"], 2)
        self.assertAlmostEqual(result["error_before_frobenius"], 5)
        self.assertAlmostEqual(result["error_after_frobenius"], 2.5)
        self.assertAlmostEqual(result["relative_error_reduction"], 0.5)
        self.assertAlmostEqual(result["global_cosine"], 1)
        self.assertAlmostEqual(result["cumulative_energy"][-1], 1)
        a, b = AggregationDiagnostics(max_nodes=5), AggregationDiagnostics(max_nodes=5)
        for _ in range(3):
            data = torch.randn(20, 4)
            for item in (a, b):
                item.add("layer", data, data * 0, data * 0.5)
        self.assertTrue(torch.equal(a.layers["layer"]["sample"], b.layers["layer"]["sample"]))
        self.assertEqual(a.result()["layers"]["layer"]["sampled_nodes"], 5)
        zeros = AggregationDiagnostics()
        zeros.add("zero", torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2))
        self.assertIsNone(zeros.result()["layers"]["zero"]["relative_error_reduction"])

    def test_bit_observer_and_hook_cleanup(self):
        for dataset, data in (("MNIST", graph_data()), ("CiteSeer", node_data())):
            model = build_model(dataset, "original", 41, data).eval()
            with torch.no_grad(), QuantizationObserver(model) as observer:
                model(data)
            result = observer.result()
            self.assertTrue(result["layers"])
            # Positive feature quantizers store 5, representing 4 effective bits.
            self.assertAlmostEqual(result["mean_activation_effective_bits"], 4)
            self.assertTrue(all(not m._forward_pre_hooks for m in model.modules()))

    def test_fixed_stratified_subset(self):
        labels = [i % 10 for i in range(1000)]
        a = stratified_indices(labels, 100, 41)
        self.assertEqual(a, stratified_indices(labels, 100, 41))
        self.assertEqual(len(set(a)), 100)
        self.assertEqual([sum(labels[index] == c for index in a) for c in range(10)], [10] * 10)

    def test_best_checkpoint_keeps_earlier_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(1, 1)
            best = BestCheckpoint(Path(directory) / "best.pt")
            self.assertTrue(best.consider(0.2, 1, model))
            self.assertFalse(best.consider(0.2, 2, model))
            self.assertFalse(best.consider(0.3, 3, model))
            self.assertEqual(best.epoch, 1)

    def test_test_evaluated_once_after_loading_selected_checkpoint(self):
        model = torch.nn.Module()
        model.prompt_aggr = LoRAP(4, 5, 2)
        epoch = 0
        calls = []

        def train(*args):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                model.prompt_aggr.P_left.fill_(epoch)
            return dict(loss=1., accuracy=0., legacy_budget_proxy=0.)

        def evaluate(model, data, config, device, split, diagnostics=None):
            calls.append(split)
            if split == "test":
                self.assertTrue(torch.equal(model.prompt_aggr.P_left, torch.ones_like(model.prompt_aggr.P_left)))
            return dict(loss=0.1 if epoch == 1 else 0.2, accuracy=0.5, examples=1, legacy_budget_proxy=0.)

        config = dict(dataset_config("CiteSeer", "pilot"), epochs=2, diagnostics=False)
        with tempfile.TemporaryDirectory() as directory, \
             patch("runs.lorap_study.run.build_model", return_value=model), \
             patch("runs.lorap_study.run.train_epoch", side_effect=train), \
             patch("runs.lorap_study.run.evaluate", side_effect=evaluate):
            result = run_one(config, "original", 41, node_data(), Path(directory) / "trial", torch.device("cpu"))
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["test_evaluations"], 1)
        self.assertEqual(calls, ["val", "val", "test", "val"])

    def test_legacy_forward_parity(self):
        # Copy the new model state into the historical layout, then compare
        # logits and legacy storage accounting. This catches changed GIN order.
        from runs.mnist.lr_gin_mnist_bit_gpf_aggr_LoRA_node import qGIN
        data = graph_data()
        class FakeDataset:
            num_classes = 10
            def __getitem__(self, index):
                return data
        legacy = qGIN(FakeDataset(), 4, 110, 4, is_q=True, num_deg=1000,
                      uniform=True, init="uniform", num_prompts=20, rank=2).eval()
        current = build_model("MNIST", "original", 41).eval()
        state = {}
        for name, value in current.state_dict().items():
            translated = name
            for new, old in (("mlp.linear1", "nn.0"), ("mlp.bn1", "nn.1.bn"),
                             ("mlp.linear2", "nn.3"), ("mlp.bn2", "nn.4.bn")):
                translated = translated.replace(new, old)
            state[translated] = value
        legacy.load_state_dict(state, strict=True)
        with torch.no_grad():
            expected, expected_budget = legacy(data)
            actual, actual_budget = current(data)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_budget, expected_budget.squeeze())


if __name__ == "__main__":
    unittest.main()

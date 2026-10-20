"""Numerical, optimizer, batching, persistence and test-release invariants."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from .full_models import build_full_backbone
from .diagnostics import QuantizationObserver
from .full_run import sha_config
from .lora import (GraphLoRAQLinear, LoRAQLinear, NodeLoRAQLinear, baseline_config,
                   baseline_counts, baseline_optimizer, build_baseline, common_backbone_digest,
                   install_lora)
from .lora_run import release_tests, require_device, test_once
from .models import graph_linear, node_linear
from .run import write_json
from .test_study import node_data
from .uniform import calibrate_uniform


def graph_samples():
    return [Data(x=torch.rand(n, 1), pos=torch.rand(n, 2), y=torch.tensor([i]),
                 edge_index=torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.long)
                 if i == 0 else torch.empty((2, 0), dtype=torch.long))
            for i, n in enumerate((3, 7))]


def fixture(dataset):
    if dataset != "MNIST":
        return node_data()
    graphs = graph_samples()
    return {split: graphs for split in ("train", "val", "test")}


class LoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(19)

    def test_default_protocol(self):
        for name in ("Cora", "CiteSeer", "MNIST"):
            c = baseline_config(name)
            self.assertEqual(c["seeds"], [41, 42, 43])
            self.assertEqual(c["variants"], ["no_prompt", "main", "lora"])
            self.assertEqual(c["epochs"], 30 if name == "MNIST" else 200)
            self.assertEqual(c["lora_rank"], 2)
            self.assertEqual(c["lora_alpha"], 2)
        self.assertEqual(baseline_config("Cora")["a_loss"], 2.)
        self.assertEqual(baseline_config("MNIST")["subset_sizes"], dict(train=512, val=256, test=256))
        with self.assertRaises(ValueError):
            baseline_config("CIFAR10")

    def test_linear_formula_and_single_quantizer_calls(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                base = graph_linear(5, 3, 20, 4, False) if graph else node_linear(5, 3, 6, 4)
                layer = (GraphLoRAQLinear if graph else NodeLoRAQLinear)(base, rank=2, alpha=3)
                with torch.no_grad():
                    layer.lora_B.normal_(std=.2)
                x = torch.randn(6, 5)
                edges = torch.tensor([[0, 2], [2, 3]])
                qx, contribution = layer.fea_quant(x, edges) if graph else (layer.fea_quant(x), 0)
                qw = layer.weight_quant(layer.weight)
                expected = F.linear(qx, qw, layer.bias) + 1.5 * F.linear(F.linear(qx, layer.lora_A), layer.lora_B)
                with (patch.object(layer.fea_quant, "forward", wraps=layer.fea_quant.forward) as q,
                      patch.object(layer.weight_quant, "forward", wraps=layer.weight_quant.forward) as w):
                    if graph:
                        actual, returned_edges, budget = layer(x, edges, torch.tensor([2.]))
                        self.assertIs(returned_edges, edges)
                        torch.testing.assert_close(budget, torch.tensor([2.]) + contribution)
                    else:
                        actual = layer(x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(q.call_count, 1)
                    self.assertEqual(w.call_count, 1)

    def test_identity_feature_quantizer(self):
        for graph in (False, True):
            base = graph_linear(5, 3, 20, 4, False) if graph else node_linear(5, 3, 6, 4)
            base.fea_quant = nn.Identity()
            layer = (GraphLoRAQLinear if graph else NodeLoRAQLinear)(copy.deepcopy(base))
            x, edges = torch.randn(6, 5), torch.empty((2, 0), dtype=torch.long)
            if graph:
                actual, _, budget = layer(x, edges, torch.tensor([3.]))
                expected, _, old_budget = base(x, edges, torch.tensor([3.]))
                torch.testing.assert_close(budget, old_budget)
            else:
                actual, expected = layer(x), base(x)
            self.assertTrue(torch.equal(actual, expected))

    def test_zero_output_and_two_stage_factor_learning(self):
        base = node_linear(5, 3, 6, 4)
        base.weight_quant = nn.Identity()
        base.fea_quant = nn.Identity()
        layer = NodeLoRAQLinear(base)
        x = torch.randn(6, 5)
        expected = base(x).detach()
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        optimizer = torch.optim.SGD(layer.parameters(), lr=.1)
        layer(x).square().mean().backward()
        self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0)
        self.assertEqual(layer.lora_A.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        self.assertGreater(layer.lora_A.grad.abs().sum().item(), 0)

    def test_full_models_initialization_pairing_and_prompt_removal(self):
        for name in ("Cora", "CiteSeer", "MNIST"):
            c, data = baseline_config(name), fixture(name)
            base = build_full_backbone(c, 41, data)
            calibration = calibrate_uniform(base, c["family"], data, 41)
            models = [build_baseline(c, v, 41, data, calibration) for v in c["variants"]]
            self.assertEqual(len({common_backbone_digest(m) for m in models}), 1)
            inputs = Batch.from_data_list(data["train"]) if name == "MNIST" else data
            for m in models:
                m.eval()
            with torch.no_grad():
                y0, b0 = models[0](inputs)
                yl, bl = models[2](inputs)
            self.assertTrue(torch.equal(y0, yl))
            self.assertTrue(torch.equal(b0, bl))
            for i in (0, 2):
                self.assertEqual(baseline_counts(models[i])["prompt_parameters"], 0)
            self.assertGreater(baseline_counts(models[1])["prompt_parameters"], 0)
            counts = baseline_counts(models[2])
            self.assertEqual(len(counts["lora_targets"]), 11 if name == "MNIST" else 2)
            expected = sum(2 * (m.in_features + m.out_features) for m in models[2].modules()
                           if isinstance(m, LoRAQLinear))
            self.assertEqual(counts["lora_parameters"], expected)
            self.assertEqual(counts["lora_fp32_bytes"], 4 * expected)
            self.assertNotEqual(common_backbone_digest(models[0]),
                                common_backbone_digest(build_baseline(c, "no_prompt", 42, data)))

    def test_installation_does_not_advance_global_rng(self):
        c, data = baseline_config("Cora"), node_data()
        model = build_baseline(c, "no_prompt", 41, data)
        before = torch.get_rng_state().clone()
        digest = common_backbone_digest(model)
        install_lora(model, 41)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(digest, common_backbone_digest(model))
        with self.assertRaises(ValueError):
            install_lora(model, 41)
        for rank, alpha in ((0, 2), (2, 0), (2, float("nan"))):
            with self.assertRaises(ValueError):
                NodeLoRAQLinear(node_linear(5, 3, 6, 4), rank, alpha)

    def test_optimizer_and_actual_qat_updates(self):
        for name in ("Cora", "MNIST"):
            c, data = baseline_config(name), fixture(name)
            model = build_baseline(c, "lora", 41, data)
            optimizer = baseline_optimizer(model, c)
            groups = {g["name"]: g for g in optimizer.param_groups}
            self.assertEqual(groups["lora"]["weight_decay"], 0)
            self.assertEqual(groups["lora"]["lr"], c["lr"])
            self.assertEqual(groups["backbone"]["weight_decay"], c["weight_decay"])
            ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            batch = Batch.from_data_list(data["train"]) if name == "MNIST" else data
            logits, budget = model(batch)
            loss = F.cross_entropy(logits, batch.y) + c["a_loss"] * F.relu(budget - c["a_storage"]).square()
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(torch.isfinite(p.grad).all(), n)
                else:
                    self.assertIsNone(p.grad, n)
            optimizer.step()
            changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
            self.assertTrue(any(n.endswith(".weight") for n in changed))
            self.assertTrue(any("weight_quant.alpha" in n for n in changed))
            self.assertTrue(any("fea_quant.gama" in n for n in changed))
            self.assertTrue(any(n.endswith("lora_B") for n in changed))
            self.assertTrue(all(torch.equal(p, before[n]) for n, p in model.named_parameters() if not p.requires_grad))

    def test_graph_batch_independence_and_checkpoint_roundtrip(self):
        c, data = baseline_config("MNIST"), fixture("MNIST")
        model = build_baseline(c, "lora", 41, data).eval()
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, LoRAQLinear):
                    m.lora_B.normal_(std=.01)
            together = model(Batch.from_data_list(data["train"]))[0]
            separate = torch.cat([model(Batch.from_data_list([g]))[0] for g in data["train"]])
        torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-6)
        # Switching train/eval cannot merge or mutate base weights.
        before = common_backbone_digest(model)
        model.train().eval()
        self.assertEqual(before, common_backbone_digest(model))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_baseline(c, "lora", 41, data).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            actual = restored(Batch.from_data_list(data["train"]))[0]
        torch.testing.assert_close(together, actual, rtol=0, atol=0)

    def test_empty_edges_and_isolated_nodes(self):
        data = node_data()
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        model = build_baseline(baseline_config("Cora"), "lora", 41, data)
        logits, budget = model(data)
        self.assertTrue(torch.isfinite(logits).all())
        (logits.square().mean() + budget).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_quantization_observer_visits_only_real_quantizers(self):
        for name in ("Cora", "MNIST"):
            c, data = baseline_config(name), fixture(name)
            model = build_baseline(c, "lora", 41, data).eval()
            batch = Batch.from_data_list(data["train"]) if name == "MNIST" else data
            with torch.no_grad():
                expected = model(batch)[0]
                with QuantizationObserver(model) as observer:
                    actual = model(batch)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            stats = observer.result()
            self.assertAlmostEqual(stats["mean_activation_effective_bits"], 4)
            self.assertTrue(stats["layers"])
            self.assertTrue(all("lora_" not in key for key in stats["layers"]))
            self.assertTrue(all(not m._forward_pre_hooks for m in model.modules()))

    def test_cuda_is_required(self):
        with self.assertRaises(RuntimeError):
            require_device("cpu")
        with patch("torch.cuda.is_available", return_value=False), self.assertRaises(RuntimeError):
            require_device("cuda:0")

    def test_fixed4_qat_freezes_bits_but_trains_scales_and_lora(self):
        from quantize_function.u_quant_normal_debug import u_quant_fea
        config, data = baseline_config("MNIST", "fixed4"), fixture("MNIST")
        base = build_baseline(config, "no_prompt", 41, data).eval()
        model = build_baseline(config, "lora", 41, data).eval()
        batch = Batch.from_data_list(data["train"])
        self.assertEqual(common_backbone_digest(base), common_backbone_digest(model))
        with torch.no_grad():
            for actual, expected in zip(model(batch), base(batch)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsInstance(model.embedding.fea_quant, u_quant_fea)
        counts = baseline_counts(model)
        self.assertEqual(counts["lora_parameters"], 4426)
        self.assertEqual(counts["prompt_parameters"], 0)
        optimizer = baseline_optimizer(model, config)
        self.assertNotIn("feature_bits", [g["name"] for g in optimizer.param_groups])
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.train()
        logits, _ = model(batch)
        F.cross_entropy(logits, batch.y).backward()
        optimizer.step()
        changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
        for n, p in model.named_parameters():
            if n.endswith(".bit"):
                self.assertFalse(p.requires_grad, n)
                self.assertIsNone(p.grad, n)
                self.assertNotIn(n, changed)
        for suffix in (".weight", "weight_quant.alpha", "fea_quant.gama", "q_xw.alpha", "lora_B"):
            self.assertTrue(any(n.endswith(suffix) for n in changed), suffix)
        with torch.no_grad(), QuantizationObserver(model) as observer:
            model.eval()(batch)
        self.assertEqual(observer.result()["mean_activation_effective_bits"], 4)
        self.assertTrue(all(s["mean_effective_bits"] == 4 for s in observer.result()["layers"].values()))

    def test_fixed4_protocol_rejects_incompatible_settings(self):
        with self.assertRaises(ValueError):
            baseline_config("Cora", "fixed4")
        for key, value in (("bit", 3), ("a_loss", .1)):
            config = baseline_config("MNIST", "fixed4")
            config[key] = value
            with self.assertRaises(ValueError):
                build_baseline(config, "lora", 41, fixture("MNIST"))


class ReleaseTests(unittest.TestCase):
    def prepare(self, root):
        c = baseline_config("Cora")
        c.update(seeds=[41], epochs=3)
        for variant in c["variants"]:
            directory = root / "Cora" / variant / "seed_41"
            directory.mkdir(parents=True)
            history = [dict(epoch=i + 1, validation=dict(loss=v)) for i, v in enumerate((.3, .2, .2))]
            best = dict(epoch=2, validation_loss=.2, model={"weight": torch.ones(1, 1)})
            torch.save(best, directory / "best.pt")
            torch.save(dict(history=history, epoch=3, best=best, config_sha256=sha_config(c)), directory / "last.pt")
            (directory / "epochs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
            write_json(directory / "result.json", dict(config_sha256=sha_config(c), epochs=3,
                       common_backbone_sha256="paired", test=None, test_evaluations=0))
        return c

    def test_all_fits_required_and_earliest_checkpoint_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c = self.prepare(root)
            directory = root / "Cora" / "main" / "seed_41"
            saved = (directory / "result.json").read_text()
            (directory / "result.json").unlink()
            with self.assertRaises(FileNotFoundError):
                release_tests(root, c)
            self.assertFalse((root / "Cora" / "test_release.json").exists())
            (directory / "result.json").write_text(saved)
            best = torch.load(directory / "best.pt", weights_only=False)
            best["epoch"] = 3
            torch.save(best, directory / "best.pt")
            with self.assertRaises(AssertionError):
                release_tests(root, c)

    def test_test_release_and_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c = self.prepare(root)
            data = node_data()
            with self.assertRaises(RuntimeError):
                test_once(root, c, "lora", 41, data, None, torch.device("cpu"))
            release_tests(root, c)
            with patch("runs.lorap_study.lora_run.build_baseline", return_value=nn.Linear(1, 1, bias=False)), \
                 patch("runs.lorap_study.lora_run.evaluate", return_value=dict(loss=.4, accuracy=.5)) as evaluate:
                first = test_once(root, c, "lora", 41, data, None, torch.device("cpu"))
                second = test_once(root, c, "lora", 41, data, None, torch.device("cpu"))
                self.assertEqual(first, second)
                self.assertEqual(evaluate.call_count, 1)
            marker = root / "Cora" / "main" / "seed_41" / "test_started.json"
            marker.write_text("{}")
            with self.assertRaises(RuntimeError):
                test_once(root, c, "main", 41, data, None, torch.device("cpu"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CUDAParityTests(unittest.TestCase):
    def test_node_and_graph_forward_backward(self):
        for name in ("Cora", "MNIST"):
            c, data = baseline_config(name), fixture(name)
            model = build_baseline(c, "lora", 41, data).eval()
            batch = Batch.from_data_list(data["train"]) if name == "MNIST" else data
            with torch.no_grad():
                expected = model(batch)[0]
            model.cuda()
            actual, budget = model(batch.clone().cuda())
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
            (actual.square().mean() + budget).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()

"""Official objective parity and the declared GraphLoRA GIN/QAT extension."""
import importlib.util
import io
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj

from .graphlora import (LowRankGIN, backbone_digest, build_graphlora, contrastive_loss,
    graphlora_optimizer, is_adapter, model_counts, structure_loss, training_objective)
from .lora import baseline_config, build_baseline, common_backbone_digest
from .mechanism import quantizer_leaves
from .test_lora import fixture


def config_for(mode='a2q'):
    return dict(baseline_config('MNIST', mode), quantization_mode=mode, graphlora_rank=2,
                contrast_weight=.1, structure_weight=.1, contrast_tau=.5, contrast_supervised_weight=.2)


class GraphLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(31)

    def test_zero_start_pairing_budget_and_single_quantizer_calls(self):
        from contextlib import ExitStack
        for mode in ('fixed4', 'a2q'):
            data, config = fixture('MNIST'), config_for(mode)
            base = build_baseline(config, 'no_prompt', 41, data).eval()
            rng, python_rng = torch.get_rng_state(), random.getstate()
            model = build_graphlora(config, 'graphlora', 41, data).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(python_rng, random.getstate())
            self.assertEqual(backbone_digest(model), common_backbone_digest(base))
            self.assertEqual(model_counts(model)['adapter_parameters'], 4404)
            self.assertEqual(model_counts(model)['prompt_parameters'], 0)
            batch = Batch.from_data_list(data['train'])
            with torch.no_grad(), ExitStack() as stack:
                mocks = [stack.enter_context(patch.object(m, 'forward', wraps=m.forward)) for _, m in quantizer_leaves(model)]
                actual, budget = model(batch, return_aux=True)[:2]
                expected, base_budget = base(batch)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(budget, base_budget, rtol=0, atol=0)
                self.assertTrue(all(m.call_count == 1 for m in mocks))

    def test_graph_branch_formula_and_live_zero_start(self):
        branch = LowRankGIN(6, 2).eval()
        x = torch.randn(8, 6); edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        optimizer = torch.optim.SGD(branch.parameters(), lr=.02)
        out = branch(x, edges)
        self.assertEqual(out.abs().sum().item(), 0)
        out.sum().backward()
        self.assertGreater(branch.last.up.grad.abs().sum().item(), 0)
        optimizer.step(); optimizer.zero_grad()
        out = branch(x, edges)
        target = torch.zeros_like(x).index_add_(0, edges[1], x[edges[0]]) + (1 + branch.eps) * x
        explicit = F.linear(F.linear(F.relu(branch.norm(F.linear(F.linear(target, branch.first.down), branch.first.up))),
                                     branch.last.down), branch.last.up)
        torch.testing.assert_close(out, explicit, rtol=0, atol=0)
        out.square().mean().backward()
        for name in ('first.down', 'first.up', 'last.down'):
            self.assertGreater(dict(branch.named_parameters())[name].grad.abs().sum().item(), 0, name)

    def test_official_contrastive_objective_values_and_gradients(self):
        path = Path(__file__).parent/'graphlora_official/util.py'
        spec = importlib.util.spec_from_file_location('graphlora_official_loss_oracle', path)
        official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
        a = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
        b = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 2, 1, 0, 2])
        mask = .2 * ((labels[:, None] == labels[None, :]).double() - torch.eye(7)) + torch.eye(7)
        expected = .5 * (official.batched_gct_loss(a, b, 3, mask, .5) + official.batched_gct_loss(b, a, 3, mask, .5)).mean()
        actual = contrastive_loss(a, b, labels)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        for x, y in zip(torch.autograd.grad(actual, (a, b), retain_graph=True), torch.autograd.grad(expected, (a, b))):
            torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
        known = torch.tensor([True, True, True, False, False, False, False])
        changed = labels.clone(); changed[~known] = 9
        torch.testing.assert_close(contrastive_loss(a, b, labels, known), contrastive_loss(a, b, changed, known), rtol=0, atol=0)

    def test_structure_official_formula_and_graph_isolation(self):
        data = fixture('MNIST')['train']; batch = Batch.from_data_list(data)
        logits = torch.randn(batch.num_nodes, 4, requires_grad=True)
        edges, _ = add_remaining_self_loops(batch.edge_index, num_nodes=batch.num_nodes)
        # One graph: same weighted adjacency BCE as official GraphLoRA.py.
        one = logits[:3]; one_edges, _ = add_remaining_self_loops(data[0].edge_index, num_nodes=3)
        adjacency = to_dense_adj(one_edges, max_num_nodes=3)[0].clamp_max(1)
        weight = torch.where(adjacency.bool(), (9 - adjacency.sum()) / adjacency.sum(), 1.)
        probability = one.softmax(-1)
        expected = F.binary_cross_entropy(torch.sigmoid(probability @ probability.T), adjacency, weight=weight)
        actual = structure_loss(one, one_edges, torch.zeros(3, dtype=torch.long), include_self=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        together = structure_loss(logits, edges, batch.batch)
        separate = torch.stack([structure_loss(logits[batch.ptr[i]:batch.ptr[i+1]], graph.edge_index,
                            torch.zeros(graph.num_nodes, dtype=torch.long)) for i, graph in enumerate(data)]).mean()
        torch.testing.assert_close(together, separate)
        together.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        singleton = structure_loss(torch.randn(1, 4), torch.empty(2, 0, dtype=torch.long), torch.zeros(1, dtype=torch.long))
        self.assertEqual(singleton.item(), 0)

    def test_optimizer_training_and_quantization_freezes(self):
        for mode in ('fixed4', 'a2q'):
            config, data = config_for(mode), fixture('MNIST')
            model = build_graphlora(config, 'graphlora', 41, data).train()
            optimizer = graphlora_optimizer(model, config)
            group = next(g for g in optimizer.param_groups if g['name'] == 'graphlora')
            self.assertEqual((group['lr'], group['weight_decay']), (config['lr'], 0))
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            batch = Batch.from_data_list(data['train'])
            loss, _, metrics = training_objective(model, batch, config)
            self.assertTrue(torch.isfinite(loss).all())
            loss.backward(); optimizer.step()
            changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
            self.assertTrue(any(n.endswith('last.up') for n in changed))
            for suffix in ('weight_quant.alpha', 'fea_quant.gama', 'embedding.weight'):
                self.assertTrue(any(n.endswith(suffix) for n in changed), suffix)
            self.assertTrue(all(torch.equal(p, before[n]) for n, p in model.named_parameters() if not p.requires_grad))
            self.assertEqual(any(n.endswith('fea_quant.bit') for n in changed), mode == 'a2q')
            self.assertGreater(metrics['contrast'].item(), 0)

    def test_eval_graph_independence_and_save_load(self):
        data = fixture('MNIST'); config = config_for()
        model = build_graphlora(config, 'graphlora', 41, data).eval()
        with torch.no_grad():
            for branch in model.graphlora_branches:branch.last.up.normal_(std=.02)
            batch = Batch.from_data_list(data['train'])
            output = model(batch)[0]
            separate = torch.cat([model(Batch.from_data_list([g]))[0] for g in data['train']])
            torch.testing.assert_close(output, separate, rtol=1e-4, atol=1e-5)
            stream = io.BytesIO(); torch.save(model.state_dict(), stream); stream.seek(0)
            restored = build_graphlora(config, 'graphlora', 41, data).eval()
            restored.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(output, restored(batch)[0], rtol=0, atol=0)


if __name__ == '__main__':unittest.main()

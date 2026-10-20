"""Small GPU checks for tunable prompts without loading any dataset."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .model import GPFLoRAP, counts, make_optimizer
from ..table1_frozen.backends import optimizer
from ..table1_frozen.common import digest


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def message(self, x):
        return x

    def aggregate(self, x):
        return x

    def update(self, x):
        return torch.tanh(x)

    def propagate(self, x):
        return self.update(self.aggregate(self.message(x)))

    def forward(self, x):
        return self.propagate(self.linear(x))


class TinyBackbone(nn.Module):
    def __init__(self, node=True, categorical=False):
        super().__init__()
        self.node = node
        if not node:
            self.embedding = nn.Embedding(7, 4) if categorical else nn.Linear(3, 4)
        self.convs = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.head = nn.Linear(4, 2)
        self.register_buffer('budget', torch.tensor([7.]))
        self.register_buffer('observations', torch.tensor(0))

    def forward(self, data):
        if self.training:
            self.observations.add_(1)
        x = data.x if self.node else self.embedding(data.x)
        for conv in self.convs:
            x = conv(x)
        return self.head(x), self.budget


class HyperparameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Hyperparameter tests require CUDA')
        torch.cuda.set_device(0)
        torch.set_num_threads(4)

    def fixture(self, node=True, categorical=False):
        torch.manual_seed(711)
        base = TinyBackbone(node, categorical).cuda()
        x = (torch.arange(6, device='cuda') if categorical else
             torch.randn(6, 4 if node else 3, device='cuda'))
        c = dict(node=node, layers=2, framework='QAT', lr=.003, weight_decay=.02)
        return base, c, SimpleNamespace(x=x)

    def manual(self, model, example, input_scale=1., aggregation_scale=1., native=False):
        def add(prompt, value, scale):
            if native:
                return prompt.add(value)
            factors = prompt.p_list if hasattr(prompt, 'p_list') else prompt.P_left @ prompt.P_right
            return value + scale * (F.softmax(prompt.a(value), dim=1) @ factors)
        base = model.backbone
        categorical = not base.node and isinstance(base.embedding, nn.Embedding)
        x = F.embedding(example.x, base.embedding.weight) if categorical else example.x
        x = add(model.prompts['input'], x, input_scale)
        if not base.node and not categorical:
            x = F.linear(x, base.embedding.weight, base.embedding.bias)
        for conv, prompt in zip(base.convs, model.prompts['aggregation']):
            x = torch.tanh(add(prompt, conv.linear(x), aggregation_scale))
        return base.head(x)

    def test_defaults_preserve_native_prompt_arithmetic(self):
        base, c, example = self.fixture()
        identical = copy.deepcopy(base)
        model = GPFLoRAP(base, c, 41, example)
        explicit = GPFLoRAP(identical, c, 41, example, input_scale=1.,
                            aggregation_scale=1., zero_init=False)
        self.addCleanup(model.close)
        self.addCleanup(explicit.close)
        model.eval()
        explicit.eval()
        self.assertTrue(torch.equal(model(example)[0], self.manual(model, example, native=True)))
        self.assertTrue(torch.equal(model(example)[0], explicit(example)[0]))
        self.assertEqual(digest(model), digest(explicit))

    def test_scale_formula_at_all_input_sites(self):
        for node, categorical in [(True, False), (False, False), (False, True)]:
            with self.subTest(node=node, categorical=categorical):
                base, c, example = self.fixture(node, categorical)
                unchanged = copy.deepcopy(base).eval()
                model = GPFLoRAP(base, c, 41, example, input_scale=.25, aggregation_scale=1.75)
                self.addCleanup(model.close)
                model.eval()
                expected = self.manual(model, example, .25, 1.75)
                torch.testing.assert_close(model(example)[0], expected, rtol=1e-6, atol=1e-7)
                model.input_scale = model.aggregation_scale = 0.
                self.assertTrue(torch.equal(model(example)[0], unchanged(example)[0]))
                self.assertEqual(counts(model)['prompt_parameters'],
                                 sum(p.numel() for p in model.prompts.parameters()))

    def test_zero_initialization_preserves_state_rng_and_learns(self):
        base, c, example = self.fixture()
        native = copy.deepcopy(base).eval()
        original, cpu_rng, gpu_rng = digest(base), torch.get_rng_state(), torch.cuda.get_rng_state()
        model = GPFLoRAP(base, c, 41, example, zero_init=True)
        self.addCleanup(model.close)
        self.assertEqual(digest(base), original)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        regular = GPFLoRAP(copy.deepcopy(native), c, 41, example)
        self.addCleanup(regular.close)
        zeroed = {'input.p_list'} | {f'aggregation.{i}.P_left' for i in range(c['layers'])}
        for name, parameter in model.prompts.named_parameters():
            if name in zeroed:
                self.assertEqual(parameter.count_nonzero().item(), 0)
            else:
                self.assertTrue(torch.equal(parameter, dict(regular.prompts.named_parameters())[name]))
        self.assertEqual(counts(model)['prompt_parameters'], counts(regular)['prompt_parameters'])
        model.eval()
        actual, budget = model(example)
        expected, native_budget = native(example)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(budget, native_budget))
        labels = torch.tensor([0, 1, 0, 0, 1, 1], device='cuda')
        opt = make_optimizer(model, c, prompt_lr=.01)
        F.cross_entropy(actual, labels).backward()
        self.assertGreater(model.prompts['input'].p_list.grad.abs().sum().item(), 0.)
        for prompt in model.prompts['aggregation']:
            self.assertGreater(prompt.P_left.grad.abs().sum().item(), 0.)
            self.assertEqual(prompt.P_right.grad.count_nonzero().item(), 0)
        opt.step()
        opt.zero_grad()
        F.cross_entropy(model(example)[0], labels).backward()
        for prompt in model.prompts['aggregation']:
            self.assertGreater(prompt.P_right.grad.abs().sum().item(), 0.)

    def test_independent_prompt_optimizer_settings(self):
        base, c, example = self.fixture()
        # Exercise native A2Q bit/scale grouping as well as backbone parameters.
        c.update(framework='A2Q', lr_bit=.0001, lr_feature=.0002)
        base.quant = nn.Module()
        base.quant.bit = nn.Parameter(torch.ones(1, device='cuda'))
        base.fea_quant = nn.Module()
        base.fea_quant.gama = nn.Parameter(torch.ones(1, device='cuda'))
        model = GPFLoRAP(base, c, 41, example)
        self.addCleanup(model.close)
        native = optimizer(base, c)
        default = make_optimizer(model, c)
        tuned = make_optimizer(model, c, prompt_lr=.04, prompt_weight_decay=.006)
        for expected, actual in zip(native.param_groups, tuned.param_groups[:-1]):
            self.assertEqual(expected['name'], actual['name'])
            self.assertEqual(expected['lr'], actual['lr'])
            self.assertEqual(expected['weight_decay'], actual['weight_decay'])
            self.assertEqual([id(p) for p in expected['params']], [id(p) for p in actual['params']])
        self.assertEqual(default.param_groups[-1]['lr'], c['lr'])
        self.assertEqual(default.param_groups[-1]['weight_decay'], 0.)
        self.assertEqual(tuned.param_groups[-1]['lr'], .04)
        self.assertEqual(tuned.param_groups[-1]['weight_decay'], .006)
        ids = [id(p) for group in tuned.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})

    def test_invalid_settings_fail_before_backbone_mutation(self):
        base, c, example = self.fixture()
        original = digest(base)
        for field in ('input_scale', 'aggregation_scale'):
            for value in (-1., float('nan'), float('inf'), True, None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    GPFLoRAP(base, c, 41, example, **{field: value})
        with self.assertRaises(ValueError):
            GPFLoRAP(base, c, 41, example, zero_init='false')
        self.assertEqual(digest(base), original)
        model = GPFLoRAP(base, c, 41, example)
        self.addCleanup(model.close)
        for field in ('prompt_lr', 'prompt_weight_decay'):
            for value in (-1., float('nan'), float('inf'), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    make_optimizer(model, c, **{field: value})


if __name__ == '__main__':
    unittest.main()

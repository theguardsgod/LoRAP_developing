"""GLoRA formula, official-code parity, compact export and search invariants."""
import copy
import importlib.util
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from .diagnostics import QuantizationObserver
from .glora import (ADAPTER_TENSORS, ARCHITECTURES, GLoRAQLinear, CompactGLoRAQLinear,
    NONE_ARCH, REFERENCE_ARCH, adapter_name, backbone_digest, build_glora, compact_model,
    glora_optimizer, install_glora, layers, model_counts, set_subnet, subnet_parameters)
from .glora_search import evolutionary_search
from .lora import baseline_config, build_baseline, common_backbone_digest
from .models import graph_linear, node_linear
from .test_lora import fixture


def plain_layer(inputs=5,outputs=3,bias=True):
    layer=nn.Linear(inputs,outputs,bias=bias)
    layer.weight_quant=nn.Identity();layer.fea_quant=nn.Identity();layer.bit=4
    return layer


class GLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_432_paths_noop_and_parameter_counts(self):
        self.assertEqual(len(ARCHITECTURES),432)
        torch.manual_seed(23)
        base=plain_layer()
        layer=GLoRAQLinear(base,2).eval()
        x=torch.randn(7,5)
        for index in range(432):
            layer.set_architecture(index)
            torch.testing.assert_close(layer(x),base(x),rtol=0,atol=0)
            compact=CompactGLoRAQLinear(layer).eval()
            actual=sum(p.numel() for n,p in compact.named_parameters() if adapter_name(n))
            self.assertEqual(actual,layer.active_parameters(),index)
            torch.testing.assert_close(compact(x),layer(x),rtol=0,atol=0)

    def test_full_formula_and_single_quantizer_calls(self):
        for graph in (False,True):
            base=graph_linear(5,3,20,4,False) if graph else node_linear(5,3,6,4)
            layer=GLoRAQLinear(base,2).eval()
            with torch.no_grad():
                for name,p in layer.named_parameters():
                    if adapter_name(name):p.normal_(std=.03)
            x=torch.randn(6,5);edges=torch.tensor([[0,2],[2,3]])
            qx,contribution=layer.fea_quant(x,edges) if graph else (layer.fea_quant(x),0)
            qw=layer.weight_quant(layer.weight)
            A=layer.glora_Ad@layer.glora_Au;B=layer.glora_Bd@layer.glora_Bu;C=layer.glora_Cd@layer.glora_Cu
            expected=F.linear(qx,qw+qw*A+B,layer.bias+layer.bias*layer.glora_D+layer.glora_E+(qw@C).flatten())
            with patch.object(layer.fea_quant,'forward',wraps=layer.fea_quant.forward) as fq, \
                 patch.object(layer.weight_quant,'forward',wraps=layer.weight_quant.forward) as wq:
                if graph:
                    actual,returned,budget=layer(x,edges,torch.tensor([2.]))
                    self.assertIs(returned,edges)
                    torch.testing.assert_close(budget,torch.tensor([2.])+contribution)
                else:actual=layer(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                self.assertEqual((fq.call_count,wq.call_count),(1,1))

    def test_support_gradients_and_zero_up_first_step(self):
        layer=GLoRAQLinear(plain_layer(),2).eval()
        x=torch.randn(8,5)
        optimizer=torch.optim.SGD(layer.parameters(),lr=.05)
        layer(x).square().mean().backward()
        for name in ('Ad','Bd','Cd','D','E'):
            self.assertGreater(getattr(layer,'glora_'+name).grad.abs().sum().item(),0,name)
        for name in ('Au','Bu','Cu'):
            self.assertEqual(getattr(layer,'glora_'+name).grad.abs().sum().item(),0,name)
        optimizer.step();optimizer.zero_grad()
        layer(x).square().mean().backward()
        for name in ('Au','Bu','Cu'):
            self.assertGreater(getattr(layer,'glora_'+name).grad.abs().sum().item(),0,name)

    def test_pairing_rng_and_observer_for_all_backends(self):
        for dataset,mode in (('Cora','a2q'),('MNIST','a2q'),('MNIST','fixed4')):
            config,data=baseline_config(dataset,mode),fixture(dataset)
            base=build_baseline(config,'no_prompt',41,data)
            rng=torch.get_rng_state().clone();python_rng=random.getstate()
            model=build_glora(config,41,data)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(python_rng,random.getstate())
            self.assertEqual(backbone_digest(model),common_backbone_digest(base))
            batch=Batch.from_data_list(data['train']) if dataset=='MNIST' else data
            model.eval();base.eval()
            with torch.no_grad(),QuantizationObserver(model) as observer:
                y,b=model(batch);by,bb=base(batch)
            torch.testing.assert_close(y,by,rtol=0,atol=0)
            torch.testing.assert_close(b,bb,rtol=0,atol=0)
            self.assertEqual(observer.result()['mean_activation_effective_bits'],4)
            self.assertEqual(model_counts(model)['external_prompt_parameters'],0)
            optimizer=glora_optimizer(model,config)
            groups={g['name']:g for g in optimizer.param_groups}
            self.assertEqual(groups['glora']['lr'],config['lr'])
            self.assertEqual(groups['glora']['weight_decay'],0)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            model.train()
            for _,layer in layers(model):layer.sample_paths=False
            out,budget=model(batch)
            (F.cross_entropy(out,batch.y)+config['a_loss']*F.relu(budget-config['a_storage']).square()).backward()
            optimizer.step()
            changed={n for n,p in model.named_parameters() if not torch.equal(p,before[n])}
            for suffix in ('weight_quant.alpha','fea_quant.gama','.weight','glora_Bd'):
                self.assertTrue(any(n.endswith(suffix) for n in changed),suffix)
            self.assertTrue(all(torch.equal(p,before[n]) for n,p in model.named_parameters() if not p.requires_grad))
            if mode=='fixed4':self.assertNotIn('feature_bits',groups)
            else:self.assertTrue(any(n.endswith('fea_quant.bit') for n in changed))

    def test_sample_rng_resume_and_eval_determinism(self):
        model=build_glora(baseline_config('MNIST'),41,fixture('MNIST'))
        batch=Batch.from_data_list(fixture('MNIST')['train'])
        model.eval()
        with torch.no_grad():
            for n,p in model.named_parameters():
                if adapter_name(n):p.normal_(std=.01)
        state=model._glora_rng.getstate()
        with torch.no_grad():a=model(batch)[0];b=model(batch)[0]
        self.assertTrue(torch.equal(a,b));self.assertEqual(state,model._glora_rng.getstate())
        layer=layers(model)[0][1]
        layer.train();x=torch.randn(5,3);edges=torch.empty((2,0),dtype=torch.long)
        state=model._glora_rng.getstate()
        first=layer(x,edges,torch.zeros(1))[0]
        model._glora_rng.setstate(state)
        second=layer(x,edges,torch.zeros(1))[0]
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_compact_graph_independence_and_checkpoint_roundtrip(self):
        for mode in ('fixed4','a2q'):
            config,data=baseline_config('MNIST',mode),fixture('MNIST')
            model=build_glora(config,41,data).eval()
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if adapter_name(name):p.normal_(std=.01)
            candidate=tuple((i*39)%432 for i in range(len(layers(model))))
            set_subnet(model,candidate)
            compact=compact_model(model).eval()
            self.assertEqual(model_counts(compact)['adapter_parameters'],subnet_parameters(model,candidate))
            batch=Batch.from_data_list(data['train'])
            with torch.no_grad():
                expected=model(batch)[0];actual=compact(batch)[0]
                separate=torch.cat([compact(Batch.from_data_list([g]))[0] for g in data['train']])
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            torch.testing.assert_close(actual,separate,rtol=1e-4,atol=1e-5)
            stream=io.BytesIO();torch.save(compact.state_dict(),stream);stream.seek(0)
            restored=build_glora(config,41,data)
            set_subnet(restored,candidate);restored=compact_model(restored).eval()
            restored.load_state_dict(torch.load(stream,weights_only=True))
            with torch.no_grad():torch.testing.assert_close(actual,restored(batch)[0],rtol=0,atol=0)

    def test_no_bias_single_output_and_invalid_install(self):
        layer=GLoRAQLinear(plain_layer(3,1,False),2).eval()
        self.assertEqual(layer(torch.randn(4,3)).shape,(4,1))
        for rank in (0,-1,True):
            with self.assertRaises(ValueError):GLoRAQLinear(plain_layer(),rank)
        model=build_glora(baseline_config('MNIST'),41,fixture('MNIST'))
        with self.assertRaises(ValueError):install_glora(model,41)
        with self.assertRaises(ValueError):set_subnet(model,[0])


class SearchTests(unittest.TestCase):
    def test_zero_mutation_random_fill_and_late_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            arguments=dict(depth=2,seed=41,population=6,generations=3,elite=1,mutation_probability=0.)
            def metrics(candidate):return dict(loss=sum(candidate)/1000,accuracy=.5)
            expected=evolutionary_search(metrics,sum,state_path=root/'full.pt',**arguments)
            calls=0
            def interrupted(candidate):
                nonlocal calls
                calls+=1
                if calls==10:raise RuntimeError('interrupted during random fill')
                return metrics(candidate)
            with self.assertRaises(RuntimeError):
                evolutionary_search(interrupted,sum,state_path=root/'resume.pt',**arguments)
            actual=evolutionary_search(metrics,sum,state_path=root/'resume.pt',**arguments)
            self.assertEqual(actual,expected)
            self.assertEqual(torch.load(root/'full.pt',weights_only=False),torch.load(root/'resume.pt',weights_only=False))

    def test_search_resume_matches_uninterrupted_and_never_uses_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def metrics(candidate):return dict(loss=sum(candidate)/1000,accuracy=.5)
            arguments=dict(depth=3,seed=41,population=6,generations=3,elite=2,objective_id='same-checkpoint')
            full=evolutionary_search(metrics,lambda c:sum(c),state_path=root/'full.pt',**arguments)
            calls=[]
            def interrupted(candidate):
                calls.append(candidate)
                if len(calls)==5:raise RuntimeError('interrupted')
                return metrics(candidate)
            with self.assertRaises(RuntimeError):
                evolutionary_search(interrupted,lambda c:sum(c),state_path=root/'resume.pt',**arguments)
            resumed=evolutionary_search(metrics,lambda c:sum(c),state_path=root/'resume.pt',**arguments)
            self.assertEqual(full,resumed)
            a=torch.load(root/'full.pt',weights_only=False);b=torch.load(root/'resume.pt',weights_only=False)
            self.assertEqual(a['records'],b['records']);self.assertEqual(a['rng'],b['rng'])
            with self.assertRaises(ValueError):
                evolutionary_search(metrics,lambda c:0,state_path=root/'test.pt',split='test',**arguments)
            with self.assertRaises(ValueError):
                evolutionary_search(metrics,lambda c:0,state_path=root/'full.pt',**dict(arguments,objective_id='changed'))

    def test_budget_and_earliest_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=evolutionary_search(lambda c:dict(loss=1.,accuracy=.5),lambda c:sum(c),depth=2,seed=1,
                state_path=Path(temporary)/'search.pt',population=4,generations=2,elite=2,max_parameters=100)
            state=torch.load(Path(temporary)/'search.pt',weights_only=False)
            self.assertTrue(all(r['parameters']<=100 for r in state['records']))
            self.assertEqual(result['best']['order'],0)
            self.assertEqual(result['evaluated'],8)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
class OfficialParityTests(unittest.TestCase):
    def test_all_official_paths_forward_and_backward(self):
        path=Path(__file__).parent/'glora_official/GLoRA/layer.py'
        spec=importlib.util.spec_from_file_location('official_glora_oracle',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ours=GLoRAQLinear(plain_layer(),2).cuda().eval()
        official=module.SuperScalableLinear(5,3,2).cuda()
        with torch.no_grad():
            for name,p in ours.named_parameters():
                if adapter_name(name):p.normal_(std=.02)
            official.weight.copy_(ours.weight);official.bias.copy_(ours.bias)
            for suffix in ADAPTER_TENSORS:getattr(official,suffix).copy_(getattr(ours,'glora_'+suffix))
        x=torch.randn(6,5,device='cuda')
        for index,path in enumerate(ARCHITECTURES):
            ours.set_architecture(index)
            official.eval_config={key:('LoRA_2' if mode=='lora' else mode) for key,mode in zip('ABCDE',path)}
            ours.zero_grad();official.zero_grad()
            actual=ours(x);expected=official(x)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.square().mean().backward();expected.square().mean().backward()
            for suffix in ADAPTER_TENSORS:
                a=getattr(ours,'glora_'+suffix).grad;b=getattr(official,suffix).grad
                self.assertEqual(a is None,b is None)
                if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()

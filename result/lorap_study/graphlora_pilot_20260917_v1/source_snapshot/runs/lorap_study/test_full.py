"""Full-split model compatibility, CUDA enforcement, resume and test release."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .full_config import DATASETS,full_config,validate
from .full_models import build_full_backbone
from .full_run import (fit,get_rng,restore_rng,require_cuda,sha_config,test_once,train_gpu_epoch)
from .full_suite import choose_job,release_tests
from .improved import bare_model
from .models import optimizer_for,state_digest
from .run import write_json
from .test_improved import fake_calibration
from .test_study import graph_data,node_data
from .uniform import install_uniform


def graphs(channels=1):
    result=graph_data().to_data_list()
    for graph in result:
        graph.x=torch.rand(graph.num_nodes,channels)
    return {split:[g.clone() for g in result] for split in ("train","val","test")}


class FullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_configs_require_official_splits_and_gpu(self):
        for name in DATASETS:
            cfg=full_config(name)
            validate(cfg)
            self.assertEqual(cfg["epochs"],200)
            self.assertEqual(cfg["seeds"],list(range(41,51)))
            self.assertIsNone(cfg["subset_sizes"])
            self.assertEqual(cfg["device"],"cuda")
            with self.assertRaises(ValueError):
                validate(dict(cfg,device="cpu"))
            with self.assertRaises(ValueError):
                validate(dict(cfg,subset_sizes={"train":512}))
        with patch("torch.cuda.is_available",return_value=False),self.assertRaises(RuntimeError):
            require_cuda()
        with self.assertRaises(RuntimeError):
            train_gpu_epoch(None,None,None,{},torch.device("cpu"))

    def test_mnist_and_citeseer_backbones_match_previous_implementation(self):
        for name,data in (("MNIST",graphs()),("CiteSeer",node_data())):
            full=build_full_backbone(full_config(name),41,data)
            old=bare_model(name,41,data if name=="CiteSeer" else None)
            self.assertEqual(state_digest(full,"all"),state_digest(old,"all"))

    def test_new_dataset_dimensions_and_paired_backbones(self):
        for name,data in (("Cora",node_data()),("CIFAR10",graphs(3))):
            cfg=full_config(name)
            width,layers=(16,1) if name=="Cora" else (110,4)
            digests=[]
            for variant in cfg["variants"]:
                model=build_full_backbone(cfg,41,data)
                install_uniform(model,cfg["family"],41,fake_calibration(width,layers),variant)
                digests.append(state_digest(model))
                if name=="CIFAR10":
                    self.assertEqual(model.embedding.in_features,5)
                    self.assertEqual(model.lin2.out_features,10)
            self.assertEqual(len(set(digests)),1)

    def test_test_release_waits_for_every_fit(self):
        cfg=dict(full_config("Cora"),seeds=[41],variants=["none","main"])
        queue=[dict(dataset="Cora",variant=v,seed=41,phase="fit",status="complete") for v in cfg["variants"]]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for job in queue:
                write_json(root/"Cora"/job["variant"]/"seed_41/result.json",dict(test_evaluations=0,test=None))
            queue[-1]["status"]="pending"
            release_tests(root,{"configs":[cfg]},queue)
            self.assertFalse((root/"Cora/test_release.json").exists())
            queue[-1]["status"]="complete"
            release_tests(root,{"configs":[cfg]},queue)
            released=json.loads((root/"Cora/test_release.json").read_text())
            self.assertEqual(len(released["fits"]),2)
            self.assertEqual(released["config_sha256"],sha_config(cfg))

    def test_queue_keeps_both_task_types_running_and_withholds_tests(self):
        queue=[dict(phase="fit",task="node",dataset="Cora",status="pending"),
               dict(phase="fit",task="graph",dataset="MNIST",status="pending"),
               dict(phase="test",task="node",dataset="Cora",status="pending")]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertIs(choose_job(root,queue,"graph"),queue[1])
            self.assertIs(choose_job(root,queue,"node"),queue[0])

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA integration test requires host GPU access")
    def test_all_variants_train_on_gpu_for_new_datasets(self):
        device=torch.device("cuda:0")
        for name,data in (("Cora",node_data()),("CIFAR10",graphs(3))):
            cfg=full_config(name)
            width,layers=(16,1) if name=="Cora" else (110,4)
            for variant in cfg["variants"]:
                model=build_full_backbone(cfg,41,data)
                install_uniform(model,cfg["family"],41,fake_calibration(width,layers),variant)
                model.to(device)
                if cfg["task"]=="node":
                    training=data.clone().to(device)
                else:
                    from torch_geometric.loader import DataLoader
                    training=DataLoader(data["train"],batch_size=2)
                metrics=train_gpu_epoch(model,training,optimizer_for(model,cfg),cfg,device)
                self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA integration test requires host GPU access")
    def test_resume_restores_optimizer_rng_and_data_order_without_testing(self):
        cfg=dict(full_config("MNIST"),epochs=3,seeds=[41],variants=["main"],diagnostics=False)
        data=graphs()
        device=torch.device("cuda:0")
        with tempfile.TemporaryDirectory() as temp:
            uninterrupted,interrupted=Path(temp)/"normal",Path(temp)/"resumed"
            expected=fit(uninterrupted,cfg,"main",41,data,device)
            actual_epoch=train_gpu_epoch
            calls=0
            def interrupt_after_first(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise InterruptedError("Simulated epoch boundary interruption")
                return actual_epoch(*args,**kwargs)
            with patch("runs.lorap_study.full_run.train_gpu_epoch",side_effect=interrupt_after_first):
                with self.assertRaises(InterruptedError):
                    fit(interrupted,cfg,"main",41,data,device)
            checkpoint=interrupted/"MNIST/main/seed_41/last.pt"
            self.assertEqual(torch.load(checkpoint,weights_only=False)["epoch"],1)
            result=fit(interrupted,cfg,"main",41,data,device)
            self.assertEqual(result["test_evaluations"],0)
            self.assertIsNone(result["test"])
            self.assertAlmostEqual(result["validation"]["loss"],expected["validation"]["loss"],places=4)
            a=torch.load(uninterrupted/"MNIST/main/seed_41/last.pt",weights_only=False)
            b=torch.load(checkpoint,weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key],b["model"][key],atol=1e-5,rtol=1e-4)
            self.assertEqual(len(b["history"]),3)
            for name in a["rng"]["loaders"]:
                self.assertTrue(torch.equal(a["rng"]["loaders"][name],b["rng"]["loaders"][name]))
            with self.assertRaises(RuntimeError):
                test_once(interrupted,cfg,"main",41,data,device)
            write_json(interrupted/"MNIST/test_release.json",dict(config_sha256=sha_config(cfg),fits=[result]))
            tested=test_once(interrupted,cfg,"main",41,data,device)
            self.assertEqual(tested["test_evaluations"],1)
            with patch("runs.lorap_study.full_run.evaluate",side_effect=AssertionError("Repeated test")):
                self.assertEqual(test_once(interrupted,cfg,"main",41,data,device),tested)


if __name__=="__main__":
    unittest.main()

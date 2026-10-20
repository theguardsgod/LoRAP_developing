"""Artifact-only checks for completed GPU fits; never evaluates the test set."""

import json
from pathlib import Path

import torch

from .run import write_json
from .full_validation import validation_reproduction, verify_checkpoint


def audit_results(root,results):
    root=Path(root)
    paired={}
    replays=[]
    for result in results:
        directory=root/result["dataset"]/result["variant"]/f"seed_{result['seed']}"
        assert result["device"].startswith("cuda") and result["gpu_name"]
        logs=[json.loads(line) for line in (directory/"epochs.jsonl").read_text().splitlines()]
        assert len(logs)==result["epochs"]
        assert [r["epoch"] for r in logs]==list(range(1,result["epochs"]+1))
        best=min(logs,key=lambda row:row["validation"]["loss"])
        assert best["epoch"]==result["best_epoch"]
        replay=validation_reproduction(best["validation"]["loss"],result["validation"]["loss"])
        replays.append(dict(dataset=result["dataset"],variant=result["variant"],seed=result["seed"],**replay))
        checkpoint=torch.load(directory/"best.pt",weights_only=False,map_location="cpu")
        verify_checkpoint(directory,checkpoint,result["config_sha256"])
        assert checkpoint["epoch"]==best["epoch"]
        assert checkpoint["validation_loss"]==best["validation"]["loss"]
        assert all(torch.isfinite(t).all() for t in checkpoint["model"].values())
        key=(result["dataset"],result["seed"])
        if key in paired:
            assert paired[key]==result["backbone_sha256"]
        paired[key]=result["backbone_sha256"]
        if result["test"] is not None:
            assert result["test_evaluations"]==1 and result["test_device"].startswith("cuda")
            assert (directory/"test_started.json").exists()
    verified=dict(fits_verified=len(results),test_results_verified=sum(r["test_evaluations"] for r in results),
                  all_training_cuda=True,all_checkpoints_selected_by_validation=True,
                  all_paired_backbones_equal=True,all_best_weights_match_committed_checkpoint=True,
                  validation_reproductions=replays,additional_test_evaluations=0)
    write_json(root/"verification.json",verified)
    return verified

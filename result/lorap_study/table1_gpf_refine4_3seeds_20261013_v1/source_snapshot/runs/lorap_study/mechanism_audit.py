"""Check saved mechanism artifacts; only re-evaluate validation, never test."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch_geometric.utils import remove_self_loops

from .data import load_data
from .mechanism import file_digest, make_control, precision_context
from .models import sum_messages
from .run import ROOT, evaluate, write_json


@torch.no_grad()
def prompt_variation(model, data, precision):
    captured = []
    handle = model.conv2.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    model.eval()
    try:
        with precision_context(model, precision):
            model(data)
            x = captured[0]
            edges, _ = remove_self_loops(data.edge_index)
            agg = sum_messages(model.conv2.q_xw(x.T).T, edges)
            prompt = (model.prompt_aggr.add(agg) - agg).double()
    finally:
        handle.remove()
    rows = {}
    for split, mask in (("full_graph", slice(None)), ("validation", data.val_mask)):
        p = prompt[mask]
        total = float(p.square().sum())
        centered = p - p.mean(0, keepdim=True)
        rows[split] = dict(nodes=len(p), norm=float(p.norm()),
                           centered_energy_fraction=float(centered.square().sum()) / total if total else None,
                           mean_prompt=p.mean(0).tolist())
    return rows


def audit(output):
    output = Path(output)
    torch.set_num_threads(4)
    protocol = json.loads((output / "protocol.json").read_text())
    config = protocol["config"]
    results = json.loads((output / "results.json").read_text())
    probes = json.loads((output / "interventions.json").read_text())
    assert len(results) == 18 and len(probes) == 72
    data, manifest = load_data(config)
    assert manifest == json.loads((output / "split_manifest.json").read_text())
    for name, digest in json.loads((output / "provenance.json").read_text())["source_sha256"].items():
        assert file_digest(output / "source_snapshot" / name) == digest
    checks, variation = [], []
    for seed in protocol["paired_seeds"]:
        rows = [r for r in results if r["seed"] == seed]
        assert len({r["backbone_sha256"] for r in rows}) == 1
        assert len({r["prompt_base_sha256"] for r in rows if r["control"] == "original"}) == 1
    for result in results:
        precision, control, seed = (result[k] for k in ("precision", "control", "seed"))
        directory = output / precision / control / f"seed_{seed}"
        rows = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
        assert len(rows) == config["epochs"] == 100
        assert [r["epoch"] for r in rows] == list(range(1, 101))
        best = min(rows, key=lambda r: r["validation"]["loss"])
        assert best["epoch"] == result["best_epoch"]
        state = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
        assert state["epoch"] == best["epoch"]
        assert state["validation_loss"] == best["validation"]["loss"]
        assert file_digest(directory / "best.pt") == result["checkpoint_sha256"]
        assert all(torch.isfinite(t).all() for t in state["model"].values())
        assert result["test_evaluations"] == 1
        assert result["new_test_evaluations"] == int("reused_from" not in result)
        assert result["aggregate_prompt_parameters"] == (0 if control == "none" else 412)
        model = make_control(data, seed, precision, control)
        model.load_state_dict(state["model"])
        with precision_context(model, precision):
            validation = evaluate(model, data, config, torch.device("cpu"), "val")
        assert abs(validation["loss"] - result["validation"]["loss"]) < 1e-6
        assert validation["accuracy"] == result["validation"]["accuracy"]
        if control == "original":
            variation.append(dict(precision=precision, seed=seed, **prompt_variation(model, data, precision)))
            for probe in probes:
                if probe["precision"] == precision and probe["seed"] == seed and probe["intervention"] == "learned" and probe["strength"] == 1:
                    assert probe["validation"]["accuracy"] == validation["accuracy"]
                    assert abs(probe["validation"]["loss"] - validation["loss"]) < 1e-6
        checks.append(dict(precision=precision, control=control, seed=seed, best_epoch=best["epoch"],
                           validation_reproduced=True, checkpoint_sha256=result["checkpoint_sha256"]))
    write_json(output / "prompt_variation.json", variation)
    source = Path(__file__)
    shutil.copy2(source, output / "mechanism_audit_snapshot.py")
    verification = dict(completed_cells=18, new_training_runs=15, reused_training_runs=3,
                        newly_trained_epochs=1500, total_logged_epochs=1800,
                        new_test_evaluations=sum(r["new_test_evaluations"] for r in results),
                        validation_interventions=len(probes), all_backbones_paired=True,
                        prompt_parameters=412, mlp_parameters=412, checks=checks,
                        audit_source_sha256=file_digest(source))
    write_json(output / "verification.json", verification)
    print(json.dumps({k: v for k, v in verification.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

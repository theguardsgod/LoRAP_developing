"""Reproduce initialization and validation without re-evaluating any test set."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .data import load_data, loaders_for
from .improved import bare_model
from .models import model_counts, state_digest
from .run import ROOT, evaluate, write_json
from .uniform import install_uniform
from .uniform_context import inspect_conditions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    output = Path(output).resolve()
    torch.set_num_threads(4)
    config = json.loads((output / "protocol.json").read_text())["config"]
    results = json.loads((output / "results.json").read_text())
    before_test = json.loads((output / "pre_test_results.json").read_text())
    expected = {(v, s) for v in config["variants"] for s in config["seeds"]}
    assert len(results) == len(before_test) == len(expected)
    assert {(r["variant"], r["seed"]) for r in results} == expected
    assert {(r["variant"], r["seed"]) for r in before_test} == expected
    assert all(r["test_evaluations"] == 0 and r["test"] is None for r in before_test)
    sources = json.loads((output / "provenance.json").read_text())["source_sha256"]
    for name, digest in sources.items():
        assert sha(output / "source_snapshot" / name) == digest, name
        # Source snapshot includes all modules used for fitting and calibration.
        assert sha(ROOT / name) == digest, name
    data, manifest = load_data(config)
    assert manifest == json.loads((output / "split_manifest.json").read_text())
    expected_counts = (dict(none=0, original=2480, main=3345, nonzero_start=3345,
                           raw_conditions=3345, no_conditions=3260, spectral_basis=3345)
                       if config["dataset"] == "MNIST" else
                       dict(none=0, original=412, main=497, nonzero_start=497,
                            raw_conditions=497, no_conditions=412, spectral_basis=497))
    checks = []
    for result in results:
        dataset, variant, seed = (result[k] for k in ("dataset", "variant", "seed"))
        directory = output / variant / f"seed_{seed}"
        assert result == json.loads((directory / "result.json").read_text())
        pre = next(r for r in before_test if (r["variant"], r["seed"]) == (variant, seed))
        assert {k: v for k, v in result.items() if k not in ("test", "test_evaluations")} == {
            k: v for k, v in pre.items() if k not in ("test", "test_evaluations")}
        calibration = torch.load(output / f"calibration_seed_{seed}/calibration.pt", weights_only=False)
        for info in calibration:
            assert info["split"] == "train" and info["sampled_nodes"] <= config["calibration_max_nodes"]
            assert info["calibration_seed"] == seed
            assert torch.isfinite(info["context_mean"]).all()
            assert torch.isfinite(info["context_scale"]).all() and (info["context_scale"] > 0).all()
            if "spectral_basis" not in config["variants"]:
                assert info["fp_reference_used"] is False and "basis" not in info
            if dataset == "CiteSeer":
                assert info["sampled_nodes"] <= int(data.train_mask.sum())
        model = bare_model(dataset, seed, data if dataset == "CiteSeer" else None)
        ranks = install_uniform(model, dataset, seed, calibration, variant)
        assert ranks == result["ranks"]
        assert ranks == ([2]*(4 if dataset == "MNIST" else 1) if variant != "none" else [])
        assert state_digest(model) == result["backbone_sha256"]
        assert state_digest(model, "all") == result["initial_state_sha256"]
        assert model_counts(model)["aggregate_prompt_parameters"] == result["aggregate_prompt_parameters"] == expected_counts[variant]
        logs = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
        assert len(logs) == config["epochs"] == result["epochs"]
        assert [row["epoch"] for row in logs] == list(range(1, config["epochs"]+1))
        best = min(logs, key=lambda row: row["validation"]["loss"])
        checkpoint = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
        assert checkpoint["epoch"] == result["best_epoch"] == best["epoch"]
        assert checkpoint["validation_loss"] == best["validation"]["loss"]
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        model.load_state_dict(checkpoint["model"])
        validation = data if dataset == "CiteSeer" else loaders_for(data, config, seed)["val"]
        checked = evaluate(model, validation, config, torch.device("cpu"), "val")
        assert abs(checked["loss"]-result["validation"]["loss"]) < 1e-6
        assert checked["accuracy"] == result["validation"]["accuracy"]
        if variant == "main":
            conditions = inspect_conditions(model, validation, config)
            assert conditions["validation"] == checked
            write_json(directory / "condition_drift.json", conditions)
        assert result["test_evaluations"] == 1 and result["test"] is not None
        diagnostic = json.loads((directory / "diagnostics.json").read_text())
        if config["diagnostics"]:
            assert diagnostic["split"] == "validation"
            assert all(layer["sampled_nodes"] <= config["diagnostic_max_nodes"] for layer in diagnostic["layers"].values())
        else:
            assert diagnostic == {"enabled": False}
            assert not (directory / "full_update_diagnostics.json").exists()
        checks.append(dict(variant=variant, seed=seed, validation_reproduced=True,
                           checkpoint_sha256=sha(directory / "best.pt")))
    for seed in config["seeds"]:
        assert len({r["backbone_sha256"] for r in results if r["seed"] == seed}) == 1
    summary = dict(training_runs=len(results), training_epochs=sum(r["epochs"] for r in results),
                   final_test_evaluations=sum(r["test_evaluations"] for r in results),
                   all_training_completed_before_test=True, all_initializations_reproduced=True,
                   all_best_checkpoints_verified=True, all_validation_metrics_reproduced=True,
                   all_source_snapshots_verified=True, all_ranks_uniform=True,
                   condition_audit_source_sha256=sha(Path(__file__).with_name("uniform_context.py")),
                   audit_source_sha256=sha(__file__), checks=checks)
    test_metadata = output / "unit_tests.json"
    if test_metadata.exists():
        unit_tests = json.loads(test_metadata.read_text())
        assert unit_tests["passed"] is True
        assert sha(output / "unit_tests.log") == unit_tests["log_sha256"]
        for name, digest in unit_tests["source_sha256"].items():
            assert sha(output / "validation_source" / name) == digest
            assert sha(ROOT / name) == digest
        summary["unit_tests_passed"] = unit_tests["tests"]
    shutil.copy2(__file__, output / "uniform_audit_snapshot.py")
    shutil.copy2(Path(__file__).with_name("uniform_context.py"), output / "uniform_context_snapshot.py")
    write_json(output / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

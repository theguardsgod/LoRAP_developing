"""Audit improved LoRAP artifacts without any new test-set evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .data import load_data, loaders_for
from .improved import allocate_ranks, bare_model, install_prompt
from .models import model_counts, state_digest
from .run import ROOT, evaluate, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    output = Path(output)
    torch.set_num_threads(4)
    protocol = json.loads((output / "protocol.json").read_text())
    configs = {c["dataset"]: c for c in protocol["configs"]}
    results = json.loads((output / "results.json").read_text())
    sources = json.loads((output / "provenance.json").read_text())["source_sha256"]
    for name, digest in sources.items():
        assert sha(output / "source_snapshot" / name) == digest
    for name in ("improved.py", "improve_run.py", "models.py", "prompts.py", "run.py", "config.py", "data.py"):
        relative = "runs/lorap_study/" + name
        assert sha(ROOT / relative) == sources[relative], relative
    datasets = {name: load_data(config)[0] for name, config in configs.items()}
    selection = None
    if (output / "CiteSeer/selection.json").exists():
        selection = json.loads((output / "CiteSeer/selection.json").read_text())
        assert selection["test_results_seen"] == 0
        scores = selection["validation_scores"]
        assert selection["selected_variant"] == min(selection["candidates"], key=lambda name: scores[name])
    verification = []
    for result in results:
        dataset, seed, variant = (result[k] for k in ("dataset", "seed", "variant"))
        config, data = configs[dataset], datasets[dataset]
        directory = output / dataset / variant / f"seed_{seed}"
        calibration = torch.load(output / dataset / f"calibration_seed_{seed}/calibration.pt", weights_only=False)
        assert all(info["split"] == "train" and info["sampled_nodes"] <= 4096 for info in calibration)
        if dataset == "CiteSeer":
            assert calibration[0]["sampled_nodes"] <= int(data.train_mask.sum())
        model = bare_model(dataset, seed, data if dataset == "CiteSeer" else None)
        ranks = install_prompt(model, dataset, variant, seed, calibration)
        assert ranks == result["ranks"]
        assert state_digest(model) == result["backbone_sha256"]
        assert state_digest(model, "all") == result["initial_state_sha256"]
        assert model_counts(model)["aggregate_prompt_parameters"] == result["aggregate_prompt_parameters"]
        if variant == "adaptive_spectral":
            assert ranks == allocate_ranks(calibration, 2*len(calibration))
            allocated = sum(float(c["normalized_energy"][:rank].sum()) for c, rank in zip(calibration, ranks))
            uniform = sum(float(c["normalized_energy"][:2].sum()) for c in calibration)
            assert allocated >= uniform - 1e-10
        logs = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
        assert len(logs) == config["epochs"]
        assert [row["epoch"] for row in logs] == list(range(1, config["epochs"]+1))
        best = min(logs, key=lambda row: row["validation"]["loss"])
        state = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
        assert state["epoch"] == best["epoch"] == result["best_epoch"]
        assert state["validation_loss"] == best["validation"]["loss"]
        assert all(torch.isfinite(t).all() for t in state["model"].values())
        model.load_state_dict(state["model"])
        validation = data if dataset == "CiteSeer" else loaders_for(data, config, seed)["val"]
        checked = evaluate(model, validation, config, torch.device("cpu"), "val")
        assert abs(checked["loss"] - result["validation"]["loss"]) < 1e-6
        assert checked["accuracy"] == result["validation"]["accuracy"]
        if selection is not None and dataset == "CiteSeer":
            expected_tests = int(variant == selection["selected_variant"])
            assert result["test_evaluations"] == expected_tests
            assert (result["test"] is not None) == bool(expected_tests)
        else:
            assert result["test_evaluations"] == 1
        verification.append(dict(dataset=dataset, variant=variant, seed=seed,
                                 validation_reproduced=True, ranks=ranks,
                                 calibration_buffer_bytes=sum(b.numel()*b.element_size() for b in model.prompt_aggr.buffers()),
                                 checkpoint_sha256=sha(directory / "best.pt")))
    for name in configs:
        for seed in configs[name]["seeds"]:
            rows = [r for r in results if r["dataset"] == name and r["seed"] == seed]
            assert len({r["backbone_sha256"] for r in rows}) == 1
            if name == "MNIST":
                ranks = [r for r in rows if r["variant"] in ("stable_spectral", "adaptive_spectral")]
                if len(ranks) == 2:
                    assert {r["aggregate_prompt_parameters"] for r in ranks} == {3345}
                    assert {sum(r["ranks"]) for r in ranks} == {8}
    source = Path(__file__)
    shutil.copy2(source, output / "improve_audit_snapshot.py")
    summary = dict(training_runs=len(results), training_epochs=sum(r["epochs"] for r in results),
                   final_test_evaluations=sum(r["test_evaluations"] for r in results),
                   rejected_candidates_not_tested=sum(r["test_evaluations"] == 0 for r in results),
                   all_initializations_reproduced=True, all_best_checkpoints_verified=True,
                   all_validation_metrics_reproduced=True, calibration_training_only=True,
                   audit_source_sha256=sha(source), checks=verification)
    write_json(output / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

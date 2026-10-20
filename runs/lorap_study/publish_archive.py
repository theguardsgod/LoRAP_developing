"""Build a portable, checksummed publication archive without modifying runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "result/lorap_study"
PRIMARY = [
    "lora_joint_qat_20260910_v1",
    "table_v_lora_full_20260911_v1",
    "table_v_control_full_20260915_v1",
    "glora_full_20260916_v1",
    "graphlora_full_20260918_v1",
    "table1_frozen_20260914_v2",
    "table1_graphlora_3seeds_20260925_v1",
    "table1_gpf_rerun_3seeds_20261001_v1",
    "table1_gpf_search_3seeds_20261007_v1",
    "table1_gpf_refine_3seeds_20261011_v1",
    "table1_gpf_refine3_3seeds_20261013_v1",
    "table1_gpf_refine4_3seeds_20261013_v1",
    "table1_complete_20261019_v1",
]
SUPPORT = [
    "lora_smoke_20260910_v2", "table_v_lora_smoke_20260911_v1",
    "table_v_control_smoke_20260915_v1", "glora_smoke_20260916_v2",
    "graphlora_smoke_20260917_v1", "graphlora_pilot_20260917_v1",
    "table1_graphlora_smoke_20260925_v1", "table1_gpf_rerun_smoke_20261001_v1",
    "table1_gpf_search_smoke_20261007_v1", "table1_gpf_refine_smoke_20261011_v1",
    "table1_gpf_refine3_smoke_20261013_v1", "table1_gpf_refine4_smoke_20261013_v1",
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(destination):
    destination = destination.resolve()
    if destination == REPO or REPO in destination.parents:
        raise ValueError("Use an independent staging directory outside the source repository")
    if any(p.name != ".git" for p in destination.iterdir()):
        raise ValueError("Staging directory must be empty except for .git")
    copied, omitted = {}, []

    def copy(path, relative=None):
        relative = relative or path.relative_to(REPO)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied[str(relative)] = {"bytes": target.stat().st_size, "sha256": sha(target)}

    for directory in ("runs/lorap_study", "quantize_function", "quantize_method", "prompt", "utils"):
        for path in sorted((REPO / directory).rglob("*")):
            if path.is_file() and not any(p.startswith(".") or p == "__pycache__" for p in path.relative_to(REPO).parts):
                if path.suffix in (".py", ".md", ".json", ".yml", ".yaml", ".sh", ".txt") or "LICENSE" in path.name:
                    copy(path)
    for path in sorted(REPO.iterdir()):
        if path.is_file() and path.suffix in (".py", ".sh", ".yml"):
            copy(path)
    copy(REPO / "README.md", Path("README_A2Q.md"))
    for path in (REPO / "docs").glob("*20261020*"):
        if path.is_file():
            copy(path)

    for name in PRIMARY + SUPPORT:
        root = RESULTS / name
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(root).parts
            reason = None
            if path.name.endswith((".lock", ".pyc")) or "__pycache__" in parts:
                reason = "runtime lock/cache"
            elif name == "table1_frozen_20260914_v2" and parts[0].startswith(("QAT_", "DQ_", "A2Q_")):
                # This archive needs the 135 paired controls, not all frozen-adapter searches.
                if len(parts) < 4 or parts[1] not in ("seed_41", "seed_42", "seed_43") or parts[2] != "baseline":
                    reason = "outside paired-control scope"
            if reason is None and "test_predictions" in parts:
                reason = "redundant per-batch prediction cache"
            if reason is None and path.suffix == ".pt":
                keep = name in PRIMARY and (
                    path.name == "best.pt" or "calibration" in path.name or "split" in path.name)
                if not keep:
                    reason = "optimizer/resume state or support-run binary; retained locally"
            if reason:
                omitted.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "reason": reason})
            else:
                copy(path)
    # Other early research is catalogued by its original summary only.
    catalog = []
    for root in sorted(RESULTS.iterdir()):
        if not root.is_dir():
            continue
        names = []
        for name in ("REPORT.md", "FINAL_REPORT.md", "status.json", "summary.csv", "completion_audit.json"):
            path = root / name
            if path.is_file():
                names.append(name)
                if root.name not in PRIMARY + SUPPORT:
                    copy(path, Path("archive/early_studies") / root.name / name)
        catalog.append({"study": root.name, "coverage": "primary" if root.name in PRIMARY else
                        "support metadata" if root.name in SUPPORT else "summary only", "summaries": names})
    manifest = {
        "version": 1, "created_date": "2026-10-20",
        "original_repository_root": str(REPO), "primary_studies": PRIMARY, "support_studies": SUPPORT,
        "policy": "Original artifact bytes preserved. Best checkpoints included for primary studies; "
                  "resume states, prediction caches and datasets excluded. Frozen study includes paired controls only.",
        "files": copied, "omitted": omitted, "catalog": catalog,
    }
    (destination / "archive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(copied), "MiB": sum(x["bytes"] for x in copied.values()) / 1024**2,
                      "omitted_files": len(omitted), "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination)

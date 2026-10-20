"""Verify uploaded bytes and final Table I statistics without loading test data."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "archive_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        assert path.stat().st_size == expected["bytes"], relative
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"], relative
    provenance = json.loads((root / "result/lorap_study/table1_complete_20261019_v1/provenance.json").read_text())
    old_root = Path(manifest["original_repository_root"])
    records = 0
    for cell in provenance:
        if not cell["artifacts"]:
            continue
        values = []
        for artifact in cell["artifacts"]:
            test_path = root / Path(artifact["test_file"]).relative_to(old_root)
            test = json.loads(test_path.read_text())
            done = json.loads(test_path.with_name("done.json").read_text())
            assert test["completed_test_evaluations"] == 1
            assert test["checkpoint_file_sha256"] == done["checkpoint_file_sha256"]
            with test_path.with_name("best.pt").open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == test["checkpoint_file_sha256"]
            values.append(test["metric"])
            records += 1
        assert len(values) == 3
        assert math.isclose(statistics.mean(values), cell["mean"], abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values), cell["sd"], abs_tol=1e-12)
    result = {"passed": True, "archived_files": len(manifest["files"]),
              "table_rows": 39, "metric_cells": len(provenance),
              "verified_final_test_records_and_checkpoints": records,
              "test_inferences_executed": 0}
    assert len(provenance) == 195 and records == 165
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()

"""Exact checkpoint identity checks, separate from CUDA evaluation drift."""

import json
import math
from pathlib import Path

import torch


# Fixed before releasing the remaining tests. Repeated validation of identical
# CUDA weights showed up to 2.10e-4 CE drift; selection itself remains exact.
VALIDATION_REPLAY_ATOL = 5e-4


def validation_reproduction(selected_loss, replay_loss):
    selected_loss, replay_loss = float(selected_loss), float(replay_loss)
    delta = replay_loss - selected_loss
    if not all(math.isfinite(x) for x in (selected_loss, replay_loss)):
        raise AssertionError("Non-finite checkpoint validation loss")
    if abs(delta) > VALIDATION_REPLAY_ATOL:
        raise AssertionError(f"Validation replay CE drift {delta:.8g} exceeds {VALIDATION_REPLAY_ATOL}")
    return dict(selected_loss=selected_loss, replay_loss=replay_loss,
                delta=delta, absolute_tolerance=VALIDATION_REPLAY_ATOL)


def verify_checkpoint(directory, checkpoint, config_hash=None):
    """Require the exact committed minimum-validation checkpoint, not nearby weights."""
    directory = Path(directory)
    logs = [json.loads(line) for line in (directory/"epochs.jsonl").read_text().splitlines()]
    last = torch.load(directory/"last.pt", weights_only=False, map_location="cpu")
    assert logs and logs == last["history"], "Epoch log differs from committed history"
    assert [row["epoch"] for row in logs] == list(range(1, last["epoch"] + 1))
    assert all(math.isfinite(row["validation"]["loss"]) for row in logs)
    if config_hash is not None:
        assert last["config_sha256"] == config_hash, "Checkpoint configuration changed"
    selected = min(logs, key=lambda row: row["validation"]["loss"])
    for saved in (checkpoint, last["best"]):
        assert saved["epoch"] == selected["epoch"], "Checkpoint is not earliest validation minimum"
        assert saved["validation_loss"] == selected["validation"]["loss"]
    assert checkpoint["model"].keys() == last["best"]["model"].keys()
    for name, value in checkpoint["model"].items():
        assert torch.isfinite(value).all(), f"Non-finite checkpoint tensor: {name}"
        assert torch.equal(value.cpu(), last["best"]["model"][name]), f"Checkpoint tensor changed: {name}"
    return selected

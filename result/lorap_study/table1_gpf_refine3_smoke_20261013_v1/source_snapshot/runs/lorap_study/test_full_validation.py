"""Validation drift may be tolerated; changed weights or selection never are."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .full_validation import validation_reproduction, verify_checkpoint


class ValidationTests(unittest.TestCase):
    def test_bounded_drift_and_nonfinite_rejection(self):
        self.assertAlmostEqual(validation_reproduction(1.15, 1.15021)["delta"], .00021)
        for selected, replay in ((1., 1.001), (1., .999), (1., float("nan")), (float("inf"), 1.)):
            with self.assertRaises(AssertionError):
                validation_reproduction(selected, replay)

    def test_exact_identity_and_earliest_minimum(self):
        history = [dict(epoch=i, validation=dict(loss=loss)) for i, loss in enumerate((2., 1., 1.), 1)]
        best = dict(epoch=2, validation_loss=1., model={"weight": torch.tensor([.5])})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"epochs.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history))
            last = dict(epoch=3, history=history, best=best, config_sha256="expected")
            torch.save(last, root/"last.pt")
            self.assertEqual(verify_checkpoint(root, best, "expected")["epoch"], 2)
            changed = copy.deepcopy(best)
            changed["model"]["weight"].add_(1e-6)
            with self.assertRaises(AssertionError):
                verify_checkpoint(root, changed, "expected")
            with self.assertRaises(AssertionError):
                verify_checkpoint(root, dict(best, epoch=3), "expected")
            with self.assertRaises(AssertionError):
                verify_checkpoint(root, best, "different")
            history[-1]["validation"]["loss"] = .9
            (root/"epochs.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history))
            with self.assertRaises(AssertionError):
                verify_checkpoint(root, best, "expected")


if __name__ == "__main__":
    unittest.main()

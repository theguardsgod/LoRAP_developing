"""Test the two-protocol release barrier and paper-row reporting."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .run import write_json
from .table_v_run import MODES, coordinator, summarize, table_config


class TableVTests(unittest.TestCase):
    def prepare(self, root, smoke=False):
        suite = dict(seeds=[41, 42], epochs=200, devices=["cuda:0"], threads=4, smoke=smoke)
        write_json(root / "suite.json", suite)
        for mode in MODES:
            config = table_config(mode, suite["seeds"], suite["epochs"], smoke)
            write_json(root / mode / "MNIST/config.json", config)
            write_json(root / mode / "study.json", dict(configs=[config], smoke=smoke))

    def test_both_protocols_fit_before_any_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            events = []
            with patch("runs.lorap_study.table_v_run.phase_workers", side_effect=lambda r, p: events.append(p)), \
                 patch("runs.lorap_study.table_v_run.release_tests", side_effect=lambda r, c: events.append(r.name)), \
                 patch("runs.lorap_study.table_v_run.summarize"):
                coordinator(root)
            self.assertEqual(events, ["fit", "fixed4", "a2q", "test"])
            with patch("runs.lorap_study.table_v_run.phase_workers", side_effect=RuntimeError("failed fit")), \
                 patch("runs.lorap_study.table_v_run.release_tests") as release:
                with self.assertRaises(RuntimeError):
                    coordinator(root)
                release.assert_not_called()

    def test_unfinished_runs_never_generate_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            summarize(root)
            status = json.loads((root / "status.json").read_text())
            self.assertFalse(status["complete"])
            self.assertTrue(all(r["mean_accuracy_percent"] is None for r in status["rows"]))
            text = (root / "table_v_lora_rows.tex").read_text()
            self.assertIn("INT4 / QAT block", text)
            self.assertIn("A2Q block", text)
            self.assertEqual(text.count("17.29"), 2)
            self.assertEqual(text.count("TBD"), 3)
            self.assertIsNone(table_config("a2q", [41], 200)["subset_sizes"])
            self.assertEqual(table_config("fixed4", [41], 200)["a_loss"], 0)


if __name__ == "__main__":
    unittest.main()

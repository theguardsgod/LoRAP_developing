"""Check passive condition observation without changing predictions or state."""

import unittest

import torch

from .improved import bare_model
from .models import state_digest
from .run import evaluate
from .test_improved import fake_calibration
from .test_study import graph_data, node_data
from .uniform import install_uniform
from .uniform_context import inspect_conditions
from .uniform_run import primary_config


class ConditionAuditTests(unittest.TestCase):
    def test_passive_observer_preserves_model_and_validation_predictions(self):
        torch.set_num_threads(2)
        for dataset, data, width, layers in (("CiteSeer", node_data(), 16, 1), ("MNIST", graph_data(), 110, 4)):
            model = bare_model(dataset, 44, data if dataset == "CiteSeer" else None)
            install_uniform(model, dataset, 44, fake_calibration(width, layers))
            validation = data if dataset == "CiteSeer" else [data]
            config = primary_config(dataset)
            before = state_digest(model, "all")
            plain = evaluate(model, validation, config, torch.device("cpu"), "val")
            observed = inspect_conditions(model, validation, config)
            self.assertEqual(plain, observed["validation"])
            self.assertEqual(before, state_digest(model, "all"))
            self.assertNotIn("add_at", model.prompt_aggr.__dict__)
            self.assertEqual(len(observed["layers"]), layers)
            for values in observed["layers"].values():
                self.assertEqual(values["nodes"], 2 if dataset == "CiteSeer" else 10)
                self.assertEqual(values["gate_mean"], 1.)
                self.assertEqual(values["prompt_to_aggregate_norm"], 0.)
                self.assertTrue(all(0 <= x <= 1 for x in values["potential_clipped_fraction"]))


if __name__ == "__main__":
    unittest.main()

"""Run with unittest; MLP_PROJECT_ROOT optionally selects the other checkout."""
import contextlib
from copy import deepcopy
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/covid_mlp_checkpoint_matplotlib")

import numpy as np
import torch

ROOT = Path(os.environ.get("MLP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "checkpoint_mlp", ROOT / "MultilayerPerceptron/MLP_1_layer.py"
)
mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp)
torch.set_num_threads(1)


class BestCheckpointTest(unittest.TestCase):
    def run_fit(self, losses, patience, estimator=None):
        estimator = estimator or mlp.Perceptron(epochs=len(losses))
        x = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
        y = np.arange(20, 28, dtype=np.float32)
        snapshots = []
        calls = 0

        def evaluate(model, *args, **kwargs):
            nonlocal calls
            validation = calls % 2 == 0
            calls += 1
            if validation:
                snapshots.append(deepcopy(model.state_dict()))
                return losses[len(snapshots) - 1]
            return 1.0

        with patch.object(mlp, "eval_huber", side_effect=evaluate), \
                patch.object(mlp.plt, "savefig"), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                result = estimator.fit(
                    x, y, x, y, fold="synthetic_test", input_size=3,
                    hidden_size=4, dropout=0, batch_size=2, patience=patience,
                )
            finally:
                mlp.plt.close("all")
        return estimator, result, snapshots

    def assert_restored(self, estimator, result, snapshots, best):
        self.assertIs(result, estimator.model)
        self.assertFalse(result.training)
        for name, tensor in result.state_dict().items():
            torch.testing.assert_close(tensor, snapshots[best][name], rtol=0, atol=0)
        counters = [name for name in snapshots[best] if name.endswith("num_batches_tracked")]
        self.assertTrue(counters)
        for name in counters:
            self.assertLess(snapshots[best][name].item(), snapshots[-1][name].item())
        self.assertTrue(any(
            not torch.equal(snapshots[best][name], snapshots[-1][name])
            for name in snapshots[best] if name.endswith("weight")
        ))

    def test_restores_parameters_and_batchnorm_after_early_stopping(self):
        estimator = mlp.Perceptron(epochs=10)
        estimator, result, states = self.run_fit([3, 1, 2, 4], 2, estimator)
        self.assertEqual(len(states), 4)
        self.assert_restored(estimator, result, states, best=1)

    def test_restores_best_checkpoint_at_epoch_limit(self):
        estimator, result, states = self.run_fit([3, 1, 2], patience=10)
        self.assertEqual(len(states), 3)
        self.assert_restored(estimator, result, states, best=1)

    def test_existing_model_also_restores_best_checkpoint(self):
        estimator, _, _ = self.run_fit([3, 1, 2], patience=10)
        estimator, result, states = self.run_fit([3, 1, 2], 10, estimator)
        self.assert_restored(estimator, result, states, best=1)

    def test_missing_valid_checkpoint_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "valid validation checkpoint"):
            self.run_fit([float("nan")] * 2, patience=10)


if __name__ == "__main__":
    unittest.main()

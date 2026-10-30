import unittest

import numpy as np
import torch

from sequence_model import baseline, config
from sequence_model.models import MODEL_NAMES, build_model
from sequence_model.train import class_weights, next_weight_path


class ModelTest(unittest.TestCase):
    def test_output_shape_for_both_window_lengths(self):
        for name in MODEL_NAMES:
            for frames in (60, 90):
                model = build_model(name, 11, config.N_CLASSES)
                self.assertEqual(model(torch.zeros(4, frames, 11)).shape, (4, config.N_CLASSES))

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            build_model("gru", 11, config.N_CLASSES)


class ClassWeightTest(unittest.TestCase):
    def test_rare_class_weighs_more(self):
        weights = class_weights(np.array([0] * 8 + [1] * 2 + [2] * 2))
        self.assertGreater(weights[1], weights[0])

    def test_missing_class_does_not_break(self):
        self.assertTrue(torch.isfinite(class_weights(np.array([0, 0, 1]))).all())


class WeightVersionTest(unittest.TestCase):
    def test_first_and_next(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            self.assertEqual(next_weight_path(out).name, "best_model_v01.pt")
            (out / "best_model_v09.pt").touch()
            (out / "best_model_v02.pt").touch()
            self.assertEqual(next_weight_path(out).name, "best_model_v10.pt")


class BaselineTest(unittest.TestCase):
    def test_probability_has_every_class_even_if_train_lacks_one(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 60, 11)).astype(np.float32)
        y = np.array([0, 2] * 10)
        for name in baseline.BASELINE_NAMES:
            proba = baseline.predict_proba(baseline.fit_baseline(name, X, y), X)
            self.assertEqual(proba.shape, (20, config.N_CLASSES))
            self.assertTrue((proba[:, 1] == 0).all())
            np.testing.assert_allclose(proba.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()

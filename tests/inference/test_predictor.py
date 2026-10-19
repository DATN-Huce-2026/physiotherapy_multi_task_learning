import unittest
from pathlib import Path

import numpy as np

from rehab_mtl.inference.predictor import load_model, predict

CHECKPOINT = Path(__file__).resolve().parents[2] / "artifacts" / "model_v6" / "best_model.pt"


@unittest.skipUnless(CHECKPOINT.is_file(), "cần artifacts/model_v6/best_model.pt")
class TestPredictor(unittest.TestCase):
    def test_model_v6_tra_6_head_phan_loai_va_quality(self):
        model = load_model(CHECKPOINT)
        rng = np.random.default_rng(0)
        keraal = rng.normal(size=(40, 25, 3)).astype(np.float32)
        kimore = rng.normal(size=(120, 25, 3)).astype(np.float32)

        probabilities, quality = predict(model, keraal, kimore)

        self.assertEqual(
            set(probabilities),
            {"exercise", "exercise_error_class", "global_evaluation",
             "error_type", "severity", "body_part"},
        )
        for values in probabilities.values():
            self.assertAlmostEqual(float(values.sum()), 1.0, places=5)
        self.assertIsNotNone(quality)
        self.assertTrue(0.0 <= quality <= 1.0)


if __name__ == "__main__":
    unittest.main()

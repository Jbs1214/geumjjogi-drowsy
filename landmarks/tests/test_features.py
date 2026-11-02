import unittest

import numpy as np
import pandas as pd

from landmarks.features import FEATURE_COLUMNS, add_features
from landmarks.points import LEFT_EYE, MOUTH, POINTS, RIGHT_EYE


def table(left_open: float, right_open: float) -> pd.DataFrame:
    """첫 행은 눈 세로 간격이 주어진 얼굴, 둘째 행은 얼굴 미검출인 표."""
    row = {"t": 0.0, "face_found": True}
    for prefix, gap in (("le", left_open), ("re", right_open)):
        for i, (x, y) in enumerate(((0, 0), (1, gap), (3, gap), (4, 0), (3, -gap), (1, -gap)), 1):
            row[f"{prefix}{i}_x"], row[f"{prefix}{i}_y"] = x, y
    for name, (x, y) in {"mouth_l": (0, 0), "mouth_r": (4, 0), "mouth_t": (2, 1), "mouth_b": (2, -1), "nose": (2, 2)}.items():
        row[f"{name}_x"], row[f"{name}_y"] = x, y
    missing = {key: np.nan for key in row} | {"t": 1.0, "face_found": False}
    return pd.DataFrame([row, missing])


class FeaturesTest(unittest.TestCase):
    def test_columns_match_auto_label_order(self):
        self.assertEqual(FEATURE_COLUMNS, ("face", "ear_l", "ear_r", "ear", "mar", "pitch", "yaw", "roll",
                                           "blink_l", "blink_r", "jaw_open"))

    def test_eye_sides_match_auto_label(self):
        self.assertEqual(LEFT_EYE, (362, 385, 387, 263, 373, 380))
        self.assertEqual(RIGHT_EYE, (33, 160, 158, 133, 153, 144))
        self.assertEqual(POINTS["le1"], 362)
        self.assertEqual(POINTS["re1"], 33)

    def test_mouth_points_match_auto_label(self):
        self.assertEqual(tuple(MOUTH[k] for k in ("mouth_l", "mouth_r", "mouth_t", "mouth_b")), (61, 291, 13, 14))

    def test_add_features_values(self):
        out = add_features(table(left_open=1.0, right_open=0.5))
        self.assertAlmostEqual(out["ear_l"][0], 0.5)
        self.assertAlmostEqual(out["ear_r"][0], 0.25)
        self.assertAlmostEqual(out["ear"][0], 0.375)
        self.assertAlmostEqual(out["mar"][0], 0.5)

    def test_missing_face(self):
        out = add_features(table(1.0, 1.0))
        self.assertEqual(out["face"].tolist(), [1.0, 0.0])
        self.assertTrue(out.loc[1, ["ear_l", "ear_r", "ear", "mar"]].isna().all())


if __name__ == "__main__":
    unittest.main()

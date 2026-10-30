import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model import config
from sequence_model.windows import (build_dataset, fill_missing, fit_normalizer, make_windows, normalize, resample,
                                    window_label)


def frame_table(n: int, fps: float = config.FPS, state: int = 0) -> pd.DataFrame:
    df = pd.DataFrame(np.ones((n, len(config.FEATURE_COLUMNS))), columns=list(config.FEATURE_COLUMNS))
    df.insert(0, "t", np.arange(n) / fps)
    df["state"] = state
    return df


class MakeWindowsTest(unittest.TestCase):
    def test_four_second_clip_gives_five_windows(self):
        # 기획서 3-3 예시: 4초 클립, 2초 창 → 0.0~2.0 … 2.0~4.0초로 5개, 모양 (60 × 11)
        X, y, starts = make_windows(frame_table(120))
        self.assertEqual(X.shape, (5, 60, 11))
        self.assertEqual(starts.tolist(), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(len(y), 5)

    def test_three_second_window(self):
        X, _, _ = make_windows(frame_table(120), window_sec=3.0)
        self.assertEqual(X.shape, (3, 90, 11))

    def test_clip_shorter_than_window_is_empty(self):
        X, y, starts = make_windows(frame_table(59))
        self.assertEqual(X.shape, (0, 60, 11))
        self.assertEqual((len(y), len(starts)), (0, 0))

    def test_exact_length_clip_gives_one_window(self):
        self.assertEqual(len(make_windows(frame_table(60))[0]), 1)


class WindowLabelTest(unittest.TestCase):
    def test_half_closed_is_closed(self):
        self.assertEqual(window_label(np.array([1, 1, 0, 0])), 1)

    def test_under_ratio_is_normal(self):
        self.assertEqual(window_label(np.array([1, 0, 0, 0])), 0)

    def test_yawn(self):
        self.assertEqual(window_label(np.array([2, 2, 2, 0])), 2)

    def test_mostly_uncertain(self):
        self.assertEqual(window_label(np.array([3, 3, 3, 0])), config.UNCERTAIN)


class ResampleTest(unittest.TestCase):
    def test_same_fps_keeps_frames(self):
        df = frame_table(120)
        df["ear"] = np.arange(120.0)
        out = resample(df, 30)
        self.assertEqual(len(out), 120)
        np.testing.assert_array_equal(out["ear"], df["ear"])

    def test_webcam_15fps_becomes_same_length(self):
        self.assertEqual(len(resample(frame_table(60, fps=15), 30)), 120)

    def test_downsample(self):
        self.assertEqual(len(resample(frame_table(120), 15)), 60)

    def test_does_not_invent_values(self):
        df = frame_table(60, fps=15)
        df["state"] = np.arange(60) % 2
        self.assertTrue(set(resample(df, 30)["state"]) <= {0, 1})

    def test_uneven_webcam_timestamps(self):
        df = frame_table(100, fps=30)
        df["t"] = np.cumsum(np.tile([0.02, 0.05], 50))
        out = resample(df, 30)
        self.assertAlmostEqual(out["t"].diff().median(), 1 / 30)


class FillMissingTest(unittest.TestCase):
    def setUp(self):
        self.df = frame_table(4)
        self.df.loc[1, "ear"] = np.nan

    def test_nan_becomes_zero_with_face_flag(self):
        out = fill_missing(self.df)
        self.assertEqual(out.loc[1, "ear"], 0.0)
        self.assertEqual(out.loc[1, "mar"], 0.0)
        self.assertEqual(out["face"].tolist(), [1.0, 0.0, 1.0, 1.0])
        self.assertFalse(out[list(config.FEATURE_COLUMNS)].isna().any().any())

    def test_undetected_normal_becomes_uncertain(self):
        self.assertEqual(fill_missing(self.df).loc[1, "state"], config.UNCERTAIN)

    def test_lost_close_label_is_kept(self):
        self.df.loc[1, "state"] = 1
        self.assertEqual(fill_missing(self.df).loc[1, "state"], 1)

    def test_face_zero_counts_as_missing(self):
        self.df.loc[2, "face"] = 0.0
        self.assertEqual(fill_missing(self.df).loc[2, "ear"], 0.0)

    def test_works_without_state(self):
        fill_missing(self.df.drop(columns="state"))


class BuildDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def write_clip(self, clip_id: str, X: list, y: list) -> None:
        data = {"X": X, "y": y, "segments": [], "params": {}}
        (self.root / f"{clip_id}_자동라벨.json").write_text(json.dumps(data), encoding="utf-8")

    def test_short_uncertain_and_undetected_clips(self):
        row = [1.0] * 11
        self.write_clip("SGA2100001S0001", [row] * 59, [0] * 59)                    # 창보다 짧음 → 제외
        self.write_clip("SGA2100002S0001", [row] * 60, [3] * 60)                    # 전부 불확실 → 창 제외
        self.write_clip("SGA2100003S0001", [[None] * 11] * 60, [0] * 60)            # 전부 미검출 → 불확실이 되어 창 제외
        self.write_clip("SGA2100004S0001", [row] * 40 + [[None] * 11] * 20, [1] * 40 + [0] * 20)   # 일부 미검출
        dataset = build_dataset(self.root)
        self.assertEqual(dataset.X.shape, (1, 60, 11))
        self.assertEqual(dataset.y.tolist(), [1])
        self.assertEqual(dataset.info["clip_id"].tolist(), ["SGA2100004S0001"])
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(dataset.X[0, 39:41, 0].tolist(), [1.0, 0.0])

    def test_empty_folder(self):
        with self.assertRaises(FileNotFoundError):
            build_dataset(self.root)


class NormalizeTest(unittest.TestCase):
    def test_uses_detected_frames_only(self):
        X = np.zeros((1, 4, 11), dtype=np.float32)
        X[0, :2, 0] = 1.0       # 앞 2프레임만 검출
        X[0, :2, 1] = [2.0, 4.0]
        mean, std = fit_normalizer(X)
        self.assertAlmostEqual(float(mean[1]), 3.0)
        self.assertAlmostEqual(float(std[1]), 1.0)

    def test_face_channel_and_missing_frames(self):
        X = np.zeros((1, 4, 11), dtype=np.float32)
        X[0, :2, 0] = 1.0
        X[0, :2, 1] = [2.0, 4.0]
        out = normalize(X, *fit_normalizer(X))
        self.assertEqual(out[0, :, 0].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(out[0, :, 1].tolist(), [-1.0, 1.0, 0.0, 0.0])

    def test_constant_feature_does_not_divide_by_zero(self):
        X = np.ones((1, 4, 11), dtype=np.float32)
        self.assertTrue(np.isfinite(normalize(X, *fit_normalizer(X))).all())


if __name__ == "__main__":
    unittest.main()

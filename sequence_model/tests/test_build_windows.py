import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model.build_windows import (EXCLUDE_END_MISSING, EXCLUDE_END_UNCERTAIN, EXCLUDE_RATIO, EYE_CLOSED, HEAD_DOWN,
                                          NORMAL, UNASSIGNED, UNCERTAIN, YAWN, build, clip_windows, driver_overlap,
                                          exclude_reason, frame_states, load_features, nearest_frames, to_frame,
                                          window_label)
from sequence_model.config import FEATURE_COLUMNS

FPS = 25.0


def label(frames: int = 250, fps: float = FPS, **channels) -> dict:
    return {"clip": "c", "source": "nitymed", "driver_id": "d", "label_status": "reviewed", "fps": fps, "frames": frames,
            "eye_closed": [], "yawn": [], "uncertain": [], "head_down": [], **channels}


def features_csv(path: Path, frames: int, fps: float = FPS, value: float = 1.0, missing: tuple = ()) -> None:
    df = pd.DataFrame(np.full((frames, len(FEATURE_COLUMNS)), value), columns=list(FEATURE_COLUMNS))
    df["face"] = 1.0
    df.loc[list(missing), "face"] = 0.0
    df.loc[list(missing), [c for c in FEATURE_COLUMNS if c != "face"]] = np.nan
    df.insert(0, "t", np.arange(frames) / fps)
    df.to_csv(path, index=False)


class FrameStatesTest(unittest.TestCase):
    def test_rounded_label_time_maps_to_its_frame(self):
        for frame in range(0, 6000, 37):
            self.assertEqual(to_frame(round(frame / 29.76, 3), 29.76), frame)

    def test_end_is_exclusive(self):
        states = frame_states(label(10, eye_closed=[[0.08, 0.16]]), 10)
        self.assertEqual(np.flatnonzero(states == EYE_CLOSED).tolist(), [2, 3])

    def test_priority(self):
        one = [[0.0, 0.04]]
        all_channels = label(1, eye_closed=one, head_down=one, uncertain=one, yawn=[{"start": 0.0, "end": 0.04, "hand": False}])
        self.assertEqual(frame_states(all_channels, 1)[0], UNCERTAIN)
        self.assertEqual(frame_states(label(1, eye_closed=one, head_down=one), 1)[0], EYE_CLOSED)
        self.assertEqual(frame_states(label(1, head_down=one, yawn=[{"start": 0.0, "end": 0.04, "hand": False}]), 1)[0], HEAD_DOWN)


class WindowLabelTest(unittest.TestCase):
    def test_end_frame_decides_eye_and_head(self):
        states = np.zeros(45, int)
        states[-1] = EYE_CLOSED
        self.assertEqual(window_label(states), EYE_CLOSED)
        states[-1] = HEAD_DOWN
        self.assertEqual(window_label(states), HEAD_DOWN)

    def test_closed_earlier_in_window_is_normal(self):
        states = np.full(45, EYE_CLOSED)
        states[-1] = NORMAL
        self.assertEqual(window_label(states), NORMAL)

    def test_yawn_needs_half_of_window(self):
        states = np.zeros(45, int)
        states[:23] = YAWN
        self.assertEqual(window_label(states), YAWN)
        states[22] = NORMAL
        self.assertEqual(window_label(states), NORMAL)

    def test_eye_closed_frames_do_not_count_as_yawn(self):
        states = np.zeros(45, int)
        states[:22] = YAWN
        states[22:30] = EYE_CLOSED      # 라벨 구간으로는 하품이었어도 상태는 눈 감김이다
        self.assertEqual(window_label(states), NORMAL)

    def test_eye_closed_end_beats_yawn(self):
        states = np.full(45, YAWN)
        states[-1] = EYE_CLOSED
        self.assertEqual(window_label(states), EYE_CLOSED)


class ExcludeTest(unittest.TestCase):
    def test_reasons(self):
        states, face = np.zeros(10, int), np.ones(10)
        self.assertIsNone(exclude_reason(states, face))
        states[-1] = UNCERTAIN
        self.assertEqual(exclude_reason(states, face), EXCLUDE_END_UNCERTAIN)
        states[-1] = NORMAL
        face[-1] = 0
        self.assertEqual(exclude_reason(states, face), EXCLUDE_END_MISSING)

    def test_ratio_boundary_counts_uncertain_and_missing(self):
        states, face = np.zeros(10, int), np.ones(10)
        states[0], face[1] = UNCERTAIN, 0
        self.assertIsNone(exclude_reason(states, face))
        states[2] = UNCERTAIN
        self.assertEqual(exclude_reason(states, face), EXCLUDE_RATIO)


class ClipWindowsTest(unittest.TestCase):
    def table(self, seconds: float, fps: float = FPS):
        n = round(seconds * fps)
        X = np.ones((n, len(FEATURE_COLUMNS)), dtype=np.float32)
        return np.arange(n) / fps, X, np.zeros(n, int), fps

    def test_eight_seconds_gives_eleven_windows(self):
        windows, labels, starts, excluded = clip_windows(*self.table(8))
        self.assertEqual((len(windows), windows[0].shape), (11, (45, 11)))
        self.assertEqual((starts[0], starts[-1]), (0.0, 5.0))
        self.assertFalse(excluded)

    def test_short_clip_has_no_window(self):
        self.assertEqual(len(clip_windows(*self.table(2.96))[0]), 0)
        self.assertEqual(len(clip_windows(*self.table(3.0))[0]), 1)

    def test_resample_uses_time_not_frame_count(self):
        for fps in (25.0, 29.76):
            t, X, states, fps = self.table(4, fps)
            X[:, 1] = t
            windows, _, starts, _ = clip_windows(t, X, states, fps)
            for window, start in zip(windows, starts):
                expected = start + np.arange(45) / 15
                self.assertLessEqual(np.abs(window[:, 1] - expected).max(), 0.5 / fps + 1e-5)

    def test_nearest_frame(self):
        self.assertEqual(nearest_frames(np.array([0.0, 1.0, 2.0]), np.array([0.4, 0.6, 5.0])).tolist(), [0, 1, 2])


class LoadFeaturesTest(unittest.TestCase):
    def test_missing_face_becomes_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c.csv"
            features_csv(path, 3, missing=(1,))
            t, X = load_features(path)
        self.assertEqual(X.shape, (3, 11))
        self.assertTrue(np.array_equal(X[1], np.zeros(11)))
        self.assertEqual(X[0, FEATURE_COLUMNS.index("face")], 1.0)
        self.assertAlmostEqual(t[2], 0.08)

    def test_extract_json_uses_t_s(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c_rgb_face.json"
            columns = ["frame", "t_s", *FEATURE_COLUMNS]
            frames = [[0, 0.0, 0, *[None] * 10], [1, 0.5, 1, *[0.2] * 10]]
            path.write_text(json.dumps({"clip": "c", "fps": 2.0, "columns": columns, "frames": frames}), encoding="utf-8")
            t, X = load_features(path)
        self.assertEqual(t.tolist(), [0.0, 0.5])
        self.assertEqual(X[0].sum(), 0.0)


class BuildTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name in ("labels", "features", "dmd_labels", "dmd_features"):
            (self.root / name).mkdir()
        self.split = self.root / "split.json"

    def clip(self, name: str, driver: str, value: float = 1.0, feature: bool = True, **channels) -> None:
        data = {**label(100, **channels), "clip": name, "driver_id": driver}
        (self.root / "labels" / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
        if feature:
            features_csv(self.root / "features" / f"{name}.csv", 100, value=value)

    def write_split(self, splits: dict, drivers: dict) -> None:
        self.split.write_text(json.dumps({"unit": "driver_id", "splits": splits, "drivers": drivers}), encoding="utf-8")

    def run_build(self, **kwargs):
        return build(self.root / "labels", self.split, self.root / "features", self.root / "out", **kwargs)

    def standard(self) -> None:
        self.clip("P1", "a", 1.0, head_down=[[0.0, 4.0]])
        self.clip("P2", "b", 3.0, eye_closed=[[0.0, 4.0]])
        self.clip("P3", "c", 100.0)
        self.clip("P4", "d", feature=False)
        self.write_split({"train": ["P1"], "val": ["P2"], "test": ["P3", "P4"]}, {"P1": "a", "P2": "b", "P3": "c", "P4": "d"})

    def test_outputs_and_keys(self):
        self.standard()
        info, _, skipped, overlap = self.run_build()
        with np.load(self.root / "out" / "windows.npz") as w:
            self.assertEqual(sorted(w.files), ["X", "clip", "driver_id", "source", "split", "start_sec", "y"])
            self.assertEqual(w["X"].shape, (9, 45, 11))
            self.assertEqual(w["y"][w["clip"] == "P1"].tolist(), [HEAD_DOWN] * 3)
            self.assertEqual(w["y"][w["clip"] == "P2"].tolist(), [EYE_CLOSED] * 3)
            self.assertEqual(w["y"][w["clip"] == "P3"].tolist(), [NORMAL] * 3)
            self.assertEqual(w["start_sec"][:3].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(skipped, [("P4", "특징 파일 없음")])
        self.assertEqual(overlap, {})
        self.assertEqual(len(info), 9)

    def test_norm_uses_train_only_without_face(self):
        self.standard()
        self.run_build()
        with np.load(self.root / "out" / "norm.npz") as n:
            self.assertEqual(n["mean"].shape, (10,))
            self.assertNotIn("face", n["columns"].tolist())
            self.assertTrue(np.allclose(n["mean"], 1.0))    # val 3.0, test 100.0이 섞이면 달라진다
            self.assertTrue(np.allclose(n["std"], 1.0))

    def test_split_file_is_not_changed(self):
        self.standard()
        before = self.split.read_bytes()
        self.run_build()
        self.assertEqual(self.split.read_bytes(), before)
        self.assertFalse((self.root / "out" / "split.json").exists())

    def test_driver_in_two_splits_is_an_error(self):
        self.clip("P1", "a")
        self.clip("P2", "a")
        self.write_split({"train": ["P1"], "test": ["P2"]}, {"P1": "a", "P2": "a"})
        with self.assertRaisesRegex(ValueError, "여러 분할"):
            self.run_build()

    def test_driver_mismatch_with_split_file_is_an_error(self):
        self.clip("P1", "a")
        self.write_split({"train": ["P1"]}, {"P1": "z"})
        with self.assertRaisesRegex(ValueError, "driver_id"):
            self.run_build()

    def test_excluded_windows_are_counted_and_dropped(self):
        self.clip("P1", "a", uncertain=[[2.9, 3.0]])
        self.write_split({"train": ["P1"]}, {"P1": "a"})
        info, excluded, _, _ = self.run_build()
        self.assertEqual(info["start_sec"].tolist(), [0.5, 1.0])
        self.assertEqual(excluded, {("train", EXCLUDE_END_UNCERTAIN, NORMAL): 1})

    def test_auto_nitymed_label_is_skipped(self):
        self.standard()
        self.clip("P5", "e", label_status="auto")
        _, _, skipped, _ = self.run_build()
        self.assertIn(("P5", "label_status=auto"), skipped)

    def test_dmd_is_excluded_by_default_and_unassigned_when_included(self):
        self.standard()
        dmd = {**label(100), "clip": "g", "source": "dmd", "driver_id": 14, "label_status": "auto"}
        (self.root / "dmd_labels" / "g.json").write_text(json.dumps(dmd), encoding="utf-8")
        features_csv(self.root / "dmd_features" / "g.csv", 100, value=50.0)
        self.assertNotIn("dmd", self.run_build()[0]["source"].tolist())
        info, *_ = self.run_build(dmd_label_dir=self.root / "dmd_labels", dmd_feature_dir=self.root / "dmd_features")
        self.assertEqual(set(info[info["source"] == "dmd"]["split"]), {UNASSIGNED})
        with np.load(self.root / "out" / "norm.npz") as n:
            self.assertTrue(np.allclose(n["mean"], 1.0))

    def test_driver_overlap_ignores_unassigned(self):
        rows = pd.DataFrame({"driver_id": ["a", "a"], "split": ["train", UNASSIGNED]})
        self.assertEqual(driver_overlap(rows), {})


if __name__ == "__main__":
    unittest.main()

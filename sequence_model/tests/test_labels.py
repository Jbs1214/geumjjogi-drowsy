import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sequence_model import config
from sequence_model.labels import (clip_group, clip_id_of, load_auto_label, load_reviewed_segments, segments_to_frames,
                                   split_clips)

# 로컬 영상의 SGA별 클립 수 (2,095개). 그룹 크기가 치우친 실제 분포로 분할을 확인한다
REAL_SIZES = (80, 1, 200, 56, 199, 47, 10, 207, 43, 192, 51, 199, 85, 1, 178, 50, 39, 327, 120, 10)


def clips_of(sizes: tuple[int, ...]) -> list[str]:
    return [f"SGA21{g:05d}S{c:04d}" for g, size in enumerate(sizes) for c in range(size)]


class ClipIdTest(unittest.TestCase):
    def test_clip_id_of_both_names(self):
        self.assertEqual(clip_id_of(Path("a/SGA2100002S0124_자동라벨.json")), "SGA2100002S0124")
        self.assertEqual(clip_id_of(Path("a/SGA2100002S0124.json")), "SGA2100002S0124")

    def test_group(self):
        self.assertEqual(clip_group("SGA2100002S0124"), "SGA2100002")

    def test_other_clip_is_its_own_group(self):
        self.assertEqual(clip_group("webcam01"), "webcam01")


class SplitTest(unittest.TestCase):
    def test_groups_never_cross_splits(self):
        splits = split_clips(clips_of(REAL_SIZES))
        by_group = {}
        for clip, name in splits.items():
            by_group.setdefault(clip_group(clip), set()).add(name)
        self.assertTrue(all(len(names) == 1 for names in by_group.values()))

    def test_ratio_is_close_on_skewed_real_sizes(self):
        splits = list(split_clips(clips_of(REAL_SIZES)).values())
        for name, ratio in config.SPLIT_RATIOS.items():
            self.assertAlmostEqual(splits.count(name) / len(splits), ratio, delta=0.02)

    def test_calibration_groups_are_never_in_test(self):
        clips = clips_of(REAL_SIZES)
        calibration = tuple(clips[::70])    # 그룹 여러 개에 걸친 30개
        calibration_groups = {clip_group(c) for c in calibration}
        for seed in range(20):
            splits = split_clips(clips, calibration, seed)
            self.assertFalse([c for c, name in splits.items() if name == "test" and clip_group(c) in calibration_groups])

    def test_same_seed_same_split(self):
        clips = clips_of(REAL_SIZES)
        self.assertEqual(split_clips(clips, seed=1), split_clips(clips, seed=1))

    def test_seed_changes_split(self):
        clips = clips_of(REAL_SIZES)
        self.assertNotEqual(split_clips(clips, seed=1), split_clips(clips, seed=2))

    def test_every_split_gets_a_group(self):
        for seed in range(20):
            self.assertEqual(set(split_clips(clips_of((10, 1, 1)), seed=seed).values()), {"train", "val", "test"})

    def test_every_clip_is_assigned_even_if_all_groups_are_calibration(self):
        clips = clips_of((3, 2))
        self.assertEqual(set(split_clips(clips, tuple(clips)).values()), {"train"})


class LabelFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "SGA2100002S0124_자동라벨.json"

    def write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_auto_label_with_null_frame(self):
        row = [1.0] * 11
        self.write({"X": [row, [None] * 11], "y": [0, 3], "segments": [], "params": {}})
        clip = load_auto_label(self.path)
        self.assertEqual(clip.clip_id, "SGA2100002S0124")
        self.assertTrue(np.isnan(clip.X[1]).all())
        df = clip.to_frame()
        self.assertEqual(df["t"].tolist(), [0.0, 1 / config.FPS])
        self.assertEqual(df["state"].tolist(), [0, 3])

    def test_wrong_feature_count_is_rejected(self):
        self.write({"X": [[1.0] * 5], "y": [0]})
        with self.assertRaises(ValueError):
            load_auto_label(self.path)

    def test_reviewed_segments_map_all_five_states(self):
        names = ["정상", "눈감김·고개숙임", "하품", "손·팔 동작", "불확실"]
        self.write({"segments": [{"start": i, "end": i + 1, "state": n} for i, n in enumerate(names)]})
        self.assertEqual([s[2] for s in load_reviewed_segments(self.path)], [0, 1, 2, 3, 3])

    def test_unknown_state_is_rejected(self):
        self.write({"segments": [{"start": 0, "end": 1, "state": "오타"}]})
        with self.assertRaises(KeyError):
            load_reviewed_segments(self.path)


class SegmentsToFramesTest(unittest.TestCase):
    def test_uncovered_frames_are_uncertain(self):
        states = segments_to_frames([(0.0, 1.0, 0), (1.0, 1.5, 1)], 60, 30)
        self.assertEqual(states[:30].tolist(), [0] * 30)
        self.assertEqual(states[30:45].tolist(), [1] * 15)
        self.assertEqual(states[45:].tolist(), [config.UNCERTAIN] * 15)


if __name__ == "__main__":
    unittest.main()

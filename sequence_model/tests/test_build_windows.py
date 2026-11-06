import unittest

import numpy as np
import pandas as pd

import config
from sequence_model.build_windows import (CLASS_NAMES, EYE_CLOSED, FEATURE_COLUMNS, HEAD_DOWN, STRIDE_SEC, WINDOW_FRAMES,
                                          channel_masks, clip_windows, normalize)

# 우선순위, 창 정답 규칙, 0.5초 간격, 25→15 리샘플, 미검출 0 처리, 미검출 창 제외, 정규화 값, 하품 전용 정답은
# tests/test_v3.py가 검사한다. 여기서는 그 밖의 build_windows 동작만 본다.


def features(n: int, fps: float = 25.0) -> pd.DataFrame:
    df = pd.DataFrame({"t": np.arange(n) / fps, "face_found": True})
    for name in config.FEATURES:
        if name != config.FACE_COLUMN:
            df[name] = 0.3
    return df


def label(frames: int = 250, fps: float = 25.0, **channels) -> dict:
    return {"fps": fps, "frames": frames, "eye_closed": [], "yawn": [], "uncertain": [], "head_down": [], **channels}


class NamesTest(unittest.TestCase):
    def test_names_used_by_train_lstm_follow_config(self):
        # 기술서 0·2.2: 3초 15 FPS = 45프레임, 학습 간격 0.5초, 입력 11개, 4클래스
        self.assertEqual(FEATURE_COLUMNS, config.FEATURES)
        self.assertEqual(len(FEATURE_COLUMNS), 11)
        self.assertEqual((WINDOW_FRAMES, STRIDE_SEC), (45, 0.5))
        self.assertEqual(CLASS_NAMES, ("정상", "눈 감김", "하품", "고개 숙임"))
        self.assertEqual((EYE_CLOSED, HEAD_DOWN), (config.CLOSED, config.HEAD))


class ChannelMaskTest(unittest.TestCase):
    def test_spans_include_start_and_exclude_end(self):
        masks = channel_masks(label(10, 10.0, eye_closed=[[0.2, 0.5]], yawn=[{"start": 0.0, "end": 0.2, "hand": True}]), 10, 10.0)
        self.assertEqual(np.flatnonzero(masks["eye_closed"]).tolist(), [2, 3, 4])
        self.assertEqual(np.flatnonzero(masks["yawn"]).tolist(), [0, 1])
        self.assertFalse(masks["uncertain"].any() or masks["head_down"].any())

    def test_missing_channel_is_empty(self):
        masks = channel_masks({"eye_closed": [[0.0, 1.0]]}, 10, 10.0)
        self.assertEqual(set(masks), {"eye_closed", "uncertain", "head_down", "yawn"})
        self.assertFalse(masks["head_down"].any())


class ClipWindowsTest(unittest.TestCase):
    def test_shape_and_end_times(self):
        X, y, ends, counts = clip_windows(features(250), label())
        self.assertEqual(X.shape, (15, WINDOW_FRAMES, len(config.FEATURES)))
        # 0.5초 배수가 반 프레임에 걸리면(3.5초 = 52.5프레임) 가장 가까운 프레임이라 반 프레임까지 어긋난다
        np.testing.assert_allclose(ends, np.arange(3.0, 10.0 + 1e-9, 0.5), atol=0.5 / config.FPS + 1e-9)
        self.assertEqual(counts, {"창": 15, "불확실": 0, "미검출": 0})

    def test_end_frame_uncertain_is_excluded_and_counted(self):
        # 끝 프레임이 불확실인 창은 정답이 없어 뺀다 (기술서 2.5). 6.0초에 끝나는 창 하나만 해당한다
        _, y, ends, counts = clip_windows(features(250), label(uncertain=[[5.9, 6.1]]))
        self.assertEqual(counts["불확실"], 1)
        self.assertFalse(np.isclose(ends, 6.0).any())
        self.assertEqual(counts["창"], len(y) + counts["불확실"] + counts["미검출"])

    def test_label_frames_limit_clip_length(self):
        # 특징 표가 라벨보다 길면 라벨 frames까지만 쓴다
        _, _, ends, _ = clip_windows(features(250), label(frames=150))
        self.assertAlmostEqual(float(ends.max()), 6.0)
        self.assertEqual(len(ends), 7)

    def test_head_down_end_frame_label(self):
        _, y, ends, _ = clip_windows(features(250), label(head_down=[[5.0, 7.0]]))
        self.assertEqual(int(y[np.isclose(ends, 6.0)][0]), HEAD_DOWN)
        self.assertEqual(int(y[np.isclose(ends, 8.0)][0]), config.NORMAL)


class NormalizeTest(unittest.TestCase):
    def test_detected_frames_are_scaled_and_face_stays_one(self):
        X = np.zeros((1, WINDOW_FRAMES, len(config.FEATURES)), np.float32)
        face, ear = config.FEATURES.index(config.FACE_COLUMN), config.FEATURES.index("ear")
        X[0, :, face], X[0, :, ear] = 1.0, 4.0
        mean, std = np.zeros(len(config.FEATURES)), np.ones(len(config.FEATURES))
        mean[ear], std[ear] = 2.0, 2.0
        out = normalize(X, {"mean": mean.tolist(), "std": std.tolist()})
        self.assertTrue(np.allclose(out[0, :, ear], 1.0))
        self.assertTrue(np.allclose(out[0, :, face], 1.0))


if __name__ == "__main__":
    unittest.main()

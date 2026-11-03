import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rule_based import config
from rule_based.alarm import CAUTION, DROWSY, NORMAL, EyeClosureAlarm, ProbToClosed, alert_times, stages
from rule_based.eyes import ear_threshold, is_closed


def clip(closed_sec: float, fps: int = 30, before: float = 1.0, after: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """눈 뜸 → closed_sec초 감음 → 눈 뜸 영상의 (t, closed). 감음 구간은 정확히 before초에 시작한다."""
    n_before, n_closed, n_after = round(before * fps), round(closed_sec * fps) + 1, round(after * fps)
    closed = np.r_[np.zeros(n_before, bool), np.ones(n_closed, bool), np.zeros(n_after, bool)]
    return np.arange(len(closed)) / fps, closed


class StageTest(unittest.TestCase):
    def test_thresholds_are_from_the_plan(self):
        self.assertEqual((config.CAUTION_SEC, config.DROWSY_SEC), (1.5, 3.0))
        self.assertFalse(hasattr(config, "ALERT_SEC"))

    def test_stage_boundaries_and_seconds(self):
        for sec, expected in ((0.0, NORMAL), (1.49, NORMAL), (1.5, CAUTION), (2.99, CAUTION), (3.0, DROWSY), (7.0, DROWSY)):
            alarm = EyeClosureAlarm()
            alarm.update(True, 10.0)
            level, closed_sec = alarm.update(True, 10.0 + sec)
            self.assertEqual(level, expected, sec)
            self.assertAlmostEqual(closed_sec, sec)

    def test_time_comes_from_t_not_frame_count(self):
        alarm = EyeClosureAlarm()
        alarm.update(True, 0.0)
        self.assertEqual(alarm.update(True, 3.0), (DROWSY, 3.0))

    def test_open_eye_resets_time(self):
        alarm = EyeClosureAlarm()
        alarm.update(True, 0.0)
        self.assertEqual(alarm.update(True, 2.0)[0], CAUTION)
        self.assertEqual(alarm.update(False, 2.1), (NORMAL, 0.0))
        self.assertEqual(alarm.update(True, 2.2), (NORMAL, 0.0))
        self.assertEqual(alarm.update(True, 3.6)[0], NORMAL)

    def test_eyes_not_visible_before_hook_uses_closed_only(self):
        alarm = EyeClosureAlarm()
        alarm.update(True, 0.0)
        self.assertEqual(alarm.update(True, 2.0, eyes_visible=False), (CAUTION, 2.0))
        self.assertEqual(alarm.update(False, 2.1, eyes_visible=False), (NORMAL, 0.0))
        self.assertEqual(alarm.update(True, 2.2), (NORMAL, 0.0))

    def test_blink_has_no_alert(self):
        self.assertEqual(alert_times(*clip(0.3)), [])

    def test_microsleep_cautions_only(self):
        alerts = alert_times(*clip(2.0))
        self.assertEqual([level for _, level in alerts], [CAUTION])
        self.assertAlmostEqual(alerts[0][0], 2.5, places=2)

    def test_long_closure_cautions_then_drowsy_once_each(self):
        alerts = alert_times(*clip(4.0))
        self.assertEqual([level for _, level in alerts], [CAUTION, DROWSY])
        self.assertAlmostEqual(alerts[0][0], 2.5, places=2)
        self.assertAlmostEqual(alerts[1][0], 4.0, places=2)

    def test_two_closures_count_separately(self):
        t1, c1 = clip(2.0)
        t2, c2 = clip(2.0)
        t = np.r_[t1, t2 + t1[-1] + 1 / 30]
        self.assertEqual([level for _, level in alert_times(t, np.r_[c1, c2])], [CAUTION, CAUTION])

    def test_stages_length_matches_input(self):
        t, closed = clip(1.0)
        self.assertEqual(len(stages(t, closed)), len(t))


class ProbToClosedTest(unittest.TestCase):
    def test_hysteresis(self):
        gate = ProbToClosed()
        results = [gate.update(p) for p in (0.5, 0.6, 0.5, 0.41, 0.4, 0.59)]
        self.assertEqual(results, [False, True, True, True, False, False])

    def test_model_output_drives_same_timer(self):
        gate, alarm = ProbToClosed(), EyeClosureAlarm()
        alarm.update(gate.update(0.9), 0.0)
        self.assertEqual(alarm.update(gate.update(0.5), 1.5), (CAUTION, 1.5))
        self.assertEqual(alarm.update(gate.update(0.3), 1.7), (NORMAL, 0.0))


class EarThresholdTest(unittest.TestCase):
    def test_missing_file_uses_ear_closed(self):
        threshold, source = ear_threshold(np.full(10, 0.3), "no/such/best_params.json")
        self.assertEqual(threshold, config.EAR_CLOSED)
        self.assertIn("EAR_CLOSED", source)

    def test_file_uses_ratio_times_p75_without_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_params.json"
            path.write_text(json.dumps({"params": {"ear_ratio": 0.5}}), encoding="utf-8")
            ear = np.r_[np.full(10, 0.4), [np.nan] * 5]
            threshold, source = ear_threshold(ear, path)
        self.assertAlmostEqual(threshold, 0.2)
        self.assertIn("ear_ratio", source)

    def test_too_few_face_frames_use_ear_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_params.json"
            path.write_text(json.dumps({"params": {"ear_ratio": 0.5}}), encoding="utf-8")
            self.assertEqual(ear_threshold(np.full(5, 0.4), path)[0], config.EAR_CLOSED)

    def test_is_closed_is_unchanged(self):
        self.assertEqual(is_closed(np.array([0.19, 0.2, np.nan])).tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()

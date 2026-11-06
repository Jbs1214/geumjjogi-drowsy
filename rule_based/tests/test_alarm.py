import unittest

import numpy as np

from rule_based import config
from rule_based.alarm import DROWSY, NORMAL, WARN, ClosureTimer, EyeClosureAlarm, alert_times, stage_starts, timer_stages
from rule_based.eyes import is_closed

# 끊김 허용(grace), 판정 불가 이어받기, 켜짐·꺼짐(eval/window_eval.py), ear_ratio × P75(eval/evaluate.py)는
# tests/test_v3.py와 eval/tests가 검사한다. 여기서는 타이머의 단계 경계와 시간 계산만 본다.

OPEN_EAR, CLOSED_EAR = 0.3, 0.05


def clip(closed_sec: float, fps: int = 30, before: float = 1.0, after: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """눈 뜸 → closed_sec초 감음 → 눈 뜸 영상의 (t, closed). 감음 구간은 정확히 before초에 시작한다."""
    n_before, n_closed, n_after = round(before * fps), round(closed_sec * fps) + 1, round(after * fps)
    closed = np.r_[np.zeros(n_before, bool), np.ones(n_closed, bool), np.zeros(n_after, bool)]
    return np.arange(len(closed)) / fps, closed


def ear_of(closed: np.ndarray) -> np.ndarray:
    return np.where(closed, CLOSED_EAR, OPEN_EAR)


class StageTest(unittest.TestCase):
    def test_thresholds_are_from_the_spec(self):
        # 기술서 0·3.1: 1.5초 주의, 3초 졸음, 켜짐 0.6·꺼짐 0.4. 3.3: 직전 감김 0.5초 이상이면 이어받기
        self.assertEqual((config.WARN_SEC, config.ALERT_SEC), (1.5, 3.0))
        self.assertEqual((config.PROB_ON, config.PROB_OFF), (0.6, 0.4))
        self.assertEqual(config.CARRY_MIN_SEC, 0.5)

    def test_stage_boundaries_and_seconds(self):
        for sec, expected in ((0.0, NORMAL), (1.49, NORMAL), (1.5, WARN), (2.99, WARN), (3.0, DROWSY), (7.0, DROWSY)):
            timer = ClosureTimer()
            timer.update(10.0, True)
            self.assertEqual(timer.update(10.0 + sec, True), expected, sec)
            self.assertAlmostEqual(timer.elapsed, sec)

    def test_time_comes_from_t_not_frame_count(self):
        timer = ClosureTimer()
        timer.update(0.0, True)
        self.assertEqual(timer.update(3.0, True), DROWSY)
        self.assertEqual(timer.elapsed, 3.0)

    def test_open_eye_resets_time(self):
        timer = ClosureTimer()
        timer.update(0.0, True)
        self.assertEqual(timer.update(2.0, True), WARN)
        self.assertEqual(timer.update(2.1, False), NORMAL)
        self.assertEqual(timer.elapsed, 0.0)
        self.assertEqual(timer.update(2.2, True), NORMAL)
        self.assertEqual(timer.update(3.6, True), NORMAL)     # 다시 잰 1.4초

    def test_stages_length_matches_input(self):
        t, closed = clip(1.0)
        self.assertEqual(len(timer_stages(t, closed)), len(t))


class AlertTest(unittest.TestCase):
    def test_blink_has_no_alert(self):
        t, closed = clip(0.3)
        self.assertEqual(stage_starts(t, timer_stages(t, closed), WARN).tolist(), [])
        self.assertEqual(alert_times(t, ear_of(closed)), [])

    def test_microsleep_warns_only(self):
        t, closed = clip(2.0)
        stages = timer_stages(t, closed)
        self.assertEqual(stage_starts(t, stages, WARN).tolist(), [2.5])
        self.assertEqual(stage_starts(t, stages, DROWSY).tolist(), [])
        self.assertEqual(alert_times(t, ear_of(closed)), [])

    def test_long_closure_warns_then_drowsy_once_each(self):
        t, closed = clip(4.0)
        stages = timer_stages(t, closed)
        self.assertEqual(len(stage_starts(t, stages, WARN)), 1)
        self.assertAlmostEqual(stage_starts(t, stages, WARN)[0], 2.5, places=2)
        alerts = alert_times(t, ear_of(closed))
        self.assertEqual(len(alerts), 1)
        self.assertAlmostEqual(alerts[0], 4.0, places=2)

    def test_two_closures_count_separately(self):
        t1, c1 = clip(2.0)
        t2, c2 = clip(2.0)
        t, closed = np.r_[t1, t2 + t1[-1] + 1 / 30], np.r_[c1, c2]
        self.assertEqual(len(stage_starts(t, timer_stages(t, closed), WARN)), 2)

    def test_missing_face_restarts_rule_timer(self):
        # 규칙 경로: 얼굴 미검출(NaN)은 감음이 아니라서 2초 + 2초로 끊겨 졸음 경고가 없다
        t, closed = clip(4.0)
        ear = ear_of(closed)
        ear[len(t) // 2] = np.nan
        self.assertEqual(alert_times(t, ear), [])

    def test_eye_closure_alarm_matches_alert_times(self):
        t, closed = clip(4.0)
        alarm = EyeClosureAlarm()
        on = [alarm.update(float(ti), float(ei)) for ti, ei in zip(t, ear_of(closed))]
        self.assertAlmostEqual(float(t[on.index(True)]), alert_times(t, ear_of(closed))[0])


class IsClosedTest(unittest.TestCase):
    def test_is_closed_is_unchanged(self):
        self.assertEqual(is_closed(np.array([0.19, 0.2, np.nan])).tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()

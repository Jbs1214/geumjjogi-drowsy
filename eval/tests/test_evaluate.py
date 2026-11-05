import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eval import window_eval as ev
from eval.metrics import alert_delay
from eval.window_eval import CAUTION, DROWSY
from sequence_model.build_windows import CLASS_NAMES


def probs_from(closed) -> np.ndarray:
    """창별 감김 여부 → 눈 감김 확률이 0.9 또는 0.1인 (N,4) 확률."""
    closed = np.asarray(closed, dtype=bool)
    p = np.where(closed, 0.9, 0.1)
    return np.column_stack([1 - p, p, np.zeros(len(p)), np.zeros(len(p))])


def run(closed_true, closed_pred, starts=None, clip=None) -> dict:
    """정답·예측 감김 목록으로 평가한다. starts를 안 주면 0.5초 간격."""
    n = len(closed_true)
    starts = np.arange(n) * 0.5 if starts is None else np.asarray(starts, dtype=float)
    clip = np.full(n, "a") if clip is None else np.asarray(clip)
    return ev.evaluate(clip, starts, np.asarray(closed_true, dtype=int), probs_from(closed_pred))


def pattern(*parts: tuple[int, int]) -> list[int]:
    """(값, 창 수) 이어 붙이기. pattern((0, 2), (1, 3)) → [0, 0, 1, 1, 1]."""
    return [value for value, count in parts for _ in range(count)]


class HysteresisTest(unittest.TestCase):
    def test_on_off_and_hold(self):
        closed = ev.closed_from_prob(np.array([0.7, 0.5, 0.4, 0.39, 0.5, 0.6]))
        self.assertEqual(closed.tolist(), [True, True, True, False, False, True])

    def test_starts_open(self):
        self.assertEqual(ev.closed_from_prob(np.array([0.5, 0.59])).tolist(), [False, False])


class MinCloseTest(unittest.TestCase):
    def test_one_window_is_kept(self):
        closed = np.array([False, True, False])
        self.assertEqual(ev.drop_short_closures(closed).tolist(), closed.tolist())


class TimerTest(unittest.TestCase):
    def test_six_windows_reach_three_seconds(self):
        alerts = ev.timer_alerts(np.arange(8) * 0.5, np.array(pattern((1, 6), (0, 2)), dtype=bool))
        self.assertEqual(alerts, [(1.5, CAUTION), (3.0, DROWSY)])

    def test_three_windows_are_caution_only(self):
        alerts = ev.timer_alerts(np.arange(5) * 0.5, np.array(pattern((1, 3), (0, 2)), dtype=bool))
        self.assertEqual(alerts, [(1.5, CAUTION)])

    def test_two_windows_have_no_alert(self):
        self.assertEqual(ev.timer_alerts(np.arange(4) * 0.5, np.array(pattern((1, 2), (0, 2)), dtype=bool)), [])


class PerfectModelTest(unittest.TestCase):
    def check(self, windows: int):
        y = pattern((0, 4), (1, windows), (0, 4))
        result = run(y, y)
        self.assertEqual(result["drowsy"]["n_events"], 1)
        self.assertEqual(result["drowsy"]["recall"], 1.0)
        self.assertEqual(result["drowsy"]["delay_mean"], 0.0)
        self.assertEqual(result["caution"]["recall"], 1.0)
        self.assertEqual(result["caution"]["delay_mean"], 0.0)
        self.assertEqual(result["span"]["f1"], 1.0)
        self.assertEqual(result["window"]["f1"], 1.0)
        self.assertEqual(result["false_alarms_total"]["count"], 0)

    def test_six_windows(self):
        self.check(6)

    def test_seven_windows(self):
        self.check(7)

    def test_short_event_is_outside_drowsy_denominator(self):
        y = pattern((0, 2), (1, 3), (0, 2))
        result = run(y, y)
        self.assertEqual(result["caution"]["recall"], 1.0)
        self.assertEqual(result["drowsy"]["n_events"], 0)
        self.assertTrue(math.isnan(result["drowsy"]["recall"]))

    def test_two_window_event_counts_for_neither(self):
        y = pattern((0, 2), (1, 2), (0, 2))
        result = run(y, y)
        self.assertEqual((result["caution"]["n_events"], result["drowsy"]["n_events"]), (0, 0))
        self.assertEqual(result["n_true_events"], 1)


class DelayTest(unittest.TestCase):
    def test_one_window_late(self):
        # 정답 2~6초(8창), 예측 2.5~6초. 졸음 경고 5.5초, 기준 도달 5.0초 → 지연 0.5초
        y = pattern((0, 4), (1, 8), (0, 4))
        result = run(y, pattern((0, 5), (1, 7), (0, 4)))
        self.assertEqual(result["drowsy"]["recall"], 1.0)
        self.assertEqual(result["drowsy"]["delay_mean"], 0.5)
        self.assertEqual(result["caution"]["delay_mean"], 0.5)

    def test_median_max_and_mean_agree_with_alert_delay(self):
        # 시작 0초 이벤트 3개. 지연 0.0, 0.5, 1.5초 → 평균 2/3, 중앙값 0.5, 최대 1.5
        events = [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]
        alerts = np.array([3.0, 13.5, 24.5])
        result = ev.level_metrics(events, alerts, 3.0)
        self.assertAlmostEqual(result["delay_mean"], 2 / 3)
        self.assertEqual(result["delay_median"], 0.5)
        self.assertEqual(result["delay_max"], 1.5)
        self.assertAlmostEqual(result["delay_mean"], alert_delay(events, alerts, 3.0, ev.ALERT_TOLERANCE_SEC))

    def test_missed_event_has_no_delay(self):
        y = pattern((0, 2), (1, 7), (0, 2))
        result = run(y, [0] * len(y))
        self.assertEqual(result["drowsy"]["recall"], 0.0)
        self.assertTrue(math.isnan(result["drowsy"]["delay_mean"]))
        self.assertTrue(math.isnan(result["drowsy"]["delay_median"]))


class FalseAlarmTest(unittest.TestCase):
    def test_by_hand(self):
        # 창 20개(10초), 정답 감김 4창(2초), 분모 8초. 정답 밖 3창 감김 → 주의 1회 → 1/(8/3600) = 450회/시간
        y = pattern((0, 2), (1, 4), (0, 14))
        pred = pattern((0, 10), (1, 3), (0, 7))
        result = run(y, pred)
        self.assertEqual(result["open_sec"], 8.0)
        self.assertEqual(result["caution"]["false_alarms"], 1)
        self.assertAlmostEqual(result["caution"]["false_alarms_per_hour"], 450.0)
        self.assertEqual(result["drowsy"]["false_alarms"], 0)
        self.assertAlmostEqual(result["false_alarms_total"]["per_hour"], 450.0)

    def test_alert_in_too_short_event_is_false(self):
        # 정답 1.0초(2창)인데 예측이 3창이라 시작 후 1.5초(끝+0.5초 안)에 주의 → 짧은 감김의 경고라 오경보
        y = pattern((0, 2), (1, 2), (0, 6))
        result = run(y, pattern((0, 2), (1, 3), (0, 5)))
        self.assertEqual(result["caution"]["n_alerts"], 1)
        self.assertEqual(result["caution"]["false_alarms"], 1)

    def test_caution_inside_long_event_is_not_false(self):
        y = pattern((0, 2), (1, 7), (0, 2))
        result = run(y, y)
        self.assertEqual(result["caution"]["n_alerts"], 1)
        self.assertEqual(result["caution"]["false_alarms"], 0)

    def test_long_false_closure_counts_both_levels(self):
        y = [0] * 10
        result = run(y, pattern((0, 2), (1, 6), (0, 2)))
        self.assertEqual(result["false_alarms_total"]["count"], 2)
        self.assertAlmostEqual(result["false_alarms_total"]["per_hour"], 2 / (5 / 3600))

    def test_no_open_time_is_nan(self):
        result = run([1, 1], [1, 1])
        self.assertTrue(math.isnan(result["false_alarms_total"]["per_hour"]))


class ToleranceTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((ev.LABEL_TOLERANCE_SEC, ev.ALERT_TOLERANCE_SEC, ev.MIN_CLOSE_SEC), (0.5, 0.5, 0.1))

    def test_alert_within_tolerance(self):
        # eval.metrics의 기본값 0.2초가 쓰였다면 끝+0.4초 경고를 놓친다
        result = ev.level_metrics([(0.0, 3.0)], np.array([3.4]), 3.0)
        self.assertEqual((result["recall"], result["false_alarms"]), (1.0, 0))
        self.assertAlmostEqual(result["delay_mean"], 0.4)

    def test_alert_outside_tolerance(self):
        result = ev.level_metrics([(0.0, 3.0)], np.array([3.6]), 3.0)
        self.assertEqual((result["recall"], result["false_alarms"]), (0.0, 1))

    def test_span_tolerance(self):
        # 정답 1~2초, 예측 2.5~3.5초: 예측 시작 − 0.5초가 정답 끝과 같아 닿지 않는다
        y = pattern((0, 2), (1, 2), (0, 6))
        result = run(y, pattern((0, 5), (1, 2), (0, 3)))
        self.assertEqual((result["span"]["tp"], result["span"]["fp"], result["span"]["fn"]), (0, 1, 1))
        # 정답 1~2초, 예측 2~3초: 닿았으므로 찾은 것이다
        result = run(y, pattern((0, 4), (1, 2), (0, 4)))
        self.assertEqual((result["span"]["tp"], result["span"]["fp"], result["span"]["fn"]), (1, 0, 0))


class WindowF1Test(unittest.TestCase):
    def test_by_hand(self):
        # 맞힘 2, 잘못 냄 1, 놓침 2 → 정밀도 2/3, 재현율 1/2, F1 4/7. 구간은 1개로 맞힘
        y = pattern((0, 2), (1, 4), (0, 4))
        result = run(y, pattern((0, 1), (1, 3), (0, 6)))
        self.assertEqual((result["window"]["tp"], result["window"]["fp"], result["window"]["fn"]), (2, 1, 2))
        self.assertAlmostEqual(result["window"]["f1"], 4 / 7)
        self.assertEqual(result["span"]["f1"], 1.0)


class GapTest(unittest.TestCase):
    def test_excluded_window_breaks_timer_and_event(self):
        # 2초 지점 창이 제외: 감김 4창 + 3창이 이어져 있어도 3초에 닿지 않고 이벤트도 2개다
        starts = np.r_[np.arange(4) * 0.5, 2.5 + np.arange(3) * 0.5]
        y = [1] * 7
        result = run(y, y, starts=starts)
        self.assertEqual(result["n_true_events"], 2)
        self.assertEqual(result["drowsy"]["n_alerts"], 0)
        self.assertEqual(result["caution"]["n_alerts"], 2)
        self.assertEqual(result["caution"]["n_events"], 2)

    def test_hysteresis_restarts_after_gap(self):
        # 0.5 확률은 유지 구간인데 끊긴 뒤에는 이전 상태가 없어 뜬 것으로 본다
        probs = np.array([[0.1, 0.9, 0, 0], [0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0]])
        result = ev.evaluate(np.full(3, "a"), np.array([0.0, 0.5, 5.0]), np.array([1, 1, 1]), probs)
        self.assertEqual(result["window"]["tp"], 2)
        self.assertEqual(result["window"]["fn"], 1)


class MultiClipTest(unittest.TestCase):
    def test_alert_in_other_clip_does_not_match(self):
        # 같은 시각대라도 a의 정답을 b의 경고로 찾으면 안 된다
        y = pattern((1, 7), (0, 2)) + [0] * 9
        pred = [0] * 9 + pattern((1, 7), (0, 2))
        clip = ["a"] * 9 + ["b"] * 9
        starts = np.r_[np.arange(9) * 0.5, np.arange(9) * 0.5]
        result = run(y, pred, starts=starts, clip=clip)
        self.assertEqual(result["n_clips"], 2)
        self.assertEqual(result["drowsy"]["recall"], 0.0)
        self.assertEqual(result["drowsy"]["false_alarms"], 1)
        self.assertEqual(result["span"]["tp"], 0)

    def test_row_order_does_not_matter(self):
        y = np.array(pattern((0, 2), (1, 7), (0, 2)) * 2)
        clip = np.array(["a"] * 11 + ["b"] * 11)
        starts = np.r_[np.arange(11) * 0.5, np.arange(11) * 0.5]
        order = np.random.default_rng(0).permutation(22)
        expected = ev.evaluate(clip, starts, y, probs_from(y))
        shuffled = ev.evaluate(clip[order], starts[order], y[order], probs_from(y)[order])
        self.assertEqual(expected, shuffled)
        self.assertEqual(expected["drowsy"]["n_events"], 2)
        self.assertEqual(expected["drowsy"]["recall"], 1.0)


class EventRowTest(unittest.TestCase):
    def rows(self, y, pred, starts=None, clip=None):
        n = len(y)
        starts = np.arange(n) * 0.5 if starts is None else np.asarray(starts, dtype=float)
        clip = np.full(n, "a") if clip is None else np.asarray(clip)
        return ev.evaluate_detail(clip, starts, np.asarray(y), probs_from(pred))[1]

    def test_hit_rows_by_hand(self):
        # 정답 1~4.5초(7창). 주의 2.5초, 졸음 4.0초에 울리고 지연은 0
        y = pattern((0, 2), (1, 7), (0, 2))
        rows = self.rows(y, y)
        self.assertEqual([(r["level"], r["alert_sec"], r["delay_sec"], r["result"]) for r in rows],
                         [(CAUTION, 2.5, 0.0, "맞힘"), (DROWSY, 4.0, 0.0, "맞힘")])
        self.assertEqual((rows[0]["start_sec"], rows[0]["end_sec"], rows[0]["length_sec"]), (1.0, 4.5, 3.5))

    def test_miss_and_short_rows(self):
        y = pattern((0, 2), (1, 7), (0, 2), (1, 1), (0, 2))
        rows = self.rows(y, [0] * len(y))
        self.assertEqual([(r.get("level"), r["result"]) for r in rows],
                         [(CAUTION, "놓침"), (DROWSY, "놓침"), (None, "기준 미만")])

    def test_false_alarm_is_in_clip_time_and_says_where(self):
        # b 클립에는 정답이 없다. 1초부터 3창 감김 → b 클립 시각 2.5초에 주의
        clip = ["a"] * 4 + ["b"] * 8
        starts = np.r_[np.arange(4) * 0.5, np.arange(8) * 0.5]
        rows = self.rows([0] * 12, [0] * 4 + pattern((0, 2), (1, 3), (0, 3)), starts, clip)
        self.assertEqual(rows, [{"clip": "b", "kind": "오경보", "level": CAUTION, "alert_sec": 2.5, "result": "정답 감김 밖"}])

    def test_false_alarm_inside_short_event(self):
        y = pattern((0, 2), (1, 2), (0, 6))
        rows = self.rows(y, pattern((0, 2), (1, 3), (0, 5)))
        self.assertEqual([r["result"] for r in rows], ["기준 미만", "기준보다 짧은 감김"])

    def test_rows_agree_with_metrics(self):
        y = pattern((0, 4), (1, 8), (0, 4), (1, 3), (0, 4))
        pred = pattern((0, 5), (1, 7), (0, 4), (0, 3), (0, 4))
        result, rows = ev.evaluate_detail(np.full(len(y), "a"), np.arange(len(y)) * 0.5, np.array(y), probs_from(pred))
        for level, name in ((CAUTION, "caution"), (DROWSY, "drowsy")):
            events = [r for r in rows if r["kind"] == "정답 이벤트" and r.get("level") == level]
            self.assertEqual(len(events), result[name]["n_events"])
            self.assertEqual(np.mean([r["result"] == "맞힘" for r in events]), result[name]["recall"])
            self.assertEqual(np.mean([r["delay_sec"] for r in events if "delay_sec" in r]), result[name]["delay_mean"])


class FileTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        # val 창 11개(a) + test 창 2개. test의 y는 틀린 값이라 읽으면 검사에 걸린다
        self.y = np.array(pattern((0, 2), (1, 7), (0, 2)))
        self.clip = np.full(11, "a")
        self.start = np.arange(11) * 0.5
        self.windows = self.dir / "windows.npz"
        np.savez(self.windows, clip=np.r_[self.clip, ["t", "t"]], start_sec=np.r_[self.start, [0.0, 0.5]],
                 y=np.r_[self.y, [-9, -9]], split=np.array(["val"] * 11 + ["test"] * 2))
        self.probs = probs_from(self.y)

    def call(self, *extra: str) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            return ev.main(["--windows", str(self.windows), "--out-dir", str(self.dir / "runs"), *extra])

    def save_npy(self, probs: np.ndarray) -> str:
        path = self.dir / "p.npy"
        np.save(path, probs)
        return str(path)

    def save_npz(self, **changes) -> str:
        path = self.dir / "val_probs.npz"
        values = {"clip": self.clip, "start_sec": self.start, "y": self.y, "classes": np.array(CLASS_NAMES),
                  "probs_v01": self.probs, "probs_v02": 1 - self.probs}
        np.savez(path, **{**values, **changes})
        return str(path)

    def test_npy_writes_results_with_settings(self):
        result = self.call("--probs", self.save_npy(self.probs))
        self.assertEqual(result["n_windows"], 11)
        self.assertEqual(result["drowsy"]["recall"], 1.0)
        saved = json.loads((self.dir / "runs" / "eval_p_val.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["settings"]["ALERT_TOLERANCE_SEC"], 0.5)
        self.assertEqual(saved["settings"]["LABEL_TOLERANCE_SEC"], 0.5)
        self.assertEqual(saved["settings"]["MIN_CLOSE_SEC"], 0.1)
        report = (self.dir / "runs" / "eval_p_val.md").read_text(encoding="utf-8")
        self.assertIn("ALERT_TOLERANCE_SEC=0.5", report)
        self.assertIn("최대 0.5초 어긋나고", report)

    def test_nan_is_saved_as_null(self):
        np.savez(self.windows, clip=np.full(2, "a"), start_sec=np.array([0.0, 0.5]), y=np.array([0, 0]), split=np.array(["val"] * 2))
        self.call("--probs", self.save_npy(probs_from([0, 0])))
        saved = json.loads((self.dir / "runs" / "eval_p_val.json").read_text(encoding="utf-8"))
        self.assertIsNone(saved["drowsy"]["recall"])

    def test_npz_key_selects_probs(self):
        result = self.call("--probs", self.save_npz(), "--probs-key", "probs_v01")
        self.assertEqual(result["drowsy"]["recall"], 1.0)
        self.assertTrue((self.dir / "runs" / "eval_val_probs_probs_v01_val.json").exists())

    def test_npz_shuffled_rows_are_aligned(self):
        order = np.random.default_rng(1).permutation(11)
        path = self.save_npz(clip=self.clip[order], start_sec=self.start[order], y=self.y[order], probs_v01=self.probs[order])
        self.assertEqual(self.call("--probs", path, "--probs-key", "probs_v01")["drowsy"]["recall"], 1.0)

    def assert_stops(self, message: str, *extra: str):
        with self.assertRaises(SystemExit) as caught:
            self.call(*extra)
        self.assertIn(message, str(caught.exception))

    def test_row_count_mismatch(self):
        self.assert_stops("행 수가 다릅니다", "--probs", self.save_npy(self.probs[:-1]))

    def test_wrong_shape(self):
        self.assert_stops("모양", "--probs", self.save_npy(self.probs[:, :3]))

    def test_logits_are_rejected(self):
        self.assert_stops("0~1", "--probs", self.save_npy(self.probs * 5))

    def test_rows_must_sum_to_one(self):
        self.assert_stops("행 합", "--probs", self.save_npy(self.probs * 0.5))

    def test_clip_mismatch(self):
        path = self.save_npz(clip=np.array(["a"] * 10 + ["z"]))
        self.assert_stops("clip", "--probs", path, "--probs-key", "probs_v01")

    def test_start_sec_mismatch(self):
        path = self.save_npz(start_sec=self.start + 0.5)
        self.assert_stops("start_sec", "--probs", path, "--probs-key", "probs_v01")

    def test_y_mismatch(self):
        y = self.y.copy()
        y[0] = 2
        self.assert_stops("y", "--probs", self.save_npz(y=y), "--probs-key", "probs_v01")

    def test_duplicate_window(self):
        start = self.start.copy()
        start[1] = 0.0
        self.assert_stops("두 번", "--probs", self.save_npz(start_sec=start), "--probs-key", "probs_v01")

    def test_class_order(self):
        path = self.save_npz(classes=np.array(CLASS_NAMES[::-1]))
        self.assert_stops("클래스 순서", "--probs", path, "--probs-key", "probs_v01")

    def test_class_names_without_spaces_pass(self):
        path = self.save_npz(classes=np.array(["정상", "눈감김", "하품", "고개숙임"]))
        self.assertEqual(self.call("--probs", path, "--probs-key", "probs_v01")["drowsy"]["recall"], 1.0)

    def test_class_names_without_spaces_still_check_order(self):
        path = self.save_npz(classes=np.array(["정상", "하품", "눈감김", "고개숙임"]))
        self.assert_stops("클래스 순서", "--probs", path, "--probs-key", "probs_v01")

    def test_missing_key(self):
        self.assert_stops("probs_v01", "--probs", self.save_npz(), "--probs-key", "probs_v09")

    def test_key_is_required_for_npz(self):
        self.assert_stops("--probs-key", "--probs", self.save_npz())

    def all_run(self, **changes) -> dict:
        path = self.save_npz(probs_v02=probs_from([0] * 11), **changes)
        with contextlib.redirect_stdout(io.StringIO()):
            return ev.main(["--windows", str(self.windows), "--probs", path, "--probs-key", "all", "--out-dir", str(self.dir / "check")])

    def test_all_writes_three_files(self):
        results = self.all_run()
        self.assertEqual(list(results), ["probs_v01", "probs_v02"])
        saved = json.loads((self.dir / "check" / "metrics.json").read_text(encoding="utf-8"))
        # v01은 완벽, v02는 놓침: 평균 0.5, 모표준편차 0.5
        self.assertEqual(saved["results"]["probs_v01"]["drowsy"]["recall"], 1.0)
        self.assertEqual(saved["results"]["probs_v02"]["drowsy"]["recall"], 0.0)
        self.assertEqual(saved["summary"]["mean"]["3초 재현율"], 0.5)
        self.assertEqual(saved["summary"]["std"]["3초 재현율"], 0.5)
        self.assertEqual(saved["probs"], "val_probs.npz")
        self.assertEqual(len(saved["sha256"]["probs"]), 64)
        self.assertEqual(saved["results"]["probs_v01"]["argmax"]["macro_f1"], 1.0)

        events = pd.read_csv(self.dir / "check" / "events.csv", encoding="utf-8-sig")
        self.assertEqual(events.columns.tolist(), ev.EVENT_COLUMNS)
        self.assertEqual(events["probs_key"].value_counts().to_dict(), {"probs_v01": 2, "probs_v02": 2})
        self.assertEqual(events[events["probs_key"] == "probs_v02"]["result"].tolist(), ["놓침", "놓침"])

        report = (self.dir / "check" / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 평균 |", report)
        self.assertIn("## probs_v02", report)
        self.assertIn("ALERT_TOLERANCE_SEC=0.5", report)

    def test_all_does_not_overwrite(self):
        self.all_run()
        before = (self.dir / "check" / "report.md").read_bytes()
        with self.assertRaises(SystemExit) as caught:
            self.all_run()
        self.assertIn("이미 있습니다", str(caught.exception))
        self.assertEqual((self.dir / "check" / "report.md").read_bytes(), before)

    def test_all_writes_nothing_if_a_key_is_bad(self):
        path = self.save_npz()      # probs_v02는 행 합이 1이 아니다
        with self.assertRaises(SystemExit):
            ev.main(["--windows", str(self.windows), "--probs", path, "--probs-key", "all", "--out-dir", str(self.dir / "check")])
        self.assertFalse((self.dir / "check").exists())

    def test_all_needs_npz(self):
        self.assert_stops(".npz에만", "--probs", self.save_npy(self.probs), "--probs-key", "all")

    def test_test_split_needs_final(self):
        path = self.save_npy(probs_from([0, 0]))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.call("--probs", path, "--split", "test")
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse((self.dir / "runs").exists())

    def test_test_split_with_final(self):
        np.savez(self.windows, clip=np.array(["t", "t"]), start_sec=np.array([0.0, 0.5]), y=np.array([0, 0]), split=np.array(["test"] * 2))
        result = self.call("--probs", self.save_npy(probs_from([0, 0])), "--split", "test", "--final")
        self.assertEqual(result["n_windows"], 2)
        self.assertTrue((self.dir / "runs" / "eval_p_test.json").exists())


if __name__ == "__main__":
    unittest.main()

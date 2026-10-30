import math
import unittest

import numpy as np

from eval.compare import simulate_fps, to_markdown
from eval.metrics import (alert_delay, classification_metrics, event_recall, false_alarms_per_hour, fps, match_segments,
                          prf, score_auc, segments_from_labels)

import pandas as pd


class ClassificationTest(unittest.TestCase):
    def test_values_by_hand(self):
        # 클래스 0: 맞힘 2, 잘못 낸 수 1, 놓침 0 → F1 0.8 / 클래스 1: 맞힘 1, 잘못 낸 수 0, 놓침 1 → F1 2/3
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]))
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["f1_per_class"][0], 0.8)
        self.assertAlmostEqual(result["f1_per_class"][1], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], (0.8 + 2 / 3) / 2)

    def test_class_only_in_prediction_counts(self):
        result = classification_metrics(np.array([0, 0]), np.array([0, 2]))
        self.assertEqual(result["f1_per_class"][2], 0.0)

    def test_empty(self):
        self.assertEqual(classification_metrics(np.array([]), np.array([]))["macro_f1"], 0.0)


class PrfTest(unittest.TestCase):
    def test_values(self):
        result = prf(3, 1, 2)
        self.assertAlmostEqual(result["precision"], 0.75)
        self.assertAlmostEqual(result["recall"], 0.6)
        self.assertAlmostEqual(result["f1"], 2 * 0.75 * 0.6 / 1.35)

    def test_nothing_found(self):
        self.assertEqual(prf(0, 0, 0)["f1"], 0.0)


class SegmentTest(unittest.TestCase):
    def test_segments_from_labels(self):
        t = np.arange(6) * 0.5
        result = segments_from_labels(t, np.array([0, 1, 1, 0, 1, 2]), 0.5, (1, 2))
        self.assertEqual(result[1], [(0.5, 1.5), (2.0, 2.5)])
        self.assertEqual(result[2], [(2.5, 3.0)])

    def test_match_within_tolerance(self):
        self.assertEqual(match_segments([(1.0, 2.0)], [(2.4, 3.0)]), (1, 0, 0))

    def test_outside_tolerance(self):
        self.assertEqual(match_segments([(1.0, 2.0)], [(2.6, 3.0)]), (0, 1, 1))

    def test_one_prediction_cannot_match_two_events(self):
        self.assertEqual(match_segments([(1.0, 2.0), (2.5, 3.0)], [(1.0, 3.0)]), (1, 0, 1))

    def test_extra_predictions_are_false_positives(self):
        self.assertEqual(match_segments([(1.0, 2.0)], [(1.0, 2.0), (5.0, 6.0)]), (1, 1, 0))

    def test_empty(self):
        self.assertEqual(match_segments([], []), (0, 0, 0))


class AucTest(unittest.TestCase):
    def test_perfect_and_reversed(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(score_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(score_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_ties_count_half(self):
        self.assertEqual(score_auc(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)

    def test_by_hand(self):
        # 양성 0.8은 음성 둘 다보다 크고, 양성 0.4는 음성 하나보다만 크다 → 3/4
        self.assertAlmostEqual(score_auc(np.array([1, 0, 1, 0]), np.array([0.8, 0.5, 0.4, 0.1])), 0.75)

    def test_one_class_is_nan(self):
        self.assertTrue(math.isnan(score_auc(np.array([1, 1]), np.array([0.1, 0.2]))))


class AlertTest(unittest.TestCase):
    def test_false_alarms(self):
        self.assertEqual(false_alarms_per_hour(np.array([1.0, 2.0, 3.0]), 0.5), 6.0)

    def test_false_alarm_needs_positive_hours(self):
        with self.assertRaises(ValueError):
            false_alarms_per_hour(np.array([]), 0)

    def test_event_recall(self):
        events = [(0.0, 3.0), (10.0, 13.0)]
        self.assertEqual(event_recall(events, np.array([3.1])), 0.5)     # 끝 + 0.2초 안

    def test_event_recall_rejects_late_alert(self):
        self.assertEqual(event_recall([(0.0, 3.0)], np.array([3.3])), 0.0)

    def test_event_recall_without_events_is_nan(self):
        self.assertTrue(math.isnan(event_recall([], np.array([1.0]))))

    def test_delay_is_measured_from_threshold(self):
        # 3초 기준: 시작 0초 + 3초 = 3초에 닿고 3.2초에 경고 → 지연 0.2초
        self.assertAlmostEqual(alert_delay([(0.0, 5.0)], np.array([3.2]), 3.0), 0.2)

    def test_delay_skips_short_and_missed_events(self):
        events = [(0.0, 1.0), (10.0, 15.0), (20.0, 25.0)]
        self.assertAlmostEqual(alert_delay(events, np.array([0.5, 13.5]), 3.0), 0.5)

    def test_delay_without_detection_is_nan(self):
        self.assertTrue(math.isnan(alert_delay([(0.0, 5.0)], np.array([]), 3.0)))

    def test_fps(self):
        self.assertAlmostEqual(fps(np.arange(31) / 30), 30.0)

    def test_fps_too_few_frames(self):
        self.assertTrue(math.isnan(fps(np.array([1.0]))))


class CompareHelperTest(unittest.TestCase):
    def test_markdown(self):
        text = to_markdown(pd.DataFrame({"method": ["규칙"], "f1": [0.91234], "ms": [float("nan")]}))
        self.assertEqual(text.splitlines(), ["| method | f1 | ms |", "| --- | --- | --- |", "| 규칙 | 0.912 | - |"])

    def test_simulate_fps_holds_previous_frame(self):
        from sequence_model import config

        df = pd.DataFrame(np.tile(np.arange(6.0)[:, None], (1, 11)), columns=list(config.FEATURE_COLUMNS))
        out = simulate_fps(df, 15, 30)
        self.assertEqual(out["ear"].tolist(), [0.0, 0.0, 2.0, 2.0, 4.0, 4.0])


if __name__ == "__main__":
    unittest.main()

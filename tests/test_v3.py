import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from sequence_model.build_windows import (channel_masks, clip_windows, end_indices, feature_matrix, fit_norm, frame_states, limit_aihub, normalize,
                           resample_index, window_label)
from sequence_model.models import CNN1D, LSTMClassifier
from dataset.convert_labels import check_driver_split, check_value, label_json, nitymed_masks, yawn_events
from eval.evaluate import ear48_closed, judge, merge, model_frames, pick

C, N, Y, H, U = config.CLOSED, config.NORMAL, config.YAWN, config.HEAD, config.UNCERTAIN


def features(n: int, fps: float = 25.0, found=None) -> pd.DataFrame:
    df = pd.DataFrame({"t": np.arange(n) / fps, "face_found": True if found is None else found})
    for name in config.FEATURES:
        if name != config.FACE_COLUMN:
            df[name] = 0.3
    return df


class PriorityTest(unittest.TestCase):
    def test_order_is_uncertain_eye_head_yawn(self):
        on = np.ones(1, bool)
        off = np.zeros(1, bool)
        self.assertEqual(frame_states({"uncertain": on, "eye_closed": on, "head_down": on, "yawn": on})[0], U)
        self.assertEqual(frame_states({"uncertain": off, "eye_closed": on, "head_down": on, "yawn": on})[0], C)
        self.assertEqual(frame_states({"uncertain": off, "eye_closed": off, "head_down": on, "yawn": on})[0], H)
        self.assertEqual(frame_states({"uncertain": off, "eye_closed": off, "head_down": off, "yawn": on})[0], Y)
        self.assertEqual(frame_states({"uncertain": off, "eye_closed": off, "head_down": off, "yawn": off})[0], N)


class WindowTest(unittest.TestCase):
    def test_window_label_rules(self):
        base = np.zeros(45, int)
        self.assertEqual(window_label(np.r_[base[:-1], C]), C)
        self.assertEqual(window_label(np.r_[base[:-1], H]), H)
        self.assertEqual(window_label(np.r_[base[:-1], U]), U)
        yawn = base.copy()
        yawn[:23] = Y
        self.assertEqual(window_label(yawn), Y)
        yawn[22] = N
        self.assertEqual(window_label(yawn), N)
        closed_middle = base.copy()
        closed_middle[10:30] = C
        self.assertEqual(window_label(closed_middle), N)

    def test_ends_every_half_second_and_full_window(self):
        ends = end_indices(150)
        self.assertEqual(ends[0], 44)
        # 15 FPS에서 0.5초 배수는 반 프레임에 걸릴 수 있어(3.5초 = 52.5프레임) 가장 가까운 프레임을 쓴다
        np.testing.assert_allclose((ends + 1) / config.FPS, np.arange(3.0, 10.0 + 1e-9, 0.5), atol=0.5 / config.FPS + 1e-9)
        self.assertEqual(len(np.unique(ends)), len(ends))

    def test_resample_25_to_15(self):
        grid, index = resample_index(np.arange(250) / 25)
        self.assertEqual(len(grid), 150)
        self.assertTrue(np.all(np.abs(grid - index / 25) <= 0.5 / 25 + 1e-9))

    def test_missing_frames_are_zero(self):
        df = features(4, found=[True, False, True, True])
        df.loc[3, "ear"] = np.nan
        X = feature_matrix(df)
        self.assertEqual(X[:, 0].tolist(), [1, 0, 1, 0])
        self.assertTrue(np.all(X[[1, 3], 1:] == 0))

    def test_too_many_missing_are_dropped(self):
        found = np.ones(250, bool)
        found[100:250] = False
        label = {"fps": 25.0, "frames": 250, "eye_closed": [], "yawn": [], "uncertain": [], "head_down": []}
        X, y, ends, counts = clip_windows(features(250, found=found), label)
        self.assertEqual(X.shape[1:], (45, 11))
        self.assertGreater(counts["미검출"], 0)
        self.assertEqual(counts["창"], counts["미검출"] + len(y))

    def test_window_labels_from_label_json(self):
        label = {"fps": 25.0, "frames": 250, "eye_closed": [[5.0, 7.0]], "yawn": [], "uncertain": [], "head_down": []}
        _, y, ends, _ = clip_windows(features(250), label)
        self.assertEqual(int(y[np.isclose(ends, 6.0)][0]), C)
        self.assertEqual(int(y[np.isclose(ends, 8.0)][0]), N)

    def test_aihub_share(self):
        info = pd.DataFrame({"split": ["train"] * 10 + ["val"], "source": ["nitymed"] * 2 + ["aihub089"] * 8 + ["nitymed"]})
        keep = limit_aihub(info, share=0.5)
        self.assertEqual(int(((info["source"] == "aihub089") & keep).sum()), 2)
        self.assertTrue(keep[-1])

    def test_norm_skips_missing_frames(self):
        X = np.zeros((2, 45, 11), np.float32)
        X[0, :, 0] = 1
        X[0, :, 1] = 2.0
        norm = fit_norm(X)
        self.assertAlmostEqual(norm["mean"][1], 2.0)
        self.assertEqual((norm["mean"][0], norm["std"][0]), (0.0, 1.0))
        self.assertTrue(np.all(normalize(X, norm)[1] == 0))


class ConvertTest(unittest.TestCase):
    def test_check_value(self):
        self.assertEqual(check_value("눈감음,고개숙임"), "눈감음")
        self.assertEqual(check_value("고개숙임"), "고개숙임")

    def test_head_hidden_removes_eye(self):
        segments = [{"start": 0, "end": 4, "state": "눈감김·고개숙임"}, {"start": 4, "end": 10, "state": "정상"}]
        heads = [{"start": 0, "end": 1, "state": "고개숙임(눈 안 보임)"}, {"start": 1, "end": 2, "state": "고개숙임"},
                 {"start": 2, "end": 10, "state": "숙이지 않음"}]
        m = nitymed_masks(segments, 250, 25.0, [], heads)
        self.assertFalse(m["eye_closed"][:25].any())
        self.assertTrue(m["eye_closed"][25:100].all())
        self.assertTrue(m["head_down"][:50].all())
        self.assertFalse(m["head_down"][50:].any())

    def test_checked_head_down_is_head_class(self):
        segments = [{"start": 0, "end": 2, "state": "눈감김·고개숙임"}]
        m = nitymed_masks(segments, 100, 25.0, [(0.0, 2.0, "고개숙임")], [])
        self.assertFalse(m["eye_closed"].any())
        self.assertTrue(m["head_down"][:50].all())
        self.assertFalse(m["uncertain"].any())

    def test_label_json_fields(self):
        m = nitymed_masks([{"start": 1.2, "end": 2.4, "state": "하품"}], 100, 25.0, [], [])
        label = label_json("c", "nitymed", "reviewed", "md01", 25.0, m, ("r",))
        for key in ("clip", "source", "driver_id", "fps", "frames", "conditions", "rules", "label_status", "eye_closed", "yawn", "uncertain", "head_down", "segments"):
            self.assertIn(key, label)
        self.assertEqual(label["yawn"], [{"start": 1.0, "end": 2.5, "hand": False}])
        masks = channel_masks(label, 100, 25.0)
        self.assertTrue(masks["yawn"][25:62].all())

    def test_driver_split_check(self):
        with self.assertRaises(SystemExit):
            check_driver_split(pd.DataFrame({"clip": ["a", "b"], "driver_id": ["d", "d"], "split": ["train", "test"]}))


class ModelTest(unittest.TestCase):
    def test_shapes(self):
        x = torch.zeros(3, 45, 11)
        self.assertEqual(tuple(CNN1D()(x).shape), (3, 4))
        self.assertEqual(tuple(LSTMClassifier()(x).shape), (3, 4))

    def test_frame_decisions(self):
        class Fake:
            def predict(self, windows):
                out = np.zeros((len(windows), 4))
                out[:, 0] = 1
                return out
        times, classes = model_frames(features(250), Fake())
        self.assertEqual(len(times), 150)
        self.assertTrue(np.all(classes[:44] == -1))
        self.assertTrue(np.all(classes[44:] == 0))


class EvalTest(unittest.TestCase):
    def test_ear48_and_merge(self):
        df = features(100)
        df.loc[40:60, "ear"] = 0.05
        self.assertTrue(ear48_closed(df, 25.0)[40:61].all())
        self.assertEqual(merge([(0.0, 1.0), (1.1, 2.0), (3.0, 4.0)]), [(0.0, 2.0), (3.0, 4.0)])

    def test_pick_and_judge(self):
        summary = pd.DataFrame({"방법": ["cnn1d", "lstm"], "눈감김_구간F1": [0.70, 0.72], "시드_표준편차": [0.03, 0.03]})
        self.assertEqual(pick(summary)[0], "cnn1d")
        rules = pd.DataFrame({"눈감김_구간F1": [0.4, 0.5], "3초_재현율": [0.5, 0.6], "3초_시간당_오경보": [5.0, 9.0]})
        good = pd.Series({"눈감김_구간F1": 0.7, "시드_표준편차": 0.02, "3초_재현율": 0.6, "3초_시간당_오경보": 4.0})
        self.assertTrue(judge(good, rules)[0])
        self.assertFalse(judge(pd.Series({**good, "3초_시간당_오경보": 5.0}), rules)[0])


class LiveTest(unittest.TestCase):
    class EndEar:
        """창 끝 프레임 EAR이 0.1 미만이면 눈감김, 창 평균 MAR이 0.5 이상이면 하품."""
        def predict(self, windows):
            out = np.zeros((len(windows), 4))
            closed = windows[:, -1, config.FEATURES.index("ear")] < 0.1
            yawn = windows[:, :, config.FEATURES.index("mar")].mean(axis=1) >= 0.5
            out[np.arange(len(windows)), np.where(closed, C, np.where(yawn, Y, N))] = 1
            return out

    def test_live_matches_file(self):
        from demo.webcam import LiveJudge
        df = features(400)
        df.loc[100:160, "ear"] = 0.05
        df.loc[250:330, "mar"] = 0.8
        df.loc[200:215, "face_found"] = False
        times, classes = model_frames(df, self.EndEar())
        judge = LiveJudge(self.EndEar())
        X = feature_matrix(df)
        for t, vector in zip(df["t"], X):
            judge.update(float(t), vector)
        live_t = np.array([h[0] for h in judge.history])
        live_c = np.array([h[1] for h in judge.history])
        n = min(len(times), len(live_t))
        np.testing.assert_allclose(live_t[:n], times[:n])
        np.testing.assert_array_equal(live_c[:n], classes[:n])
        self.assertTrue((live_c == C).any() and (live_c == Y).any())

    def test_live_missing_window_is_not_judged(self):
        from demo.webcam import LiveJudge
        found = np.ones(150, bool)
        found[80:150] = False
        df = features(150, found=found)
        judge = LiveJudge(self.EndEar())
        for t, vector in zip(df["t"], feature_matrix(df)):
            judge.update(float(t), vector)
        self.assertEqual(judge.history[-1][1], -1)

    def test_live_rule_uses_past_only(self):
        from demo.webcam import LiveRule
        rule = LiveRule()
        for i in range(150):            # 6초 동안 EAR 0.3
            rule.update(i / 25, True, 0.3)
        self.assertTrue(rule.ready)
        for i in range(150, 230):       # 3.2초 동안 EAR 0.1 → 졸음
            rule.update(i / 25, True, 0.1)
        self.assertEqual(rule.stage, 2)


class PostTest(unittest.TestCase):
    def test_grace_zero_is_old_timer_and_grace_bridges(self):
        from rule_based.alarm import ClosureTimer, timer_stages
        t = np.arange(60) / 15
        closed = np.ones(60, bool)
        closed[20:22] = False                   # 0.133초 끊김
        self.assertEqual(int(timer_stages(t, closed)[-1]), 1)                 # 끊긴 뒤 다시 잼: 2.47초 → 주의
        self.assertEqual(int(timer_stages(t, closed, grace_sec=0.2)[-1]), 2)  # 이어 가면 3.93초 → 졸음
        self.assertEqual(int(timer_stages(t, closed, grace_sec=0.1)[-1]), 1)  # 끊김(마지막 감음부터 0.133초)이 허용보다 길면 다시 잰다
        timer = ClosureTimer(grace_sec=0.2)
        for ti, ci in zip(t[:23], closed[:23]):
            timer.update(float(ti), bool(ci))
        self.assertAlmostEqual(timer.elapsed, t[22])

    def test_bridge_matches_timer(self):
        from sequence_model.postprocess import bridge
        from rule_based.alarm import timer_stages
        rng = np.random.default_rng(0)
        t = np.arange(300) / 15
        closed = rng.random(300) < 0.7
        for grace in (0.2, 0.33):
            bridged = bridge(t, closed, grace)
            a = timer_stages(t, closed, grace_sec=grace)
            b = timer_stages(t, bridged)
            # 타이머가 이어 가는 끊김과 구간을 잇는 끊김이 같아서 졸음 경고 수가 같다.
            # grace 타이머는 끊김 동안 시간을 늘리지 않으므로 경고가 같거나 grace 안에서 늦다
            from rule_based.alarm import stage_starts
            sa, sb = stage_starts(t, a), stage_starts(t, b)
            self.assertTrue(np.all(bridged >= closed))
            self.assertEqual(len(sa), len(sb))
            self.assertTrue(np.all((sa >= sb - 1e-9) & (sa - sb <= grace + 1e-9)))

    def test_hysteresis(self):
        from sequence_model.postprocess import Post, decide
        probs = np.array([[0.2, 0.7, 0.05, 0.05], [0.55, 0.35, 0.05, 0.05], [0.7, 0.2, 0.05, 0.05], [0.5, 0.45, 0.0, 0.05]])
        judged = np.ones(4, bool)
        self.assertEqual(decide(probs, judged).tolist(), [C, N, N, N])
        self.assertEqual(decide(probs, judged, Post(0.3, 0)).tolist(), [C, C, N, N])
        self.assertEqual(decide(probs, np.array([True, False, True, True]), Post(0.3, 0)).tolist(), [C, -1, N, N])

    def test_live_matches_file_with_post(self):
        from sequence_model.postprocess import Post
        from demo.webcam import LiveJudge

        class Soft:
            def predict(self, windows):
                ear = windows[:, -1, config.FEATURES.index("ear")]
                p = np.clip(1 - ear * 3, 0, 1)
                out = np.zeros((len(windows), 4))
                out[:, 1], out[:, 0] = p, 1 - p
                return out
        df = features(300)
        df["ear"] = 0.3 - 0.25 * np.sin(np.arange(300) / 9) ** 2
        post = Post(0.3, 0.2)
        times, classes = model_frames(df, Soft(), post)
        judge = LiveJudge(Soft(), post)
        for t, vector in zip(df["t"], feature_matrix(df)):
            judge.update(float(t), vector)
        live = np.array([h[1] for h in judge.history])
        n = min(len(live), len(classes))
        np.testing.assert_array_equal(live[:n], classes[:n])

    def test_enter_min_close_display(self):
        from sequence_model.postprocess import CANDIDATES, NONE, Post, decide, display_classes, drop_short
        probs = np.array([[0.4, 0.55, 0.05, 0.0], [0.2, 0.75, 0.05, 0.0], [0.5, 0.45, 0.05, 0.0], [0.1, 0.62, 0.0, 0.28]])
        judged = np.ones(4, bool)
        self.assertEqual(decide(probs, judged).tolist(), [C, C, N, C])
        self.assertEqual(decide(probs, judged, Post(enter=0.7)).tolist(), [N, C, N, H])     # 확률이 모자라면 다른 클래스
        self.assertEqual(decide(probs, judged, Post(exit=0.4, enter=0.7)).tolist(), [N, C, C, C])
        self.assertEqual(drop_short(np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], bool), 1 / 15, 0.5).tolist(),
                         [False, False, False] + [True] * 8)
        self.assertEqual(display_classes(np.array([C, C, N]), np.array([0.2, 0.6, 0.0])).tolist(), [N, C, N])
        self.assertEqual(CANDIDATES[0], NONE)
        self.assertEqual(len(CANDIDATES), 96)

    def test_live_matches_file_with_enter(self):
        from sequence_model.postprocess import Post
        from demo.webcam import LiveJudge

        class Soft:
            def predict(self, windows):
                ear = windows[:, -1, config.FEATURES.index("ear")]
                p = np.clip(1 - ear * 3, 0, 1)
                out = np.zeros((len(windows), 4))
                out[:, 1], out[:, 0] = p, 1 - p
                return out
        df = features(300)
        df["ear"] = 0.3 - 0.25 * np.sin(np.arange(300) / 9) ** 2
        post = Post(0.3, 0.2, 0.7, 0.5)
        times, classes = model_frames(df, Soft(), post)
        judge = LiveJudge(Soft(), post)
        for t, vector in zip(df["t"], feature_matrix(df)):
            judge.update(float(t), vector)
        live = np.array([h[1] for h in judge.history])
        n = min(len(live), len(classes))
        np.testing.assert_array_equal(live[:n], classes[:n])

    def test_augment(self):
        from sequence_model.train import augment
        rng = np.random.default_rng(1)
        z = torch.from_numpy(rng.normal(size=(64, 45, 11)).astype(np.float32))
        z[:, :, 0] = 1
        z[:5, 10:20] = 0                         # 이미 미검출 10프레임
        mean, std = torch.zeros(11), torch.ones(11)
        out = augment(z, mean, std, torch.Generator().manual_seed(0))
        self.assertEqual(tuple(out.shape), (64, 45, 11))
        missing = (out[:, :, 0] < 0.5).float().mean(dim=1)
        self.assertTrue(bool((missing <= config.MAX_MISSING + 1e-6).all()))
        self.assertTrue(bool((out[:, -1, 0] == 1).all()))
        self.assertTrue(bool((out[:5, 10:20] == 0).all()))


class Test2Test(unittest.TestCase):
    def test_scored_once_and_excludes_seen_videos(self):
        from unittest import mock
        from eval import score_user
        from auto_label.user_review.server import video_set
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "eval_test2").mkdir()
            (root / "eval_test2" / "report.md").write_text("done", encoding="utf-8")
            with mock.patch.object(config, "RUNS", root), self.assertRaises(SystemExit):
                score_user.score_test2()
            videos = root / "videos"
            videos.mkdir()
            for name in ("new_a", Path(config.USER_VIDEOS[0]).stem):
                (videos / f"{name}.mp4").write_bytes(b"")
            with mock.patch.object(config, "TEST2_DIR", videos):
                self.assertEqual([p.stem for p in video_set("test2")[0]], ["new_a"])


class DmdTest(unittest.TestCase):
    def test_dmd_masks(self):
        from dataset.convert_labels import dmd_driver, dmd_masks

        def act(kind, *spans):
            return {"type": kind, "frame_intervals": [{"frame_start": s, "frame_end": e} for s, e in spans]}
        ol = {"actions": {"0": act("eyes_state/close", (10, 19), (30, 33)), "1": act("eyes_state/closing", (5, 9)),
                          "2": act("eyes_state/opening", (20, 24)), "3": act("eyes_state/undefined", (40, 44)),
                          "4": act("yawning/Yawning with hand", (60, 79)), "5": act("yawning/Yawning without hand", (90, 120))}}
        m = dmd_masks(ol, 100, 10.0)
        self.assertEqual(np.flatnonzero(m["eye_closed"]).tolist(), list(range(10, 20)))     # closing·opening은 뜸, 0.4초 감음은 정상
        self.assertEqual(np.flatnonzero(m["uncertain"]).tolist(), list(range(40, 45)) + list(range(60, 80)))   # 손 하품은 불확실
        self.assertTrue(m["hand"][60:80].all() and not m["yawn"][60:80].any())
        self.assertTrue(m["yawn"][90:100].all() and not m["head_down"].any())
        self.assertEqual(dmd_driver("gB_10_s5_2019-03-12T10;35;20+01;00_rgb_face"), "dmd_gB_10")


class SpecFTest(unittest.TestCase):
    def test_carry_over_unknown(self):
        from rule_based.alarm import timer_stages
        t = np.arange(75) / 15
        closed = np.zeros(75, bool)
        unknown = np.zeros(75, bool)
        closed[:9], unknown[9:] = True, True              # 0.53초 감은 뒤 판정 불가 → 이어받아 졸음
        self.assertEqual(int(timer_stages(t, closed, unknown=unknown)[-1]), 2)
        closed[:], unknown[:] = False, False
        closed[:4], unknown[4:] = True, True              # 0.2초만 감은 뒤 판정 불가 → 다시 잰다
        self.assertEqual(int(timer_stages(t, closed, unknown=unknown)[-1]), 0)

    def test_yawn_closure_excluded_and_select_limit(self):
        from rule_based.alarm import timer_stages
        from eval.evaluate import select
        t = np.arange(60) / 15
        closed = np.ones(60, bool)
        yawn = np.zeros(60, bool)
        yawn[10:50] = True
        self.assertEqual(int(timer_stages(t, closed & ~yawn).max()), 0)
        table = pd.DataFrame({"_order": [0, 1, 2], "주의_시간당_오경보": [40.0, 59.0, 61.0],
                              "주의_재현율": [0.7, 0.8, 0.9], "눈감김_구간F1": [0.8, 0.8, 0.8]})
        self.assertEqual(int(select(table, 40.0 * 1.5)["_order"]), 1)

    def test_judge_every_02s(self):
        from eval.evaluate import model_probs

        class Count:
            def __init__(self):
                self.n = 0

            def predict(self, windows):
                out = np.zeros((len(windows), 4))
                out[:, 0] = np.arange(self.n, self.n + len(windows))
                self.n += len(windows)
                return out
        df = features(250)
        grid, probs, judged = model_probs(df, Count())
        after = probs[config.WINDOW_FRAMES - 1:, 0]
        self.assertTrue(np.all(after[:3] == after[0]) and after[3] != after[0])     # 3칸마다 새로 판정
        df.loc[200, "face_found"] = False
        _, _, judged = model_probs(df, Count())
        self.assertFalse(judged.all())          # 창 끝 프레임에 얼굴이 없으면 판정 불가

    def test_live_timer_matches_file(self):
        from sequence_model.postprocess import Post, display_classes, yawn_classes
        from rule_based.alarm import timer_stages
        from rule_based.alerts import MouthClosedWatch, YawnTracker
        from demo.demo_video import elapsed
        from eval.evaluate import grid_mouth_closed, grid_yawning, model_timer_unknown
        from demo.webcam import LiveJudge

        class Soft:
            relative = False

            def predict(self, windows):
                ear = windows[:, -1, config.FEATURES.index("ear")]
                pitch = windows[:, -1, config.FEATURES.index("pitch")]
                p = np.clip(1 - ear * 3, 0, 1)
                out = np.zeros((len(windows), 4))
                out[:, 1], out[:, 0], out[:, 3] = p, 1 - p, (pitch > 10) * 2.0     # pitch가 크면 고개숙임
                return out
        for relative, post in ((False, Post(grace=0.2)), (True, Post(grace=0.2)), (True, Post(grace=0.2, yawn_mode="union", carry_max=1.0))):
            Soft.relative = relative
            df = features(600)
            df["ear"] = 0.3 - 0.27 * (np.sin(np.arange(600) / 40) > 0.3)
            df["mar"] = 0.1 + 0.4 * ((np.arange(600) // 90) % 3 == 1)
            df["pitch"] = 20.0 * ((np.arange(600) % 200) > 170)
            df.loc[300:330, "face_found"] = False
            times, classes = model_frames(df, Soft(), post)
            unknown = model_timer_unknown(classes)
            stages = timer_stages(times, classes == C, grace_sec=0.2, unknown=unknown, carry_max_sec=post.carry_max)
            mouth_grid = grid_mouth_closed(df)
            shown = display_classes(yawn_classes(classes, mouth_grid, grid_yawning(df), post.yawn_mode),
                                    elapsed(times, classes == C, 0.2, unknown, post.carry_max), mouth_closed=mouth_grid)
            judge, mouth, tracker = LiveJudge(Soft(), post), MouthClosedWatch(), YawnTracker()
            for t, vector, mar, found in zip(df["t"], feature_matrix(df), df["mar"], df["face_found"]):
                m = mar if found else np.nan
                judge.update(float(t), vector, mouth.update(float(t), m), tracker.update(float(t), m))
            live = np.array([h[2] for h in judge.history])
            live_shown = np.array([h[3] for h in judge.history])
            n = min(len(live), len(stages))
            np.testing.assert_array_equal(live[:n], stages[:n])
            np.testing.assert_array_equal(live_shown[:n], shown[:n])

    def test_head_down_carries_and_rule_inputs(self):
        from rule_based.alarm import timer_stages
        from eval.evaluate import model_timer_unknown, rule_timer_inputs
        t = np.arange(75) / 15
        classes = np.full(75, N)
        classes[:9], classes[9:] = C, H                  # 0.53초 감은 뒤 고개숙임 → 이어서 세 졸음
        self.assertEqual(int(timer_stages(t, classes == C, unknown=model_timer_unknown(classes))[-1]), 2)
        classes[:4], classes[4:] = C, H                  # 0.2초만 감은 뒤 숙임 → 이어서 세지 않음
        self.assertEqual(int(timer_stages(t, classes == C, unknown=model_timer_unknown(classes))[-1]), 0)
        df = features(10)
        df["pitch"] = [0, 0, 20, 20, 0, 0, 0, 0, 0, 0]
        df.loc[5, "face_found"] = False
        self.assertEqual(np.flatnonzero(rule_timer_inputs(df)["unknown"]).tolist(), [2, 3, 5])
        self.assertNotIn("yawn", rule_timer_inputs(df))  # 최종 기획서: 하품 중 감음도 센다

    def test_session_baseline_and_display_yawn(self):
        from sequence_model.postprocess import display_classes
        from landmarks.session import BASE_SEC, relative_matrix
        t = np.arange(2000) / 25
        X = np.zeros((2000, 11), np.float32)
        X[:, 0] = 1
        X[:, config.FEATURES.index("ear")] = 0.2
        X[:, config.FEATURES.index("pitch")] = 8.0
        X[1700:, config.FEATURES.index("pitch")] = 30.0   # 60초 뒤의 값은 평소 값에 들어가지 않는다
        R = relative_matrix(t, X)
        self.assertAlmostEqual(float(R[10, config.FEATURES.index("ear")]), 1.0, places=5)
        self.assertAlmostEqual(float(R[100, config.FEATURES.index("pitch")]), 0.0, places=5)
        self.assertAlmostEqual(float(R[1800, config.FEATURES.index("pitch")]), 22.0, places=4)
        self.assertGreater(BASE_SEC, t[1500] - 1)
        part = relative_matrix(t[:500], X[:500])          # 지금까지 프레임만 쓴다
        np.testing.assert_allclose(part, R[:500], atol=1e-6)
        shown = display_classes(np.array([Y, Y, Y]), np.zeros(3), mouth_closed=np.array([0.0, 0.4, 0.6]))
        self.assertEqual(shown.tolist(), [Y, Y, N])

    def test_yawn_modes_and_carry_cap(self):
        from sequence_model.postprocess import shown_yawn, yawn_classes
        from rule_based.alarm import timer_stages
        classes = np.array([Y, Y, N, N, C])
        mouth = np.array([0.0, 0.2, 0.0, 0.0, 0.0])           # 두 번째 프레임은 입을 다문 지 0.2초
        rule = np.array([False, True, True, False, True])
        self.assertEqual(yawn_classes(classes, mouth, rule, "model").tolist(), [Y, Y, N, N, C])
        self.assertEqual(yawn_classes(classes, mouth, rule, "gate").tolist(), [Y, N, N, N, C])
        self.assertEqual(yawn_classes(classes, mouth, rule, "union").tolist(), [Y, Y, Y, N, C])
        self.assertEqual(yawn_classes(classes, mouth, rule, "rule").tolist(), [N, Y, Y, N, C])
        self.assertEqual(shown_yawn(classes, np.array([0.0, 0.6, 0, 0, 0]), rule, "model").tolist(), [True, False, False, False, False])
        t = np.arange(90) / 15
        closed, unknown = np.zeros(90, bool), np.zeros(90, bool)
        closed[:9], unknown[9:] = True, True                  # 0.53초 감은 뒤 고개숙임 5.4초
        self.assertEqual(int(timer_stages(t, closed, unknown=unknown)[-1]), 2)
        self.assertEqual(int(timer_stages(t, closed, unknown=unknown, carry_max_sec=1.0)[-1]), 0)       # 1초 넘게 이어받으면 다시 잰다
        self.assertEqual(int(timer_stages(t, closed, unknown=unknown, carry_max_sec=3.0)[40]), 1)       # 상한 안에서는 이어서 센다

    def test_dmd_yawn_trim(self):
        from dataset.convert_labels import trim_yawn
        yawn = np.zeros(100, bool)
        yawn[10:90] = True
        mar = np.full(100, 0.1)
        mar[30:50], mar[55:70] = 0.6, 0.6                 # 입 벌림 두 번, 사이 5프레임(0.2초) 끊김
        out = trim_yawn(yawn, mar, 25.0)
        self.assertEqual(np.flatnonzero(out).tolist(), list(range(30, 70)))

    def test_yawn_tracker_and_alerts(self):
        from rule_based.alerts import UnknownWatch, YawnRepeat, YawnTracker
        from rule_based.yawn import yawning
        t = np.arange(200) / 15
        mar = 0.1 + 0.5 * ((np.arange(200) // 30) % 2 == 1)
        tracker = YawnTracker()
        self.assertEqual([tracker.update(float(a), float(b)) for a, b in zip(t, mar)], yawning(t, mar).tolist())
        rep = YawnRepeat()
        fired = []
        for start in (0, 60, 120, 150, 300):             # 하품 다섯 번(각 1.5초)
            for dt in np.arange(0, 1.6, 0.1):
                if rep.update(start + dt, True):
                    fired.append(start)
            rep.update(start + 2, False)
        self.assertEqual(fired, [120, 300])              # 세 번째에 알림, 2분 쿨다운 동안은 안 알림
        watch = UnknownWatch()
        hits = [watch.update(x / 10, True) for x in range(40)]
        self.assertEqual(hits.index(True), 30)


class RecordTest(unittest.TestCase):
    def test_timed_writer_and_convert(self):
        import cv2
        from demo.cfr import TimedWriter, convert
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "src.mp4"
            writer = TimedWriter(src, fps=30)
            times = [0.0, 0.05, 0.12, 0.5, 0.52, 1.0]            # 불규칙한 시각, 값은 시각마다 다른 밝기
            for i, t in enumerate(times):
                writer.write(np.full((48, 64, 3), 40 * i, np.uint8), t)
            writer.close()
            self.assertEqual(writer.frames, 31)                  # 0~1초 = 30 FPS 칸 31개
            cap = cv2.VideoCapture(str(src))
            levels = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                levels.append(frame.mean())
            cap.release()
            self.assertEqual(len(levels), 31)
            expect = [40 * (np.searchsorted(times, k / 30 + 1e-9, side="right") - 1) for k in range(31)]
            self.assertTrue(np.all(np.abs(np.array(levels) - np.array(expect)) < 12))   # 칸마다 그 시각의 최신 프레임
            out = convert(src, Path(folder) / "dst.mp4", fps=15)
            self.assertEqual(out["frames"], 16)

    def test_record_blind_never_loads_model(self):
        from unittest import mock
        import cv2
        from demo import webcam
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "in.mp4"
            w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for i in range(20):
                w.write(np.full((48, 64, 3), i * 10, np.uint8))
            w.release()
            with mock.patch("sequence_model.models.Predictor", side_effect=AssertionError("모델을 불렀다")), \
                    mock.patch("landmarks.extract.LandmarkExtractor", side_effect=AssertionError("추출기를 불렀다")):
                result = webcam.record_blind(str(src), Path(folder) / "out.mp4", show=False, max_seconds=None)
            self.assertEqual(result["frames"], 20)
            self.assertTrue((Path(folder) / "out.mp4").exists())

    def test_webcam_video_set(self):
        from unittest import mock
        from auto_label.user_review.server import video_set
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "webcam_1.mp4").write_bytes(b"")
            with mock.patch.object(config, "WEBCAM_DIR", Path(folder)):
                videos, done, _ = video_set("webcam")
            self.assertEqual([p.name for p in videos], ["webcam_1.mp4"])
            self.assertEqual(done, config.WEBCAM_LABEL_DIR)


class HeadReviewTest(unittest.TestCase):
    def test_saves_three_states_only(self):
        from auto_label.head_review.server import ReviewStore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = ReviewStore(["c"], root, root / "done", root / "draft")
            store.save("c", [{"start": 0, "end": 1, "state": "고개숙임"}, {"start": 1, "end": 2, "state": "고개숙임(눈 안 보임)"}], True)
            self.assertEqual(json.loads((root / "done" / "c.json").read_text(encoding="utf-8"))["segments"][1]["state"], "고개숙임(눈 안 보임)")
            with self.assertRaises(ValueError):
                store.save("c", [{"start": 0, "end": 1, "state": "눈감김·고개숙임"}], True)


class UserReviewTest(unittest.TestCase):
    def test_saves_six_states_and_gives_fps(self):
        from auto_label.head_review.server import ReviewStore
        from auto_label.user_review.server import STATE_BY_NAME
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = ReviewStore(["v"], root, root / "done", root / "draft", states=STATE_BY_NAME, fps={"v": 22.2})
            names = list(STATE_BY_NAME)
            store.save("v", [{"start": i, "end": i + 1, "state": name} for i, name in enumerate(names)], True)
            saved = json.loads((root / "done" / "v.json").read_text(encoding="utf-8"))["segments"]
            self.assertEqual([s["state"] for s in saved], names)
            self.assertEqual(store.listing(), [{"clip": "v", "status": "done", "fps": 22.2}])
            with self.assertRaises(ValueError):
                store.save("v", [{"start": 0, "end": 1, "state": "숙이지 않음"}], True)

    def test_user_masks(self):
        from eval.score_user import user_masks
        segments = [{"start": 0, "end": 1, "state": "하품"}, {"start": 1, "end": 1.4, "state": "눈감김"},
                    {"start": 1.4, "end": 2, "state": "하품"}, {"start": 2, "end": 3, "state": "고개숙임(눈 안 보임)"},
                    {"start": 3, "end": 4, "state": "불확실"}, {"start": 4, "end": 5, "state": "정상"}]
        m = user_masks(segments, 125, 25.0)
        self.assertTrue(m["yawn"][:50].all())               # 감음만 끼어 끊긴 하품은 잇는다
        self.assertTrue(m["eye_closed"][25:35].all() and not m["eye_closed"][35:].any())
        self.assertTrue(m["head_down"][50:75].all() and not m["head_down"][75:].any())
        self.assertTrue(m["uncertain"][75:100].all() and not m["uncertain"][:75].any())
        states = frame_states(m)
        self.assertEqual(int(states[30]), C)                  # 하품 중 감음은 눈감김
        self.assertEqual(int(states[60]), H)
        self.assertEqual(int(states[110]), N)


if __name__ == "__main__":
    unittest.main()

# 합성 데이터 스모크 테스트. 패턴이 뚜렷한 가짜 데이터라 점수는 성능 수치가 아니다.

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from eval import window_eval as ev
from sequence_model import train_lstm as tl
from sequence_model.build_windows import CLASS_NAMES, FEATURE_COLUMNS, WINDOW_FRAMES
from sequence_model.models import build_model


def synthetic_windows(rng: np.random.Generator, labels: list[int]) -> np.ndarray:
    """정답별로 ear·mar·pitch가 달라지는 창 (N, 45, 11). 모두 얼굴이 검출된 프레임이다."""
    X = rng.normal(0, 0.05, (len(labels), WINDOW_FRAMES, len(FEATURE_COLUMNS))).astype(np.float32)
    X[..., FEATURE_COLUMNS.index("face")] = 1.0
    for i, label in enumerate(labels):
        X[i, :, FEATURE_COLUMNS.index("ear")] += 1.0 if label == 1 else 0.0
        X[i, :, FEATURE_COLUMNS.index("mar")] += 1.0 if label == 2 else 0.0
        X[i, :, FEATURE_COLUMNS.index("pitch")] += 1.0 if label == 3 else 0.0
    return X


def write_bundle(folder: Path) -> dict:
    """train 40창, val 16창, test 8창. test 행의 X는 NaN이라 학습에 섞이면 결과가 NaN이 된다."""
    rng = np.random.default_rng(0)
    y_train, y_val, y_test = [0, 1, 2, 3] * 10, [0, 1, 2, 3] * 4, [0, 1, 2, 3] * 2
    y = np.array(y_train + y_val + y_test)
    split = np.array(["train"] * 40 + ["val"] * 16 + ["test"] * 8)
    X = np.concatenate([synthetic_windows(rng, y_train), synthetic_windows(rng, y_val), synthetic_windows(rng, y_test)])
    clip = np.array([f"c{i // 8}" for i in range(len(y))])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "windows.npz", X=X, y=y, clip=clip, driver_id=clip, source=np.full(len(y), "nitymed"), split=split,
             start_sec=np.array([(i % 8) * 0.5 for i in range(len(y))]))
    columns = [name for name in FEATURE_COLUMNS if name != "face"]
    np.savez(folder / "norm.npz", mean=np.zeros(10), std=np.ones(10), columns=np.array(columns))
    return {"X": X, "y": y, "split": split}


class Base(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.bundle = self.dir / "bundle"
        self.data = write_bundle(self.bundle)

    def call(self, *extra: str, tag: str = "t", device: tuple[str, ...] = ("--device", "cpu")) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            return tl.main(["--tag", tag, "--bundle", str(self.bundle), *device, "--checkpoints-dir", str(self.dir / "ck"),
                            "--runs-dir", str(self.dir / "runs"), *extra])


class SmokeTest(Base):
    def test_outputs_are_marked_cpu(self):
        result = self.call("--epochs", "2")
        self.assertEqual(result["device"], "cpu")
        self.assertTrue((self.dir / "ck" / "lstm_t_seed42_cpu.pt").exists())
        probs = np.load(self.dir / "runs" / "lstm_t_seed42_cpu_val_probs.npy")
        self.assertEqual(probs.shape, (16, 4))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

        saved = json.loads((self.dir / "runs" / "lstm_t_cpu_results.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["device"], "cpu")
        self.assertEqual(saved["seeds"][0]["seed"], 42)
        self.assertEqual(saved["class_counts"]["val"][CLASS_NAMES[3]], 4)
        report = (self.dir / "runs" / "lstm_t_cpu_results.md").read_text(encoding="utf-8")
        self.assertIn("장치: cpu", report)
        self.assertIn("| 시드 | 멈춘 epoch | macro F1 | 정상 | 눈 감김 | 하품 | 고개 숙임(참고) |", report)
        self.assertIn("가장 좋았던 epoch", report)

    def test_test_rows_are_not_used_without_final(self):
        # test 행을 NaN으로 바꾼 번들로 학습해도 가중치와 확률이 유한값이어야 한다
        X = self.data["X"].copy()
        X[self.data["split"] == "test"] = np.nan
        with np.load(self.bundle / "windows.npz") as old:
            np.savez(self.bundle / "windows.npz", **{**{k: old[k] for k in old.files}, "X": X})
        result = self.call("--epochs", "2")
        self.assertTrue(np.isfinite(np.load(self.dir / "runs" / "lstm_t_seed42_cpu_val_probs.npy")).all())
        self.assertEqual(result["class_counts"].keys(), {"train", "val"})
        self.assertFalse(list((self.dir / "runs").glob("*test*")))

    def test_same_seed_repeats_on_cpu(self):
        self.call("--epochs", "2", tag="a")
        self.call("--epochs", "2", tag="b")
        np.testing.assert_array_equal(np.load(self.dir / "runs" / "lstm_a_seed42_cpu_val_probs.npy"),
                                      np.load(self.dir / "runs" / "lstm_b_seed42_cpu_val_probs.npy"))

    def test_seed_table_has_a_row_per_seed(self):
        result = self.call("--epochs", "1", "--seeds", "1", "2")
        self.assertEqual([row["seed"] for row in result["seeds"]], [1, 2])
        self.assertTrue((self.dir / "ck" / "lstm_t_seed2_cpu.pt").exists())

    def test_tag_is_not_overwritten(self):
        self.call("--epochs", "1")
        with self.assertRaises(SystemExit) as caught:
            self.call("--epochs", "1")
        self.assertIn("새 --tag", str(caught.exception))

    def test_probs_work_with_evaluate(self):
        self.call("--epochs", "2")
        with contextlib.redirect_stdout(io.StringIO()):
            result = ev.main(["--windows", str(self.bundle / "windows.npz"), "--out-dir", str(self.dir / "runs"),
                              "--probs", str(self.dir / "runs" / "lstm_t_seed42_cpu_val_probs.npy")])
        self.assertEqual(result["n_windows"], 16)
        self.assertTrue((self.dir / "runs" / "eval_lstm_t_seed42_cpu_val_probs_val.json").exists())


class DeviceTest(Base):
    def test_stops_without_gpu_by_default(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(SystemExit) as caught:
                self.call(device=())
        self.assertIn("--device cpu", str(caught.exception))
        self.assertFalse((self.dir / "ck").exists())

    def test_default_device_is_cuda(self):
        self.assertEqual(tl.parse_args(["--tag", "t"]).device, "cuda")


class NormalizeTest(unittest.TestCase):
    def test_by_hand(self):
        columns = len(FEATURE_COLUMNS)
        X = np.zeros((1, 2, columns), dtype=np.float32)
        X[0, 0] = [1.0, *range(2, 12)]      # 검출 프레임: face 1, 나머지 2~11
        X[0, 1] = [0.0, *range(2, 12)]      # 미검출 프레임
        mean, std = np.full(10, 2.0), np.full(10, 2.0)
        out = tl.normalize(X, mean, std)
        self.assertEqual(out[0, 0, 0], 1.0)     # face는 정규화하지 않는다
        np.testing.assert_allclose(out[0, 0, 1:], [(v - 2) / 2 for v in range(2, 12)])
        np.testing.assert_array_equal(out[0, 1], np.zeros(columns))

    def test_input_is_unchanged(self):
        X = np.ones((1, 1, len(FEATURE_COLUMNS)), dtype=np.float32)
        tl.normalize(X, np.full(10, 2.0), np.full(10, 2.0))
        self.assertEqual(X[0, 0, 1], 1.0)


class BundleCheckTest(Base):
    def test_wrong_window_shape(self):
        with np.load(self.bundle / "windows.npz") as old:
            np.savez(self.bundle / "windows.npz", **{**{k: old[k] for k in old.files}, "X": old["X"][:, :30]})
        with self.assertRaises(SystemExit) as caught:
            tl.load_bundle(self.bundle, ("train",))
        self.assertIn("모양", str(caught.exception))

    def test_norm_columns_must_match(self):
        np.savez(self.bundle / "norm.npz", mean=np.zeros(10), std=np.ones(10), columns=np.array(["x"] * 10))
        with self.assertRaises(SystemExit) as caught:
            tl.load_bundle(self.bundle, ("train",))
        self.assertIn("norm.npz", str(caught.exception))

    def test_only_requested_splits_are_returned(self):
        data = tl.load_bundle(self.bundle, ("train", "val"))
        self.assertNotIn("test", data)
        self.assertEqual(len(data["train"][0]), 40)

    def test_missing_split(self):
        with self.assertRaises(SystemExit):
            tl.load_bundle(self.bundle, ("calibration",))


class AverageTest(unittest.TestCase):
    def test_class_weights(self):
        # 8창: 정상 4, 눈 감김 2, 하품 2, 고개 숙임 0(1개로 침) → 8/(4×개수)
        weights = tl.class_weights(np.array([0, 0, 0, 0, 1, 1, 2, 2]))
        np.testing.assert_allclose(weights.numpy(), [0.5, 1.0, 1.0, 2.0])

    def test_fixed_averages_fill_missing_with_zero(self):
        result = tl.fixed_averages({0: 0.9, 1: 0.6})
        self.assertAlmostEqual(result["macro_f1_4class"], (0.9 + 0.6) / 4)
        self.assertAlmostEqual(result["macro_f1_3class"], (0.9 + 0.6) / 3)

    def test_fixed_3class_leaves_out_head_down(self):
        result = tl.fixed_averages({0: 0.9, 1: 0.6, 2: 0.3, 3: 0.0})
        self.assertAlmostEqual(result["macro_f1_3class"], 0.6)
        self.assertAlmostEqual(result["macro_f1_4class"], 0.45)


class EarlyStoppingTest(Base):
    def run_with_scores(self, scores: list[float]) -> dict:
        metrics = [{"macro_f1": s, "accuracy": s, "f1_per_class": {0: s}} for s in scores]
        with mock.patch.object(tl, "classification_metrics", side_effect=metrics):
            return self.call("--epochs", "50", "--seeds", "42")["seeds"][0]     # 점수 목록은 시드 1개 분량이다

    def test_stops_eight_epochs_after_best(self):
        row = self.run_with_scores([0.1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(row["best_epoch"], 2)
        self.assertEqual(row["stopped_epoch"], 10)
        self.assertEqual(row["macro_f1"], 0.5)

    def test_improvement_resets_patience(self):
        row = self.run_with_scores([0.1] * 8 + [0.2] + [0.1] * 8)
        self.assertEqual(row["best_epoch"], 9)
        self.assertEqual(row["stopped_epoch"], 17)

    def test_max_epoch_stops_without_patience(self):
        self.call("--epochs", "2", tag="short")
        saved = json.loads((self.dir / "runs" / "lstm_short_cpu_results.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["seeds"][0]["stopped_epoch"], 2)

    def test_best_epoch_weights_are_saved(self):
        # 2 epoch 뒤 점수가 떨어지면 저장 가중치는 2 epoch 것이어야 한다. 확률은 그 가중치로 낸다
        self.run_with_scores([0.1, 0.5] + [0.2] * 8)
        checkpoint = torch.load(self.dir / "ck" / "lstm_t_seed42_cpu.pt", weights_only=True)
        self.assertEqual(checkpoint["best_epoch"], 2)
        model = tl.LSTMClassifier(11, 4, hidden=checkpoint["hidden"], dropout=checkpoint["dropout"])
        model.load_state_dict(checkpoint["state_dict"])
        X_val = tl.normalize(self.data["X"][self.data["split"] == "val"], np.zeros(10), np.ones(10))
        np.testing.assert_allclose(tl.predict_proba(model, X_val, 64, "cpu"),
                                   np.load(self.dir / "runs" / "lstm_t_seed42_cpu_val_probs.npy"), atol=1e-6)


class FileNameTest(Base):
    def test_cpu_is_not_added_twice(self):
        self.call("--epochs", "1", tag="v1_cpu")
        self.assertTrue((self.dir / "runs" / "lstm_v1_cpu_results.json").exists())
        self.assertTrue((self.dir / "runs" / "lstm_v1_cpu_seed42_val_probs.npy").exists())
        self.assertTrue((self.dir / "ck" / "lstm_v1_cpu_seed42.pt").exists())
        self.assertFalse(list((self.dir / "runs").glob("*cpu_cpu*")))

    def test_final_finds_the_same_names(self):
        self.call("--epochs", "1", tag="v1_cpu")
        self.call("--final", tag="v1_cpu")
        self.assertTrue((self.dir / "runs" / "lstm_v1_cpu_final.json").exists())

    def test_gpu_gets_no_suffix(self):
        args = tl.parse_args(["--tag", "v1", "--device", "cuda"])
        self.assertEqual(tl.results_path(args, "results", "json").name, "lstm_v1_results.json")

    def test_same_results_name_is_not_overwritten(self):
        # --tag v1 --device cpu와 --tag v1_cpu --device cpu는 결과 이름이 같다
        self.call("--epochs", "1", tag="v1")
        with self.assertRaises(SystemExit) as caught:
            self.call("--epochs", "1", tag="v1_cpu")
        self.assertIn("이미 있습니다", str(caught.exception))


class PoolTest(Base):
    def test_mean_is_the_existing_model(self):
        self.assertIs(type(tl.make_model("mean", 11, 4, 8, 0.0)), tl.LSTMClassifier)

    def test_last_uses_only_the_last_step(self):
        torch.manual_seed(0)
        model = tl.make_model("last", 11, 4, 8, 0.0).eval()
        x = torch.randn(2, 45, 11)
        output, _ = model.lstm(x)
        torch.testing.assert_close(model(x), model.head(output[:, -1]))

    def test_mean_is_default_and_is_shown(self):
        result = self.call("--epochs", "1")
        self.assertEqual(result["settings"]["pool"], "mean")
        report = (self.dir / "runs" / "lstm_t_cpu_results.md").read_text(encoding="utf-8")
        self.assertIn("풀링: mean", report)
        self.assertIn("고개 숙임(참고) | 풀링 |", report)
        self.assertEqual(torch.load(self.dir / "ck" / "lstm_t_seed42_cpu.pt", weights_only=True)["model"], "lstm")

    def test_last_is_shown_and_saved(self):
        result = self.call("--epochs", "1", "--pool", "last")
        self.assertEqual(result["seeds"][0]["pool"], "last")
        report = (self.dir / "runs" / "lstm_t_cpu_results.md").read_text(encoding="utf-8")
        self.assertIn("풀링: last", report)
        checkpoint = torch.load(self.dir / "ck" / "lstm_t_seed42_cpu.pt", weights_only=True)
        self.assertEqual((checkpoint["pool"], checkpoint["model"]), ("last", "lstm_last"))

    def test_predictor_refuses_last(self):
        with self.assertRaises(ValueError):
            build_model(tl.POOLS["last"])

    def test_final_uses_checkpoint_pool(self):
        self.call("--epochs", "1", "--pool", "last")
        with mock.patch.object(tl, "make_model", wraps=tl.make_model) as made:
            result = self.call("--final")
        self.assertEqual(made.call_args.args[0], "last")
        self.assertEqual(result["seed"]["pool"], "last")
        self.assertIn("풀링: last", (self.dir / "runs" / "lstm_t_cpu_final.md").read_text(encoding="utf-8"))


class NormCheckTest(unittest.TestCase):
    def setUp(self):
        # 검출 프레임 5개: 열 번호 × 1이 3개, 열 번호 × 3이 2개. 미검출 프레임은 값이 커도 계산에서 빠져야 한다
        scale = np.arange(1, 11, dtype=np.float32)
        self.X = np.ones((2, 3, 11), dtype=np.float32)
        self.X[0, :, 1:] = scale
        self.X[1, :, 1:] = scale * 3
        self.X[1, 2, 0] = 0.0
        self.X[1, 2, 1:] = 100.0
        frames = self.X[self.X[..., 0] > 0.5]
        self.mean, self.std = frames[:, 1:].mean(axis=0), frames[:, 1:].std(axis=0)

    def test_stats_after_normalizing(self):
        check = tl.norm_check(self.X, self.X, self.mean, self.std)
        self.assertTrue(check["matches_saved"])
        self.assertTrue(check["face_unchanged"])
        self.assertTrue(check["missing_all_zero"])
        self.assertEqual((check["missing_frames"], check["total_frames"]), (1, 6))
        self.assertEqual(check["face_values"], [0.0, 1.0])
        self.assertEqual((check["nan_raw"], check["nan_normalized"]), (0, 0))
        self.assertEqual([c["column"] for c in check["columns"]], list(FEATURE_COLUMNS[1:]))
        for column in check["columns"]:
            self.assertAlmostEqual(column["train_mean"], 0.0, places=5)
            self.assertAlmostEqual(column["train_std"], 1.0, places=5)

    def test_by_hand(self):
        # 첫 열: 값 1, 1, 1, 3, 3 → 평균 1.8
        self.assertAlmostEqual(tl.norm_check(self.X, self.X, self.mean, self.std)["columns"][0]["raw_mean"], 1.8, places=5)

    def test_wrong_column_order_is_caught(self):
        check = tl.norm_check(self.X, self.X, self.mean[::-1].copy(), self.std[::-1].copy())
        self.assertFalse(check["matches_saved"])


class DiagnoseTest(Base):
    def run_diagnose(self, *extra: str) -> dict:
        return self.call("--epochs", "3", "--patience", "50", "--diagnose", "--seeds", "42", "43", *extra)

    def test_history_has_a_row_per_epoch(self):
        for row in self.run_diagnose()["seeds"]:
            self.assertEqual([h["epoch"] for h in row["history"]], [1, 2, 3])
            self.assertEqual(set(row["history"][0]), {"epoch", "train_loss", "train_macro_f1", "train_eye_f1", "val_loss", "val_macro_f1", "val_eye_f1"})
            self.assertAlmostEqual(row["history"][row["best_epoch"] - 1]["val_macro_f1"], row["macro_f1"])

    def test_confusion_matrix_agrees_with_metrics(self):
        y_val = self.data["y"][self.data["split"] == "val"]
        for row in self.run_diagnose()["seeds"]:
            matrix = np.array(row["confusion_matrix"])
            self.assertEqual(matrix.sum(axis=1).tolist(), np.bincount(y_val, minlength=4).tolist())
            self.assertAlmostEqual(np.trace(matrix) / matrix.sum(), row["accuracy"])

    def test_confusion_matrix_by_hand(self):
        self.assertEqual(tl.confusion_matrix(np.array([0, 0, 1, 3]), np.array([0, 1, 1, 2])),
                         [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    def test_saved_with_weights_and_spread(self):
        saved = self.run_diagnose()["diagnostics"]
        np.testing.assert_allclose(list(saved["class_weights"].values()),
                                   tl.class_weights(self.data["y"][self.data["split"] == "train"].astype(np.int64)).numpy())
        report = (self.dir / "runs" / "lstm_t_cpu_results.md").read_text(encoding="utf-8")
        for title in ("## 정규화 점검", "## 클래스 가중치", "## 시드별 변동", "## 시드 43 혼동행렬", "## 시드 43 epoch 기록"):
            self.assertIn(title, report)

    def test_seed_spread_by_hand(self):
        rows = [{"macro_f1": 0.2, "f1_per_class": {name: 0.0 for name in CLASS_NAMES}}, {"macro_f1": 0.4, "f1_per_class": {}}]
        self.assertEqual(tl.seed_spread(rows)[0][0], "macro F1")
        np.testing.assert_allclose(tl.seed_spread(rows)[0][1:], [0.3, 0.1, 0.2, 0.4])

    def test_does_not_change_training(self):
        self.call("--epochs", "3", "--patience", "50", tag="plain")
        self.call("--epochs", "3", "--patience", "50", "--diagnose", tag="diag")
        np.testing.assert_array_equal(np.load(self.dir / "runs" / "lstm_plain_seed42_cpu_val_probs.npy"),
                                      np.load(self.dir / "runs" / "lstm_diag_seed42_cpu_val_probs.npy"))

    def test_nothing_extra_without_option(self):
        result = self.call("--epochs", "1")
        self.assertNotIn("history", result["seeds"][0])
        self.assertNotIn("diagnostics", result)

    def test_not_with_final(self):
        with self.assertRaises(SystemExit) as caught:
            self.call("--final", "--diagnose")
        self.assertIn("--final", str(caught.exception))


class FinalTest(Base):
    def test_needs_trained_weights(self):
        with self.assertRaises(SystemExit) as caught:
            self.call("--final")
        self.assertIn("먼저 학습", str(caught.exception))
        self.assertFalse(list((self.dir / "runs").glob("*test*")) if (self.dir / "runs").exists() else [])

    def test_writes_test_probs_once(self):
        self.call("--epochs", "1")
        result = self.call("--final")
        self.assertEqual(result["device"], "cpu")
        probs = np.load(self.dir / "runs" / "lstm_t_seed42_cpu_test_probs.npy")
        self.assertEqual(probs.shape, (8, 4))
        self.assertTrue((self.dir / "runs" / "lstm_t_cpu_final.json").exists())
        with self.assertRaises(SystemExit) as caught:
            self.call("--final")
        self.assertIn("한 번만", str(caught.exception))

    def test_seed_is_chosen_by_val(self):
        self.call("--epochs", "1", "--seeds", "1", "2")
        path = self.dir / "runs" / "lstm_t_cpu_results.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["seeds"][0]["macro_f1"], saved["seeds"][1]["macro_f1"] = 0.1, 0.9
        path.write_text(json.dumps(saved), encoding="utf-8")
        self.assertEqual(self.call("--final")["seed"]["seed"], 2)
        self.assertTrue((self.dir / "runs" / "lstm_t_seed2_cpu_test_probs.npy").exists())

    def test_does_not_change_weights(self):
        self.call("--epochs", "1")
        before = (self.dir / "ck" / "lstm_t_seed42_cpu.pt").read_bytes()
        self.call("--final")
        self.assertEqual(before, (self.dir / "ck" / "lstm_t_seed42_cpu.pt").read_bytes())


if __name__ == "__main__":
    unittest.main()

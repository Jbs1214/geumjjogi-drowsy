# 합성 데이터로 묶음 구조를 확인한다. 점수는 성능 수치가 아니다.

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from eval import evaluate as ev
from sequence_model import share_lstm as sl
from sequence_model import train_lstm as tl
from sequence_model.build_windows import CLASS_NAMES
from sequence_model.tests.test_train_lstm import write_bundle

FILES = {"results.md", "eval_report.md", "eval_metrics.json", "eval_events.csv", "val_probs.npz", "README.txt",
         "lstm_t_seed1_cpu.pt", "lstm_t_seed2_cpu.pt"}


class Base(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.bundle = self.dir / "bundle"
        self.data = write_bundle(self.bundle)
        patcher = mock.patch.object(sl, "private_strings", return_value=["비밀문자열"])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.train("t", "last", "1", "2")
        self.train("old", "mean", "42")

    def train(self, tag: str, pool: str, *seeds: str) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            tl.main(["--tag", tag, "--bundle", str(self.bundle), "--device", "cpu", "--epochs", "1", "--pool", pool, "--seeds", *seeds,
                     "--checkpoints-dir", str(self.dir / "ck"), "--runs-dir", str(self.dir / "runs")])

    def share(self, *extra: str) -> Path:
        with contextlib.redirect_stdout(io.StringIO()):
            return sl.main(["--tag", "t", "--bundle", str(self.bundle), "--bundle-name", "묶음A", "--bundle-note", "임시 기준이다",
                            "--compare-results", str(self.dir / "runs" / "lstm_old_cpu_results.json"),
                            "--compare-probs", str(self.dir / "runs" / "lstm_old_seed42_cpu_val_probs.npy"), "--compare-name", "old",
                            "--checkpoints-dir", str(self.dir / "ck"), "--runs-dir", str(self.dir / "runs"), *extra])


class BundleTest(Base):
    def test_files_and_zip(self):
        out = self.share()
        self.assertEqual(out.name, "share_lstm_t")
        self.assertEqual({p.name for p in out.iterdir()}, FILES)
        with zipfile.ZipFile(out.with_suffix(".zip")) as archive:
            self.assertEqual({Path(n).name for n in archive.namelist() if not n.endswith("/")}, FILES)
            self.assertTrue(all(n.startswith("share_lstm_t/") for n in archive.namelist()))
            self.assertEqual(archive.read("share_lstm_t/README.txt"), (out / "README.txt").read_bytes())

    def test_probs_file_has_v8_format_and_passes_evaluate(self):
        out = self.share()
        rows = ev.load_rows(self.bundle / "windows.npz", "val")
        with np.load(out / "val_probs.npz", allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"clip", "start_sec", "y", "classes", "probs_s1", "probs_s2"})
            self.assertEqual(data["classes"].tolist(), list(CLASS_NAMES))
            self.assertEqual(data["probs_s1"].dtype, np.float32)
            np.testing.assert_array_equal(data["y"], rows["y"])
            np.testing.assert_array_equal(data["probs_s2"], np.load(self.dir / "runs" / "lstm_t_seed2_cpu_val_probs.npy"))
        self.assertEqual(ev.load_probs(out / "val_probs.npz", "probs_s1", rows).shape, (16, 4))

    def test_metrics_are_from_evaluate_with_run_info_and_comparison(self):
        out = self.share()
        saved = json.loads((out / "eval_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(list(saved["results"]), ["probs_s1", "probs_s2"])
        self.assertEqual(saved["run_info"]["pool"], "last")
        self.assertEqual(saved["run_info"]["seeds"], [1, 2])
        self.assertEqual((saved["comparison"]["pool"], saved["comparison"]["seeds"]), ("mean", [42]))
        # 평균은 비교 행 없이 두 시드로만 낸다
        expected = np.mean([r["argmax"]["macro_f1"] for r in saved["results"].values()])
        self.assertAlmostEqual(saved["summary"]["mean"]["argmax macro F1"], expected)
        events = pd.read_csv(out / "eval_events.csv", encoding="utf-8-sig")
        self.assertEqual(set(events["probs_key"]), {"probs_s1", "probs_s2"})

    def test_reports_show_pool_device_and_seed(self):
        out = self.share()
        evaluation = (out / "eval_report.md").read_text(encoding="utf-8")
        self.assertIn("풀링: last, 장치: cpu, 시드: 1, 2, 입력 묶음: 묶음A", evaluation)
        self.assertIn("| 키 | 풀링 | 장치 | 시드 | argmax macro F1 |", evaluation)
        self.assertIn("| old (비교) | mean | cpu | 42 |", evaluation)
        results = (out / "results.md").read_text(encoding="utf-8")
        self.assertIn("| 시드 | 풀링 | 장치 | 멈춘 epoch | 가장 좋았던 epoch | macro F1 | 정상 | 눈 감김 | 하품 | 고개 숙임(참고) |", results)
        self.assertIn("| 평균 | last | cpu |", results)
        self.assertIn("| 42 (old, 비교) | mean | cpu |", results)

    def test_results_average_by_hand(self):
        out = self.share()
        saved = json.loads((self.dir / "runs" / "lstm_t_cpu_results.json").read_text(encoding="utf-8"))
        expected = np.mean([s["macro_f1"] for s in saved["seeds"]])
        line = next(l for l in (out / "results.md").read_text(encoding="utf-8").splitlines() if l.startswith("| 평균 |"))
        self.assertEqual(line.split("|")[6].strip(), f"{expected:.3f}")

    def test_readme_has_inputs_and_limits(self):
        out = self.share()
        text = (out / "README.txt").read_text(encoding="utf-8")
        for needle in ("풀링 방식: last", "장치: cpu", "시드: 1 2", "입력 묶음: 묶음A", "데이터 출처: nitymed", sl._sha256(self.bundle / "windows.npz"),
                       sl._sha256(self.bundle / "norm.npz"), "--pool last --seeds 1 2 --device cpu", "<번들 폴더>", "predictor.py", "임시 기준이다"):
            self.assertIn(needle, text)
        self.assertNotIn(str(self.bundle), text)

    def test_reference_sentence(self):
        self.train("ref", "mean", "1", "2")
        out = self.share("--reference-results", str(self.dir / "runs" / "lstm_ref_cpu_results.json"), "--reference-name", "ref")
        self.assertIn("mean 풀링 시드 2개(ref)", (out / "results.md").read_text(encoding="utf-8"))


class CompareNoteTest(unittest.TestCase):
    def test_says_alarm_never_rang_below_threshold(self):
        note = sl.compare_note("old", 0.55)
        self.assertIn("0.550", note)
        self.assertIn("한 번도 켜지지 않았", note)

    def test_stays_short_at_threshold(self):
        self.assertNotIn("한 번도", sl.compare_note("old", 0.6))


class SafetyTest(Base):
    def test_does_not_overwrite(self):
        out = self.share()
        before = (out / "results.md").read_bytes()
        with self.assertRaises(SystemExit) as caught:
            self.share()
        self.assertIn("이미 있습니다", str(caught.exception))
        self.assertEqual((out / "results.md").read_bytes(), before)

    def test_private_string_stops_and_leaves_nothing(self):
        with mock.patch.object(sl, "private_strings", return_value=["묶음A"]):
            with self.assertRaises(SystemExit) as caught:
                self.share()
        self.assertIn("README.txt", str(caught.exception))
        self.assertEqual([p.name for p in (self.dir / "runs").iterdir() if "share_lstm" in p.name], [])

    def test_find_private_sees_escaped_text(self):
        folder = self.dir / "scan"
        folder.mkdir()
        (folder / "a.json").write_text(json.dumps({"path": "비밀"}), encoding="utf-8")     # ensure_ascii로 \uXXXX가 된다
        (folder / "b.txt").write_text("괜찮음", encoding="utf-8")
        self.assertEqual(sl.find_private(folder, ["비밀"]), ["a.json: 비밀"])

    def test_probs_that_disagree_with_results_stop(self):
        path = self.dir / "runs" / "lstm_t_cpu_results.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["seeds"][0]["macro_f1"] = 0.123
        path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            self.share()
        self.assertIn("다릅니다", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

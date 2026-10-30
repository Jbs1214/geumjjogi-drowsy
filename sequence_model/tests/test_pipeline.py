# 합성 라벨 파일 생성 → 학습 → 저장 → 로드 → 비교표까지 전체 흐름을 한 번 돌린다.

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval import compare
from sequence_model import config, train
from sequence_model.labels import clip_group
from sequence_model.predictor import SeqPredictor
from sequence_model.tests.synthetic import write_clips
from sequence_model.windows import build_dataset, window_frames


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        root = Path(cls.dir.name)
        cls.labels, cls.reviewed, cls.out = root / "auto", root / "reviewed", root / "out"
        cls.clip_ids = write_clips(cls.labels, cls.reviewed)
        cls.calibration = root / "calibration.txt"
        cls.calibration.write_text("\n".join(cls.clip_ids[:6]), encoding="utf-8")
        cls.common = ["--labels-dir", str(cls.labels), "--calibration-clips", str(cls.calibration)]
        cls.weights = {
            name: train.main(["--model", name, *cls.common, "--epochs", "12", "--device", "cpu",
                              "--out-dir", str(cls.out)])
            for name in ("cnn1d", "lstm")
        }

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_compare(self, *extra: str):
        weights = [str(p) for p in self.weights.values()]
        return compare.main([*self.common, "--reviewed-dir", str(self.reviewed), "--weights", *weights,
                             "--out-dir", str(self.out), *extra])

    def test_weights_get_new_versions(self):
        self.assertEqual({p.name for p in self.weights.values()}, {"best_model_v01.pt", "best_model_v02.pt"})

    def test_predictor_gives_probabilities(self):
        predictor = SeqPredictor(self.weights["cnn1d"])
        window = np.zeros((window_frames(predictor.window_sec), 11), dtype=np.float32)
        proba = predictor.predict_proba(window)
        self.assertEqual(proba.shape, (config.N_CLASSES,))
        self.assertAlmostEqual(float(proba.sum()), 1.0, places=5)

    def test_predictor_rejects_wrong_window(self):
        with self.assertRaises(ValueError):
            SeqPredictor(self.weights["cnn1d"]).predict_proba(np.zeros((10, 11), dtype=np.float32))

    def test_calibration_group_stays_out_of_test(self):
        info = build_dataset(self.labels, tuple(self.clip_ids[:6])).info
        self.assertEqual(set(info[info["group"] == clip_group(self.clip_ids[0])]["split"]), {"train"})

    def test_compare_table(self):
        df = self.run_compare().set_index("method")
        self.assertEqual(set(df.index), {"규칙", "logreg", "forest", "best_model_v01", "best_model_v02"})
        # 자동 라벨과 검수 라벨이 같은 합성 데이터라 규칙은 만점이어야 한다
        self.assertAlmostEqual(df.loc["규칙", "macro_f1"], 1.0)
        self.assertAlmostEqual(df.loc["규칙", "eye_segment_f1"], 1.0)
        self.assertAlmostEqual(df.loc["규칙", "auc"], 1.0)
        # 패턴이 뚜렷해 모델도 충분히 배워야 한다. 파이프라인이 끊기면 여기서 드러난다
        for name in ("best_model_v01", "best_model_v02"):
            self.assertGreater(df.loc[name, "macro_f1"], 0.8)
            self.assertGreater(df.loc[name, "eye_segment_f1"], 0.7)
            self.assertGreater(df.loc[name, "auc"], 0.9)
        self.assertTrue((self.out / "compare.md").exists())
        self.assertTrue((self.out / "compare.csv").exists())

    def test_fps_comparison_drops_rule(self):
        df = self.run_compare("--fps", "15", "--no-baseline")
        self.assertEqual(set(df["method"]), {"best_model_v01", "best_model_v02"})
        self.assertTrue((self.out / "compare_15fps.md").exists())

    def test_conditions_add_rows(self):
        clips = compare.load_eval_clips(self.labels, self.reviewed, tuple(self.clip_ids[:6]))
        table = self.out / "conditions.csv"
        table.parent.mkdir(exist_ok=True)
        table.write_text("clip_id,glasses\n" + "\n".join(f"{c.clip_id},yes" for c in clips[:2]), encoding="utf-8")
        df = self.run_compare("--conditions", str(table), "--no-baseline")
        self.assertEqual(set(df["condition"]), {"전체", "glasses=yes"})

    def test_no_review_labels_stops_with_message(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit) as caught:
                compare.main([*self.common, "--reviewed-dir", empty, "--no-baseline"])
            self.assertIn("평가 클립이 없습니다", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

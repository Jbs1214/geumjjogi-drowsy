# 규칙 기반과 학습 모델을 같은 평가 클립으로 평가해 비교표를 runs/에 저장한다 (기획서 9절).
# 정답은 검수 라벨이고, 평가 클립은 test 분할 중 검수 라벨이 있는 클립이다. 보정에 쓴 클립의 그룹은 test에 들어가지 않는다.
# 규칙 행은 자동 라벨 y를 쓴다. y가 곧 규칙(best_params.json)의 프레임별 결과이기 때문이다.
#
# 실행: python -m eval.compare --reviewed-dir <검수 라벨 폴더> --calibration-clips <목록.txt> --weights checkpoints/best_model_v01.pt

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import classification_metrics, match_segments, prf, score_auc, segments_from_labels
from sequence_model import config
from sequence_model.labels import clip_id_of, load_auto_label, load_reviewed_segments, segments_to_frames, split_clips
from sequence_model.predictor import SeqPredictor
from sequence_model.train import read_clip_list
from sequence_model.windows import build_dataset, fill_missing, fit_normalizer, make_windows, normalize, resample, window_frames

TARGET_CLASSES = {"eye": 1, "yawn": 2}


@dataclass
class EvalClip:
    clip_id: str
    df: pd.DataFrame    # 열: t, 특징 11개, state(검수 라벨), rule(자동 라벨)


def simulate_fps(df: pd.DataFrame, input_fps: float, fps: float = config.FPS) -> pd.DataFrame:
    """특징을 input_fps로 들어온 것처럼 만든다. 행 수는 그대로 두고 프레임 사이는 직전 값을 유지한다 (기획서 9절 FPS별 비교)."""
    kept = np.floor(np.floor(np.arange(len(df)) * input_fps / fps) * fps / input_fps).astype(int)
    out = df.copy()
    columns = list(config.FEATURE_COLUMNS)
    out[columns] = df[columns].to_numpy()[kept]
    return out


def load_eval_clips(labels_dir: Path, reviewed_dir: Path, calibration_clips: tuple[str, ...],
                    input_fps: float | None = None) -> list[EvalClip]:
    """test 분할 중 검수 라벨이 있는 클립을 읽는다."""
    paths = sorted(labels_dir.glob("*.json"))
    splits = split_clips([clip_id_of(p) for p in paths], calibration_clips)
    reviewed = {clip_id_of(p): p for p in reviewed_dir.rglob("*.json")}

    clips = []
    for path in paths:
        clip_id = clip_id_of(path)
        if splits[clip_id] != "test" or clip_id not in reviewed:
            continue
        clip = load_auto_label(path)
        df = clip.to_frame()
        df["rule"] = clip.y
        df["state"] = segments_to_frames(load_reviewed_segments(reviewed[clip_id]), len(df), clip.fps)
        df = fill_missing(resample(df))
        clips.append(EvalClip(clip_id, simulate_fps(df, input_fps) if input_fps else df))
    return clips


def _evaluate(clips: list[EvalClip], window_sec: float, predict) -> dict:
    """클립을 돌며 창 단위·구간 단위 지표를 모은다. predict(clip, 창, 창 시작 초) → (예측, 졸음 점수, 클래스별 예측 구간)."""
    ys, preds, scores = [], [], []
    counts = {name: np.zeros(3, dtype=int) for name in TARGET_CLASSES}
    for clip in clips:
        X, y, starts = make_windows(clip.df, window_sec)
        keep = y != config.UNCERTAIN
        if not keep.any():
            continue
        pred, score, pred_segments = predict(clip, X[keep], starts[keep])
        ys.append(y[keep])
        preds.append(pred)
        scores.append(score)
        # 정답 구간은 창 라벨이 아니라 프레임 라벨에서 구한다. 창 라벨은 짧은 눈 감김을 지우기 때문이다
        true_segments = segments_from_labels(clip.df["t"], clip.df["state"], 1 / config.FPS, tuple(TARGET_CLASSES.values()))
        for name, c in TARGET_CLASSES.items():
            counts[name] += match_segments(true_segments[c], pred_segments[c])

    if not ys:
        raise ValueError("평가할 창이 없습니다: 평가 클립이 모두 창보다 짧거나 불확실입니다")
    y, pred, score = np.concatenate(ys), np.concatenate(preds), np.concatenate(scores)
    window = classification_metrics(y, pred)
    row = {"window_sec": window_sec, "n_windows": len(y), "accuracy": window["accuracy"], "macro_f1": window["macro_f1"]}
    for name in TARGET_CLASSES:
        segment = prf(*counts[name])
        row[f"{name}_segment_f1"] = segment["f1"]
        row[f"{name}_segment_recall"] = segment["recall"]
    row["auc"] = score_auc(y != 0, score)
    return row


def evaluate_rule(clips: list[EvalClip], window_sec: float) -> dict:
    """자동 라벨 y를 규칙의 예측으로 평가한다. 구간은 프레임 해상도 그대로, 졸음 점수는 창 안 이상 프레임 비율."""
    size = window_frames(window_sec)

    def predict(clip: EvalClip, X: np.ndarray, starts: np.ndarray):
        states = clip.df["rule"].to_numpy(dtype=int)
        _, rule_labels, rule_starts = make_windows(clip.df, window_sec, label_col="rule")
        pred = rule_labels[np.searchsorted(rule_starts, starts)]
        pred = np.where(pred == config.UNCERTAIN, 0, pred)      # 규칙이 판단을 못 한 창은 검출하지 않은 것으로 본다
        anomaly = np.isin(states, list(TARGET_CLASSES.values()))
        score = np.array([anomaly[s:s + size].mean() for s in np.round(starts * config.FPS).astype(int)])
        segments = segments_from_labels(clip.df["t"], states, 1 / config.FPS, tuple(TARGET_CLASSES.values()))
        return pred, score, segments

    return {"method": "규칙", **_evaluate(clips, window_sec, predict), "ms_per_window": float("nan")}


def evaluate_model(name: str, predict_proba, clips: list[EvalClip], window_sec: float) -> dict:
    """창 → 클래스 확률 함수를 평가한다. 졸음 점수는 1 − 정상 확률, 구간은 0.5초 칸(창 가운데)으로 잇는다."""
    elapsed = 0.0
    n_windows = 0

    def predict(clip: EvalClip, X: np.ndarray, starts: np.ndarray):
        nonlocal elapsed, n_windows
        began = time.perf_counter()
        proba = predict_proba(X)
        elapsed += time.perf_counter() - began
        n_windows += len(X)
        pred = proba.argmax(axis=1)
        cells = starts + window_sec / 2 - config.STRIDE_SEC / 2
        return pred, 1 - proba[:, 0], segments_from_labels(cells, pred, config.STRIDE_SEC, tuple(TARGET_CLASSES.values()))

    row = _evaluate(clips, window_sec, predict)
    return {"method": name, **row, "ms_per_window": 1000 * elapsed / n_windows}


def fit_baselines(labels_dir: Path, calibration_clips: tuple[str, ...], window_sec: float) -> dict:
    """기준선을 train 분할로 학습한다 → 이름별 창 → 클래스 확률 함수."""
    from sequence_model import baseline     # scikit-learn이 없어도 --no-baseline으로 비교표를 낼 수 있게 여기서 읽는다

    X, y = build_dataset(labels_dir, calibration_clips, window_sec).split("train")
    mean, std = fit_normalizer(X)
    X = normalize(X, mean, std)
    models = {name: baseline.fit_baseline(name, X, y) for name in baseline.BASELINE_NAMES}
    return {name: (lambda w, m=model: baseline.predict_proba(m, normalize(w, mean, std))) for name, model in models.items()}


def to_markdown(df: pd.DataFrame) -> str:
    """비교표 → 마크다운 표 문자열. 소수는 셋째 자리까지, NaN은 '-'."""
    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "-" if np.isnan(value) else f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def compare(clips: list[EvalClip], methods: dict, window_sec: float, include_rule: bool) -> list[dict]:
    """클립 묶음 하나로 모든 방법을 평가한다. methods: 이름 → (창 → 클래스 확률 함수, 창 길이)."""
    rows = [evaluate_rule(clips, window_sec)] if include_rule else []
    rows += [evaluate_model(name, predict, clips, sec) for name, (predict, sec) in methods.items()]
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="규칙 대 학습 모델 비교표")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/processed/labels_auto"), help="자동 라벨 JSON 폴더")
    parser.add_argument("--reviewed-dir", type=Path, required=True, help="검수 라벨 JSON 폴더")
    parser.add_argument("--calibration-clips", type=Path, help="보정에 쓴 클립 번호 목록(한 줄에 하나)")
    parser.add_argument("--weights", type=Path, nargs="*", default=[], help="학습한 가중치 파일")
    parser.add_argument("--window-sec", type=float, default=config.WINDOW_SEC, help="규칙·기준선의 창 길이")
    parser.add_argument("--no-baseline", action="store_true", help="로지스틱 회귀·랜덤 포레스트를 뺀다")
    parser.add_argument("--fps", type=float, help="입력 FPS 흉내 (예: 15). 규칙은 이미 낸 결과라 표에서 뺀다")
    parser.add_argument("--conditions", type=Path, help="clip_id 열과 조건 열(조명·안경 등)이 있는 CSV. 조건별 행을 더한다")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    calibration_clips = read_clip_list(args.calibration_clips)
    if not calibration_clips:
        print("경고: --calibration-clips가 없어 보정에 쓴 클립이 평가에 섞일 수 있습니다")

    clips = load_eval_clips(args.labels_dir, args.reviewed_dir, calibration_clips, args.fps)
    if not clips:
        raise SystemExit("평가 클립이 없습니다: test 분할에 검수 라벨이 있는 클립이 필요합니다 (보정에 쓴 클립은 평가에 쓸 수 없습니다)")

    methods = {}
    if not args.no_baseline:
        methods = {name: (predict, args.window_sec)
                   for name, predict in fit_baselines(args.labels_dir, calibration_clips, args.window_sec).items()}
    for path in args.weights:
        predictor = SeqPredictor(path)
        methods[path.stem] = (predictor.predict_batch, predictor.window_sec)

    include_rule = args.fps is None
    rows = [{"condition": "전체", **row} for row in compare(clips, methods, args.window_sec, include_rule)]
    if args.conditions:
        table = pd.read_csv(args.conditions, dtype=str).set_index("clip_id")
        for column in table.columns:
            for value, ids in table.groupby(column).groups.items():
                subset = [clip for clip in clips if clip.clip_id in ids]
                if subset:
                    rows += [{"condition": f"{column}={value}", **row}
                             for row in compare(subset, methods, args.window_sec, include_rule)]

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.fps:g}fps" if args.fps else ""
    df.to_csv(args.out_dir / f"compare{suffix}.csv", index=False, encoding="utf-8-sig")
    text = to_markdown(df)
    (args.out_dir / f"compare{suffix}.md").write_text(text + "\n", encoding="utf-8")
    print(f"평가 클립 {len(clips)}개\n{text}")
    return df


if __name__ == "__main__":
    main()

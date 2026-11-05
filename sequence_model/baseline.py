# 창 요약 특징 + 로지스틱 회귀·랜덤 포레스트 기준선. 1D-CNN·LSTM이 이보다 나은지 비교한다 (기술서 4.2).
# train 창으로 학습하고 val 창의 4클래스 지표를 출처별로 낸다. Test 창은 읽지 않는다.
#
# 실행: python -m sequence_model.baseline --tag base1 --windows data/windows_sig3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from eval.metrics import classification_metrics
from sequence_model.train import load_windows

BASELINE_NAMES = ("logreg", "forest")


def summarize(X: np.ndarray) -> np.ndarray:
    """창 (N, 프레임, 특징 수) → 특징별 평균·표준편차·최솟값·최댓값 (N, 특징 수 × 4)."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1)], axis=1)


def fit_baseline(name: str, X: np.ndarray, y: np.ndarray, seed: int = config.SEEDS[0]):
    """창과 정답으로 기준선 모델을 학습한다. 클래스 불균형은 class_weight로 맞춘다."""
    if name == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    elif name == "forest":
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"기준선 이름은 {BASELINE_NAMES} 중 하나여야 합니다: {name}")
    return model.fit(summarize(X), y)


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """창 (N, 프레임, 특징 수) → 클래스 확률 (N, N_CLASSES). train에 없던 클래스의 확률은 0이다."""
    proba = np.zeros((len(X), config.N_CLASSES))
    proba[:, model.classes_] = model.predict_proba(summarize(X))
    return proba


def main() -> None:
    parser = argparse.ArgumentParser(description="창 요약 특징 + scikit-learn 기준선을 학습하고 val 창으로 채점한다")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--windows", type=Path, default=config.WINDOW_DIR, help="창 폴더 (build_windows --out)")
    args = parser.parse_args()
    out_dir = config.RUNS / args.tag
    if (out_dir / "baseline.csv").exists():
        raise SystemExit(f"{out_dir}에 결과가 있습니다. --tag에 새 이름을 쓰세요")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, info, _ = load_windows(args.windows)
    train = (info["split"] == "train").to_numpy()
    val = (info["split"] == "val").to_numpy()
    rows = []
    for name in BASELINE_NAMES:
        model = fit_baseline(name, X[train], y[train])
        pred = predict_proba(model, X[val]).argmax(axis=1)
        sources = info.loc[val, "source"].to_numpy()
        for source in ("전체", *np.unique(sources)):
            keep = np.ones(len(pred), dtype=bool) if source == "전체" else sources == source
            metrics = classification_metrics(y[val][keep], pred[keep])
            rows.append({"모델": name, "출처": source, "창": int(keep.sum()), "macro_F1": metrics["macro_f1"]})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "baseline.csv", index=False, encoding="utf-8-sig")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

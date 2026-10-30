# 창 요약 특징 + 로지스틱 회귀·랜덤 포레스트 기준선. 딥러닝이 이보다 나은지 비교한다 (기획서 5-2).
# 저장하지 않고 eval/compare.py가 train 분할로 그때그때 학습한다.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sequence_model import config

BASELINE_NAMES = ("logreg", "forest")


def summarize(X: np.ndarray) -> np.ndarray:
    """창 (N, 프레임, 특징 수) → 특징별 평균·표준편차·최솟값·최댓값 (N, 특징 수 × 4)."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1)], axis=1)


def fit_baseline(name: str, X: np.ndarray, y: np.ndarray, seed: int = config.SPLIT_SEED):
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

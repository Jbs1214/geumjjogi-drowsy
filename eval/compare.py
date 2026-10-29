# 같은 평가 창으로 규칙 기반과 학습 모델을 평가해 비교표를 runs/에 저장한다.
# 실행: python -m eval.compare

import numpy as np
import pandas as pd


def evaluate_rule(windows: np.ndarray, info: pd.DataFrame) -> dict:
    """규칙 기반 졸음 점수·판정(rule_based/)으로 평가 창의 지표를 계산한다."""
    raise NotImplementedError


def evaluate_model(weight_path: str, windows: np.ndarray, masks: np.ndarray, info: pd.DataFrame) -> dict:
    """학습한 분류기(sequence_model.predictor)로 평가 창의 지표를 계산한다."""
    raise NotImplementedError


def to_markdown(df: pd.DataFrame) -> str:
    """비교표 → 마크다운 표 문자열."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

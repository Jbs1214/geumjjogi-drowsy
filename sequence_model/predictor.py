# 학습한 가중치를 불러와 10초 창 하나의 클래스 확률을 낸다.
# 실시간 데모는 1초마다 최근 10초 창을 predict_proba()에 넣는다.

from pathlib import Path

import numpy as np


class SeqPredictor:
    def __init__(self, path: Path) -> None:
        """가중치 파일(path)과 함께 저장한 설정(모델 이름·특징 열·임계치)을 불러온다."""
        raise NotImplementedError

    def predict_proba(self, window: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """창 (WINDOW_FRAMES, 특징 수) → 클래스 확률 (N_CLASSES,). 졸음 확률은 졸음 점수처럼 비교에 쓴다."""
        raise NotImplementedError

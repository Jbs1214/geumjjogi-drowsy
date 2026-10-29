# 기획서 9절 평가 지표 계산. 규칙 기반과 학습 모델에 똑같이 쓴다.

import numpy as np


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """창 단위 정답·예측 → 정확도, macro F1, 졸음 클래스 정밀도·재현율·F1."""
    raise NotImplementedError


def score_auc(y: np.ndarray, score: np.ndarray) -> float:
    """정상 대 졸음 창의 졸음 점수 ROC-AUC. 분류기는 졸음 확률을 점수로 쓴다."""
    raise NotImplementedError


def false_alarms_per_hour(alerts: np.ndarray, hours: float) -> float:
    """정상·다른 이상행동 영상에서 시간당 잘못 울린 경고 횟수."""
    raise NotImplementedError


def event_recall(events: list[tuple[float, float]], alerts: np.ndarray) -> float:
    """눈 감김 2초 이벤트(시작·끝 초) 중 경고가 울린 비율."""
    raise NotImplementedError


def alert_delay(events: list[tuple[float, float]], alerts: np.ndarray) -> float:
    """눈 감김 시작부터 경고까지 걸린 평균 시간(초)."""
    raise NotImplementedError


def fps(frame_times: np.ndarray) -> float:
    """프레임 처리 시각(초) → 실시간 처리 속도."""
    raise NotImplementedError

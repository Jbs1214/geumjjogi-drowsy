# 기획서 9절 평가 지표 계산. 규칙 기반과 학습 모델에 똑같이 쓴다.
# 라벨에 ±0.5초 오차가 있어 프레임이 아니라 구간(눈 감김 구간을 찾았는가)을 기본 단위로 본다.

import numpy as np
import pandas as pd

from rule_based.common import runs

LABEL_TOLERANCE_SEC = 0.5   # 구간 라벨 오차 (기획서 3-5)
ALERT_TOLERANCE_SEC = 0.2   # Euro NCAP 시험의 경고 허용 오차 (기획서 9절)


def prf(tp: int, fp: int, fn: int) -> dict:
    """맞힌 수·잘못 낸 수·놓친 수 → 정밀도, 재현율, F1. 분모가 0이면 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """창 단위 정답·예측 → 정확도, macro F1, 클래스별 F1. macro는 정답이나 예측에 나온 클래스로만 평균낸다."""
    y, pred = np.asarray(y), np.asarray(pred)
    per_class = {}
    for c in np.union1d(y, pred):
        hit = int(np.count_nonzero((y == c) & (pred == c)))
        per_class[int(c)] = prf(hit, int(np.count_nonzero(pred == c)) - hit, int(np.count_nonzero(y == c)) - hit)["f1"]
    return {
        "accuracy": float(np.mean(y == pred)) if len(y) else 0.0,
        "macro_f1": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "f1_per_class": per_class,
    }


def segments_from_labels(t: np.ndarray, labels: np.ndarray, step: float, classes: tuple[int, ...]) -> dict:
    """칸 시작 시각·칸별 상태 → 클래스별 구간 목록 [(시작 초, 끝 초), …]. 칸 길이는 step초."""
    t, labels = np.asarray(t, dtype=float), np.asarray(labels)
    return {c: [(float(t[s]), float(t[e] + step)) for s, e in zip(*runs(labels == c))] for c in classes}


def match_segments(true: list[tuple[float, float]], pred: list[tuple[float, float]],
                   tolerance: float = LABEL_TOLERANCE_SEC) -> tuple[int, int, int]:
    """정답·예측 구간을 1:1로 짝지어 (맞힌 수, 잘못 낸 수, 놓친 수)를 센다. tolerance초 이내로 닿으면 찾은 것으로 본다."""
    unused = sorted(pred)
    tp = 0
    for start, end in sorted(true):
        found = next((p for p in unused if p[0] - tolerance < end and p[1] + tolerance > start), None)
        if found is not None:
            unused.remove(found)
            tp += 1
    return tp, len(unused), len(true) - tp


def score_auc(y: np.ndarray, score: np.ndarray) -> float:
    """정상(0) 대 이상(1)의 졸음 점수 ROC-AUC. 순위합(Mann-Whitney)으로 계산하고, 한쪽이 없으면 NaN."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(score, dtype=float).rank().to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def false_alarms_per_hour(alerts: np.ndarray, hours: float) -> float:
    """정상 운전 영상에서 시간당 잘못 울린 경고 횟수. alerts는 경고 시각 목록."""
    if hours <= 0:
        raise ValueError(f"영상 길이는 0보다 커야 합니다: {hours}")
    return len(alerts) / hours


def _first_alert(event: tuple[float, float], alerts: np.ndarray, tolerance: float) -> float | None:
    """구간 시작부터 끝 + tolerance 사이의 첫 경고 시각. 없으면 None."""
    hits = [a for a in sorted(alerts) if event[0] <= a <= event[1] + tolerance]
    return hits[0] if hits else None


def event_recall(events: list[tuple[float, float]], alerts: np.ndarray, tolerance: float = ALERT_TOLERANCE_SEC) -> float:
    """눈 감김 이벤트(시작·끝 초) 중 경고가 울린 비율. 이벤트가 없으면 NaN."""
    if not events:
        return float("nan")
    return sum(_first_alert(e, alerts, tolerance) is not None for e in events) / len(events)


def alert_delay(events: list[tuple[float, float]], alerts: np.ndarray, threshold_sec: float,
                tolerance: float = ALERT_TOLERANCE_SEC) -> float:
    """눈 감김이 기준 시간(1.5초·3초)에 닿은 뒤 경고까지 걸린 평균 시간(초). 기준보다 짧은 이벤트와 놓친 이벤트는 뺀다."""
    delays = []
    for event in events:
        alert = _first_alert(event, alerts, tolerance)
        if event[1] - event[0] >= threshold_sec and alert is not None:
            delays.append(alert - (event[0] + threshold_sec))
    return float(np.mean(delays)) if delays else float("nan")


def fps(frame_times: np.ndarray) -> float:
    """프레임 처리 시각(초) → 실시간 처리 속도. 프레임이 2개 미만이면 NaN."""
    if len(frame_times) < 2 or frame_times[-1] <= frame_times[0]:
        return float("nan")
    return (len(frame_times) - 1) / float(frame_times[-1] - frame_times[0])

# rule_based 판정들이 같이 쓰는 계산: 임계값 판정, 연속 구간.

import numpy as np


def over(x: np.ndarray, threshold: float) -> np.ndarray:
    """프레임별 값 > threshold (bool). NaN(얼굴 미검출)과의 비교는 False라 판정하지 않은 것으로 남는다."""
    return np.asarray(x, dtype=float) > threshold


def under(x: np.ndarray, threshold: float) -> np.ndarray:
    """프레임별 값 < threshold (bool). NaN은 False."""
    return np.asarray(x, dtype=float) < threshold


def runs(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True가 이어진 구간마다 (첫 프레임 번호 배열, 마지막 프레임 번호 배열)."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flags, dtype=int), 0]))
    return edges[::2], edges[1::2] - 1

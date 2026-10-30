# rule_based 지표들이 같이 쓰는 계산: 임계값 판정, 연속 구간, 최근 config.WINDOW_SEC초 창.

import math

import numpy as np

from rule_based import config


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


def window_start(t: np.ndarray) -> np.ndarray:
    """프레임마다 (t − WINDOW_SEC, t] 창의 첫 프레임 번호."""
    return np.searchsorted(t, t - config.WINDOW_SEC, side="right")


def window_full(t: np.ndarray) -> np.ndarray:
    """첫 프레임부터 WINDOW_SEC가 지났는지. 지나기 전 값은 NaN으로 둔다 (창에 프레임이 적으면 깜빡임 한 번에도 값이 튄다)."""
    return t - t[0] >= config.WINDOW_SEC


def window_ratio(t: np.ndarray, flags: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """프레임마다 창 안에서 valid 프레임 중 flags인 비율. valid 프레임이 없거나 창이 덜 찼으면 NaN."""
    hit = np.r_[0, np.cumsum(flags)]
    n = np.r_[0, np.cumsum(valid)]
    left = window_start(t)
    right = np.arange(1, len(t) + 1)
    n_valid = n[right] - n[left]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(window_full(t) & (n_valid > 0), (hit[right] - hit[left]) / n_valid, np.nan)


def every_second(t: np.ndarray) -> list[int]:
    """CLI 출력용: 1초, 2초, …마다 그 시각 직전 프레임 번호."""
    return [int(np.searchsorted(t, sec, side="right")) - 1 for sec in range(1, math.ceil(t[-1]) + 1)]

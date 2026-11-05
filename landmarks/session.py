# 세션(영상) 기준 정규화 (README "최종 기획서 반영과 v7"). 사람·카메라마다 다른 평소 눈 크기와 고개 각도를 맞춘다.
# 영상 시작부터 BASE_SEC초 안의 검출 프레임으로 지금까지의 평소 값을 구한다(미래 프레임은 쓰지 않음).
#   EAR 3개: 평소 EAR(90% 분위) 대비 비율 / MAR: 평소 MAR(중앙값)과의 차이 / pitch·yaw·roll: 평소 각도(중앙값)와의 차이
# blink·jaw_open·face는 그대로다. 파일 평가(build_windows.feature_matrix)와 실시간(demo.webcam)이 같은 코드를 쓴다.

import bisect

import numpy as np

import config

BASE_SEC = 60.0
EAR_QUANTILE = 0.9
RATIO = ("ear_l", "ear_r", "ear")
CENTER = ("mar", "pitch", "yaw", "roll")
FACE = config.FEATURES.index(config.FACE_COLUMN)


def _quantile(sorted_values: list[float], q: float) -> float:
    """정렬된 값의 분위수 (np.quantile 기본 선형 보간과 같다)."""
    pos = (len(sorted_values) - 1) * q
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


class SessionBaseline:
    """특징 벡터(11개, feature_matrix 한 행)를 하나씩 받아 세션 기준으로 바꾼다."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self._index = {name: config.FEATURES.index(name) for name in RATIO + CENTER}
        self._values: dict[str, list[float]] = {name: [] for name in self._index}

    def transform(self, t: float, vector: np.ndarray) -> np.ndarray:
        if self._t0 is None:
            self._t0 = t
        out = np.array(vector, dtype=np.float32)
        if out[FACE] <= 0.5:
            return out                       # 미검출 프레임은 0 그대로
        if t - self._t0 <= BASE_SEC + 1e-9:
            for name, i in self._index.items():
                bisect.insort(self._values[name], float(out[i]))
        for name, i in self._index.items():
            values = self._values[name]
            if name in RATIO:
                base = _quantile(values, EAR_QUANTILE)
                out[i] = out[i] / base if base > 1e-6 else 1.0
            else:
                out[i] = out[i] - _quantile(values, 0.5)
        return out


def relative_matrix(t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """프레임별 특징 (N, 11)과 시각 → 세션 기준으로 바꾼 (N, 11)."""
    baseline = SessionBaseline()
    return np.stack([baseline.transform(float(ti), row) for ti, row in zip(t, X)]) if len(X) else X

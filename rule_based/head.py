# 고개 숙임 판정(HeadDown). 기술서 3.3.
# pitch는 숙이면 +, 들면 − (landmarks/README.md).
#
# 실행: python -m rule_based.head <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config


class HeadDown:
    """프레임을 차례로 받아 고개 숙임 여부를 낸다 (기술서 3.3). 지금까지의 프레임만 쓰므로 파일과 실시간이 같은 값을 낸다.
    기준은 정면 pitch + PITCH_DOWN_RELATIVE이고, 정면 pitch가 아직 없으면 PITCH_DOWN_ABSOLUTE다.
    정면 pitch는 세션 시작 FRONT_PITCH_SEC초 안에서 두 눈의 EAR이 모두 EAR_CLOSED 이상인 프레임의 pitch 중앙값이다."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self._front: list[float] = []
        self.limit = config.PITCH_DOWN_ABSOLUTE

    def update(self, t: float, pitch: float, ear_l: float, ear_r: float) -> bool:
        """프레임 시각, pitch(도), 좌우 EAR → 고개 숙임 여부. 얼굴 미검출(NaN)은 False."""
        if self._t0 is None:
            self._t0 = t
        if np.isnan(pitch):
            return False
        down = pitch >= self.limit
        if ear_l >= config.EAR_CLOSED and ear_r >= config.EAR_CLOSED and t - self._t0 <= config.FRONT_PITCH_SEC:
            self._front.append(pitch)
            self.limit = float(np.median(self._front)) + config.PITCH_DOWN_RELATIVE
        return down


def is_down(t: np.ndarray, pitch: np.ndarray, ear_l: np.ndarray, ear_r: np.ndarray) -> np.ndarray:
    """프레임별 시각, pitch(도), 좌우 EAR → 고개 숙임 여부(bool). HeadDown과 같은 값이다."""
    head = HeadDown()
    return np.array([head.update(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(t, pitch, ear_l, ear_r)], dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="랜드마크 표(CSV)에서 고개 숙임 프레임 수를 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t, pitch, ear_l, ear_r = (df[name].to_numpy() for name in ("t", "pitch", "ear_l", "ear_r"))
    down = is_down(t, pitch, ear_l, ear_r)
    print(f"{args.csv}: 고개 숙임 {down.sum()}프레임 ({down.mean():.1%}), 기준 정면 pitch + {config.PITCH_DOWN_RELATIVE}")


if __name__ == "__main__":
    main()

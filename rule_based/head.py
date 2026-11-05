# 고개 숙임 비율: 최근 config.WINDOW_SEC초 동안 고개를 숙인(HeadDown) 프레임의 비율. 졸음 점수의 지표 하나.
# pitch는 숙이면 +, 들면 − (landmarks/README.md). 창 자르기와 분모는 perclos.py와 같다.
#
# 실행: python -m rule_based.head <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.common import every_second, window_ratio


class HeadDown:
    """프레임을 차례로 받아 고개 숙임 여부를 낸다 (기술서 3.3). 지금까지의 프레임만 쓰므로 파일과 실시간이 같은 값을 낸다.
    기준은 정면 pitch + PITCH_DOWN_RELATIVE이고, 정면 pitch가 아직 없으면 PITCH_DOWN_ABSOLUTE다.
    정면 pitch는 세션 시작 FRONT_PITCH_SEC초 안에서 EAR이 EAR_CLOSED 이상인 프레임의 pitch 중앙값이다."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self._front: list[float] = []
        self.limit = config.PITCH_DOWN_ABSOLUTE

    def update(self, t: float, pitch: float, ear: float) -> bool:
        """프레임 시각, pitch(도), EAR → 고개 숙임 여부. 얼굴 미검출(NaN)은 False."""
        if self._t0 is None:
            self._t0 = t
        if np.isnan(pitch):
            return False
        down = pitch >= self.limit
        if ear >= config.EAR_CLOSED and t - self._t0 <= config.FRONT_PITCH_SEC:
            self._front.append(pitch)
            self.limit = float(np.median(self._front)) + config.PITCH_DOWN_RELATIVE
        return down


def is_down(t: np.ndarray, pitch: np.ndarray, ear: np.ndarray) -> np.ndarray:
    """프레임별 시각, pitch(도), EAR → 고개 숙임 여부(bool). HeadDown과 같은 값이다."""
    head = HeadDown()
    return np.array([head.update(float(a), float(b), float(c)) for a, b, c in zip(t, pitch, ear)], dtype=bool)


def head_down_ratio(t: np.ndarray, pitch: np.ndarray, ear: np.ndarray) -> np.ndarray:
    """프레임마다 (t − WINDOW_SEC, t] 구간에서 고개 숙인 프레임 비율.

    분모는 얼굴이 잡힌 프레임 수. 잡힌 프레임이 없거나, 첫 프레임부터 WINDOW_SEC가 지나기 전에는 NaN (perclos와 같은 규칙).
    """
    t = np.asarray(t, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    return window_ratio(t, is_down(t, pitch, ear), ~np.isnan(pitch))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"랜드마크 표(CSV)에서 1초마다 최근 {config.WINDOW_SEC}초 고개 숙임 비율을 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t, pitch, ear = df["t"].to_numpy(), df["pitch"].to_numpy(), df["ear"].to_numpy()
    df["head_down"] = head_down_ratio(t, pitch, ear)
    down = is_down(t, pitch, ear)
    print(f"{args.csv}: 고개 숙임 {down.sum()}프레임 ({down.mean():.1%}), 기준 정면 pitch + {config.PITCH_DOWN_RELATIVE}, 최대 비율 {df['head_down'].max():.3f}")

    # 1초마다 그 시각 직전 프레임의 값. NaN은 아직 창 길이만큼 쌓이지 않은 구간
    print(df.iloc[every_second(t)][["t", "pitch", "head_down"]].round(3).to_string(index=False, na_rep="NaN"))


if __name__ == "__main__":
    main()

# 고개 숙임 비율: 최근 config.WINDOW_SEC초 동안 고개를 숙인(pitch > PITCH_DOWN) 프레임의 비율. 졸음 점수의 지표 하나.
# pitch는 숙이면 +, 들면 − (landmarks/README.md). 창 자르기와 분모는 perclos.py와 같다.
#
# 실행: python -m rule_based.head <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rule_based import config
from rule_based.common import every_second, over, window_ratio


def is_down(pitch: np.ndarray) -> np.ndarray:
    """프레임별 pitch(도) → 고개 숙임 여부(bool). 얼굴 미검출(NaN)은 False."""
    return over(pitch, config.PITCH_DOWN)


def head_down_ratio(t: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """프레임마다 (t − WINDOW_SEC, t] 구간에서 고개 숙인 프레임 비율.

    분모는 얼굴이 잡힌 프레임 수. 잡힌 프레임이 없거나, 첫 프레임부터 WINDOW_SEC가 지나기 전에는 NaN (perclos와 같은 규칙).
    """
    t = np.asarray(t, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    return window_ratio(t, is_down(pitch), ~np.isnan(pitch))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"랜드마크 표(CSV)에서 1초마다 최근 {config.WINDOW_SEC}초 고개 숙임 비율을 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    t, pitch = df["t"].to_numpy(), df["pitch"].to_numpy()
    df["head_down"] = head_down_ratio(t, pitch)
    down = is_down(pitch)
    print(f"{args.csv}: 고개 숙임 {down.sum()}프레임 ({down.mean():.1%}), 기준 pitch > {config.PITCH_DOWN}, 최대 비율 {df['head_down'].max():.3f}")

    # 1초마다 그 시각 직전 프레임의 값. NaN은 아직 창 길이만큼 쌓이지 않은 구간
    print(df.iloc[every_second(t)][["t", "pitch", "head_down"]].round(3).to_string(index=False, na_rep="NaN"))


if __name__ == "__main__":
    main()

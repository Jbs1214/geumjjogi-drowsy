# PERCLOS: 최근 config.WINDOW_SEC초 동안 눈을 감은 프레임의 비율 (기획서 3-2).
# 시간은 프레임 수가 아니라 t(초)로 자르므로 15 FPS 표와 웹캠 원본 FPS에서 똑같이 동작한다.
#
# 실행: python -m rule_based.perclos <landmarks.extract CSV>

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.eyes import is_closed


def perclos(t: np.ndarray, ear: np.ndarray) -> np.ndarray:
    """프레임마다 (t − WINDOW_SEC, t] 구간의 PERCLOS.

    분모는 얼굴이 잡힌 프레임 수다 (NaN은 감음도 뜸도 아니므로 뺀다). 잡힌 프레임이 없으면 NaN.
    첫 프레임부터 WINDOW_SEC가 지나기 전에는 NaN (창에 프레임이 적으면 깜빡임 한 번에도 값이 1.0까지 튄다).
    """
    t = np.asarray(t, dtype=float)
    ear = np.asarray(ear, dtype=float)
    closed = np.r_[0, np.cumsum(is_closed(ear))]
    valid = np.r_[0, np.cumsum(~np.isnan(ear))]
    left = np.searchsorted(t, t - config.WINDOW_SEC, side="right")
    right = np.arange(1, len(t) + 1)
    n_valid = valid[right] - valid[left]
    full = t - t[0] >= config.WINDOW_SEC
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(full & (n_valid > 0), (closed[right] - closed[left]) / n_valid, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"랜드마크 표(CSV)에서 1초마다 최근 {config.WINDOW_SEC}초 PERCLOS를 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t = df["t"].to_numpy()
    df["perclos"] = perclos(t, df["ear"].to_numpy())
    print(f"{args.csv}: 창 {config.WINDOW_SEC}초, 기준 ear < {config.EAR_CLOSED}, 최대 PERCLOS {df['perclos'].max():.3f}")

    # 1초마다 그 시각 직전 프레임의 값. NaN은 아직 창 길이만큼 쌓이지 않은 구간
    idx = [int(np.searchsorted(t, sec, side="right")) - 1 for sec in range(1, math.ceil(t[-1]) + 1)]
    print(df.iloc[idx][["t", "perclos"]].round(3).to_string(index=False, na_rep="NaN"))


if __name__ == "__main__":
    main()

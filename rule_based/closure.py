# 가장 긴 눈 감김: 최근 config.WINDOW_SEC초 안에서 가장 길게 이어진 눈 감음의 길이(초). 졸음 점수의 지표 하나.
# 길이는 eyes.closed_runs()와 같은 정의 (첫 감음 프레임 시각 → 마지막 감음 프레임 시각).
#
# 실행: python -m rule_based.closure <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.common import every_second, runs, window_full, window_start
from rule_based.eyes import is_closed


def longest_closure(t: np.ndarray, ear: np.ndarray) -> np.ndarray:
    """프레임마다 (t − WINDOW_SEC, t] 구간에서 가장 긴 연속 감음 길이(초). 감음이 없으면 0.

    창 경계에 걸친 구간은 창 안의 프레임만으로 재고, 아직 감고 있는 구간은 지금 프레임까지 잰다 (alarm.py와 같음).
    첫 프레임부터 WINDOW_SEC가 지나기 전에는 NaN (perclos와 같은 규칙).
    """
    t = np.asarray(t, dtype=float)
    starts, ends = runs(is_closed(ear))
    left = window_start(t)
    out = np.zeros(len(t))
    for i in range(len(t)):
        inside = (starts <= i) & (ends >= left[i])
        if inside.any():
            out[i] = (t[np.minimum(ends[inside], i)] - t[np.maximum(starts[inside], left[i])]).max()
    return np.where(window_full(t), out, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"랜드마크 표(CSV)에서 1초마다 최근 {config.WINDOW_SEC}초 안 가장 긴 눈 감김(초)을 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t = df["t"].to_numpy()
    df["longest_closure"] = longest_closure(t, df["ear"].to_numpy())
    print(f"{args.csv}: 창 {config.WINDOW_SEC}초, 기준 ear < {config.EAR_CLOSED}, 최대 {df['longest_closure'].max():.3f}초")

    # 1초마다 그 시각 직전 프레임의 값. NaN은 아직 창 길이만큼 쌓이지 않은 구간
    print(df.iloc[every_second(t)][["t", "longest_closure"]].round(3).to_string(index=False, na_rep="NaN"))


if __name__ == "__main__":
    main()

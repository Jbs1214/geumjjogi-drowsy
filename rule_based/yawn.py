# 하품 횟수: 최근 config.WINDOW_SEC초 안에 하품을 몇 번 했는지. 졸음 점수의 지표 하나.
# 입 벌림(mar > MAR_YAWN)이 YAWN_MIN_SEC 이상 이어지면 하품 1회로, 그 시간에 닿은 프레임에서 센다 (alarm.py와 같은 방식).
#
# 실행: python -m rule_based.yawn <landmarks.extract CSV>

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.eyes import closed_runs


def is_open(mar: np.ndarray) -> np.ndarray:
    """프레임별 mar → 입 벌림 여부(bool). 얼굴 미검출(NaN)은 False."""
    mar = np.asarray(mar, dtype=float)
    return ~np.isnan(mar) & (np.nan_to_num(mar, nan=-np.inf) > config.MAR_YAWN)


def yawn_times(t: np.ndarray, mar: np.ndarray) -> np.ndarray:
    """하품으로 센 시각(초) 목록. 입 벌림 구간마다 벌린 시간이 YAWN_MIN_SEC에 처음 닿은 프레임 시각."""
    t = np.asarray(t, dtype=float)
    edges = np.flatnonzero(np.diff(np.r_[0, is_open(mar).astype(int), 0]))
    times = []
    for s, e in zip(edges[::2], edges[1::2]):
        reached = np.flatnonzero(t[s:e] - t[s] >= config.YAWN_MIN_SEC)
        if len(reached):
            times.append(t[s + reached[0]])
    return np.array(times)


def yawn_count(t: np.ndarray, mar: np.ndarray) -> np.ndarray:
    """프레임마다 (t − WINDOW_SEC, t] 구간에서 센 하품 횟수. 첫 프레임부터 WINDOW_SEC가 지나기 전에는 NaN (perclos와 같은 규칙)."""
    t = np.asarray(t, dtype=float)
    times = yawn_times(t, mar)
    count = np.searchsorted(times, t, side="right") - np.searchsorted(times, t - config.WINDOW_SEC, side="right")
    full = t - t[0] >= config.WINDOW_SEC
    return np.where(full, count, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"랜드마크 표(CSV)에서 입 벌림 구간과 1초마다 최근 {config.WINDOW_SEC}초 하품 횟수를 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t, mar = df["t"].to_numpy(), df["mar"].to_numpy()
    times = yawn_times(t, mar)
    print(f"{args.csv}: 하품 {len(times)}회, 기준 mar > {config.MAR_YAWN}가 {config.YAWN_MIN_SEC:g}초 이상")

    runs = pd.DataFrame(closed_runs(t, is_open(mar)), columns=["start", "end"])
    runs["length"] = runs["end"] - runs["start"]
    runs["yawn_at"] = [next((y for y in times if s <= y <= e), np.nan) for s, e in zip(runs["start"], runs["end"])]
    print(runs.round(3).to_string(index=False, na_rep="-"))

    df["yawn_count"] = yawn_count(t, mar)
    # 1초마다 그 시각 직전 프레임의 값. NaN은 아직 창 길이만큼 쌓이지 않은 구간
    idx = [int(np.searchsorted(t, sec, side="right")) - 1 for sec in range(1, math.ceil(t[-1]) + 1)]
    print(df.iloc[idx][["t", "yawn_count"]].round(3).to_string(index=False, na_rep="NaN"))


if __name__ == "__main__":
    main()

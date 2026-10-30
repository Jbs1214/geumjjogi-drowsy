# 하품 횟수: 최근 config.WINDOW_SEC초 안에 하품을 몇 번 했는지. 졸음 점수의 지표 하나.
# 입 벌림(mar > MAR_YAWN)이 YAWN_MIN_SEC 이상 이어지면 하품 1회로, 그 시간에 닿은 프레임에서 센다 (alarm.py와 같은 방식).
#
# 실행: python -m rule_based.yawn <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.common import every_second, over, runs, window_full
from rule_based.eyes import closed_runs


def is_open(mar: np.ndarray) -> np.ndarray:
    """프레임별 mar → 입 벌림 여부(bool). 얼굴 미검출(NaN)은 False."""
    return over(mar, config.MAR_YAWN)


def yawn_times(t: np.ndarray, mar: np.ndarray) -> np.ndarray:
    """하품으로 센 시각(초) 목록. 입 벌림 구간마다 벌린 시간이 YAWN_MIN_SEC에 처음 닿은 프레임 시각."""
    t = np.asarray(t, dtype=float)
    times = []
    for s, e in zip(*runs(is_open(mar))):
        reached = np.flatnonzero(t[s:e + 1] - t[s] >= config.YAWN_MIN_SEC)
        if len(reached):
            times.append(t[s + reached[0]])
    return np.array(times)


def yawn_count(t: np.ndarray, mar: np.ndarray) -> np.ndarray:
    """프레임마다 (t − WINDOW_SEC, t] 구간에서 센 하품 횟수. 첫 프레임부터 WINDOW_SEC가 지나기 전에는 NaN (perclos와 같은 규칙)."""
    t = np.asarray(t, dtype=float)
    times = yawn_times(t, mar)
    count = np.searchsorted(times, t, side="right") - np.searchsorted(times, t - config.WINDOW_SEC, side="right")
    return np.where(window_full(t), count, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"랜드마크 표(CSV)에서 입 벌림 구간과 1초마다 최근 {config.WINDOW_SEC}초 하품 횟수를 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t, mar = df["t"].to_numpy(), df["mar"].to_numpy()
    times = yawn_times(t, mar)
    print(f"{args.csv}: 하품 {len(times)}회, 기준 mar > {config.MAR_YAWN}가 {config.YAWN_MIN_SEC:g}초 이상")

    # 구간 찾기는 눈 감음과 같아서 eyes.closed_runs()를 입 벌림 여부에 그대로 쓴다
    runs_df = pd.DataFrame(closed_runs(t, is_open(mar)), columns=["start", "end"])
    runs_df["length"] = runs_df["end"] - runs_df["start"]
    runs_df["yawn_at"] = [next((y for y in times if s <= y <= e), np.nan) for s, e in zip(runs_df["start"], runs_df["end"])]
    print(runs_df.round(3).to_string(index=False, na_rep="-"))

    df["yawn_count"] = yawn_count(t, mar)
    # 1초마다 그 시각 직전 프레임의 값. NaN은 아직 창 길이만큼 쌓이지 않은 구간
    print(df.iloc[every_second(t)][["t", "yawn_count"]].round(3).to_string(index=False, na_rep="NaN"))


if __name__ == "__main__":
    main()

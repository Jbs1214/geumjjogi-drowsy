# 하품 판정(MAR 규칙).
# 입 벌림(mar > MAR_YAWN)이 YAWN_MIN_SEC 이상 이어지면 하품 1회로, 그 시간에 닿은 프레임에서 센다 (alarm.py와 같은 방식).
#
# 실행: python -m rule_based.yawn <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.common import over, runs
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


def yawning(t: np.ndarray, mar: np.ndarray) -> np.ndarray:
    """프레임별 지금 하품 중인지(bool). 입을 벌린 시간이 YAWN_MIN_SEC에 닿은 프레임부터 다물 때까지 참이다.

    yawn_times()가 센 시각에서 참이 되고, 그 프레임까지의 기록만 쓰므로 실시간으로도 같은 값이 나온다.
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros(len(t), dtype=bool)
    for s, e in zip(*runs(is_open(mar))):
        out[s:e + 1] = t[s:e + 1] - t[s] >= config.YAWN_MIN_SEC
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="랜드마크 표(CSV)에서 입 벌림 구간과 하품 횟수를 낸다")
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


if __name__ == "__main__":
    main()

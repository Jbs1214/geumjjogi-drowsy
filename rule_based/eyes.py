# 프레임마다 눈을 감았는지 판정하고, 연속으로 감은 구간을 찾는다 (기획서 3-4 ①의 첫 단계).
#
# 실행: python -m rule_based.eyes <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config


def is_closed(ear: np.ndarray) -> np.ndarray:
    """프레임별 ear → 눈 감음 여부(bool). 얼굴 미검출(NaN)은 감지 못 한 것이므로 False."""
    ear = np.asarray(ear, dtype=float)
    return ~np.isnan(ear) & (np.nan_to_num(ear, nan=np.inf) < config.EAR_CLOSED)


def closed_runs(t: np.ndarray, closed: np.ndarray) -> list[tuple[float, float]]:
    """연속으로 감은 구간마다 (시작 초, 끝 초). 길이는 끝 − 시작 (첫 감음 프레임 → 마지막 감음 프레임)."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(closed, dtype=int), 0]))
    return [(float(t[s]), float(t[e - 1])) for s, e in zip(edges[::2], edges[1::2])]


def main() -> None:
    parser = argparse.ArgumentParser(description="랜드마크 표(CSV)에서 눈 감은 프레임과 연속 구간을 찾는다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    closed = is_closed(df["ear"].to_numpy())
    print(f"{args.csv}: {len(df)}프레임, 눈 감음 {closed.sum()}프레임 ({closed.mean():.1%}), 기준 ear < {config.EAR_CLOSED}")

    runs = pd.DataFrame(closed_runs(df["t"].to_numpy(), closed), columns=["start", "end"])
    runs["length"] = runs["end"] - runs["start"]
    print(runs.sort_values("length", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

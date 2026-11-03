# 프레임마다 눈을 감았는지 판정하고, 연속으로 감은 구간을 찾는다 (기획서 3-4 ①의 첫 단계).
#
# 실행: python -m rule_based.eyes <landmarks.extract CSV>

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.common import runs, under


def is_closed(ear: np.ndarray) -> np.ndarray:
    """프레임별 ear → 눈 감음 여부(bool). 얼굴 미검출(NaN)은 감지 못 한 것이므로 False."""
    return under(ear, config.EAR_CLOSED)


def ear_threshold(ear: np.ndarray, path: str | Path = config.BEST_PARAMS_PATH) -> tuple[float, str]:
    """클립의 ear → (타이머 입력용 임계값, 값의 출처). 파일이 없거나 얼굴 프레임이 5개 이하이면 config.EAR_CLOSED."""
    path = Path(path)
    if not path.exists():
        return config.EAR_CLOSED, f"{path} 없음, EAR_CLOSED"
    valid = np.asarray(ear, dtype=float)
    valid = valid[~np.isnan(valid)]
    if len(valid) <= 5:
        return config.EAR_CLOSED, "얼굴 프레임 부족, EAR_CLOSED"
    ratio = json.loads(path.read_text(encoding="utf-8"))["params"]["ear_ratio"]
    return float(ratio * np.percentile(valid, config.EAR_BASE_PERCENTILE)), f"{path}: ear_ratio {ratio:g} × P{config.EAR_BASE_PERCENTILE}"


def closed_runs(t: np.ndarray, closed: np.ndarray) -> list[tuple[float, float]]:
    """연속으로 감은 구간마다 (시작 초, 끝 초). 길이는 끝 − 시작 (첫 감음 프레임 → 마지막 감음 프레임)."""
    return [(float(t[s]), float(t[e])) for s, e in zip(*runs(closed))]


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

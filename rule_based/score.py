# 졸음 점수(0~100)와 단계(정상/주의/졸음). 최근 config.WINDOW_SEC초의 지표 4개를 0~1로 맞춰 가중합한다.
# 지표: PERCLOS, 가장 긴 눈 감김, 하품 횟수, 고개 숙임 비율. 창이 차기 전에는 점수 NaN, 단계 "-".
#
# 실행: python -m rule_based.score <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.closure import longest_closure
from rule_based.common import every_second
from rule_based.head import head_down_ratio
from rule_based.perclos import perclos
from rule_based.yawn import yawn_count


def indicators(t: np.ndarray, ear: np.ndarray, mar: np.ndarray, pitch: np.ndarray) -> pd.DataFrame:
    """프레임마다 지표 4개를 0~1로 맞춘 표. 열 이름은 config.WEIGHTS의 키."""
    return pd.DataFrame({
        "perclos": np.clip(perclos(t, ear) / config.PERCLOS_FULL, 0, 1),
        "closure": np.clip(longest_closure(t, ear) / config.CLOSURE_FULL_SEC, 0, 1),
        "yawn": np.clip(yawn_count(t, mar) / config.YAWN_FULL, 0, 1),
        "head_down": head_down_ratio(t, pitch),
    })


def drowsiness_score(parts: pd.DataFrame) -> np.ndarray:
    """indicators() 표 → 졸음 점수 0~100. 지표 중 하나라도 NaN이면 NaN."""
    return 100 * sum(w * parts[name].to_numpy() for name, w in config.WEIGHTS.items())


def stage(score: np.ndarray) -> np.ndarray:
    """졸음 점수 → 단계. STAGE_DROWSY 이상 졸음, STAGE_WARN 이상 주의, 그 밑은 정상, NaN은 "-"."""
    score = np.asarray(score, dtype=float)
    return np.where(np.isnan(score), "-",
                    np.where(score >= config.STAGE_DROWSY, "졸음",
                             np.where(score >= config.STAGE_WARN, "주의", "정상")))


def main() -> None:
    parser = argparse.ArgumentParser(description="랜드마크 표(CSV)에서 1초마다 지표 4개, 졸음 점수, 단계를 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t = df["t"].to_numpy()
    parts = indicators(t, df["ear"].to_numpy(), df["mar"].to_numpy(), df["pitch"].to_numpy())
    df = pd.concat([df[["t"]], parts], axis=1)
    df["score"] = drowsiness_score(parts)
    df["stage"] = stage(df["score"])
    # 영상이 WINDOW_SEC보다 짧거나 얼굴이 계속 안 잡히면 점수가 모두 NaN이라 최고점이 없다
    best = f"{df['score'].max():.1f}점 ({df['t'][df['score'].idxmax()]:.3f}초)" if df["score"].notna().any() else "없음"
    print(f"{args.csv}: 최고 {best}, 주의 {config.STAGE_WARN} / 졸음 {config.STAGE_DROWSY}, 가중치 {config.WEIGHTS}")

    # 1초마다 그 시각 직전 프레임의 값. 지표는 0~1로 맞춘 값
    print(df.iloc[every_second(t)].round(3).to_string(index=False, na_rep="NaN"))


if __name__ == "__main__":
    main()

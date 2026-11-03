# 연속 눈 감김 시간으로 단계를 정한다 (기획서 3-4): config.CAUTION_SEC 이상 주의, config.DROWSY_SEC 이상 졸음.
# 규칙과 모델이 같은 타이머를 쓰도록 입력은 closed(True/False)와 프레임 시각 t만 받는다.
# EAR 규칙은 eyes.ear_threshold()로, 모델 확률은 ProbToClosed로 closed를 만든다.
# 감은 시간은 eyes.closed_runs()와 같은 정의다 (첫 감음 프레임 시각 → 지금 프레임 시각).
#
# 실행: python -m rule_based.alarm <landmarks.extract CSV> [--best-params <경로>]

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.common import under
from rule_based.eyes import closed_runs, ear_threshold

NORMAL, CAUTION, DROWSY = "정상", "주의", "졸음"


class EyeClosureAlarm:
    def __init__(self) -> None:
        self._closed_since: float | None = None

    def update(self, closed: bool, t: float, eyes_visible: bool = True) -> tuple[str, float]:
        """프레임 하나(눈 감음 여부, 시각 t초) → (단계, 연속 눈 감김 초). 세지 않는 프레임이면 (정상, 0.0)이고 처음부터 다시 잰다.

        eyes_visible=False의 현재 동작: _resume_closure 훅이 구현되기 전에는 이 값을 판정에 쓰지 않고 closed만 본다.
        closed=False면 타이머를 지우고 (정상, 0.0)을 낸다. 이어받기와 '판정 불가' 표시는 없다. closed=True면 계속 센다.
        """
        counting = bool(closed) and not self._yawn_excluded(t)
        if not counting and not self._resume_closure(t, eyes_visible):
            self._closed_since = None
            return NORMAL, 0.0
        if self._closed_since is None:
            self._closed_since = t
        closed_sec = t - self._closed_since
        if closed_sec >= config.DROWSY_SEC:
            return DROWSY, closed_sec
        return (CAUTION if closed_sec >= config.CAUTION_SEC else NORMAL), closed_sec

    def _yawn_excluded(self, t: float) -> bool:
        """훅 (기획서 3-4, 미구현): 하품 판정 직후 1.0초 이내이거나 MAR이 높은 동안 True를 내 세지 않게 한다."""
        return False

    def _resume_closure(self, t: float, eyes_visible: bool) -> bool:
        """훅 (기획서 3-4, 미구현): 직전 눈 감김이 0.5초 이상이면 눈이 안 보이는 동안 True를 내 이어서 세게 한다."""
        return False


class ProbToClosed:
    """모델의 눈감김 확률 → closed. 켜기 config.PROB_ON, 끄기 config.PROB_OFF, 그 사이는 이전 상태를 유지해 흔들림을 막는다."""

    def __init__(self) -> None:
        self._closed = False

    def update(self, prob: float) -> bool:
        if prob >= config.PROB_ON:
            self._closed = True
        elif prob <= config.PROB_OFF:
            self._closed = False
        return self._closed


def stages(t: np.ndarray, closed: np.ndarray) -> np.ndarray:
    """표 전체 → 프레임별 단계."""
    alarm = EyeClosureAlarm()
    return np.array([alarm.update(bool(c), float(ti))[0] for ti, c in zip(t, closed)])


def alert_times(t: np.ndarray, closed: np.ndarray) -> list[tuple[float, str]]:
    """표 전체 → 단계가 주의·졸음으로 오른 (시각 초, 단계) 목록. 한 번 감은 구간에서 단계마다 한 번만 센다."""
    levels = stages(t, closed)
    previous = np.concatenate(([NORMAL], levels[:-1]))
    return [(float(t[i]), str(levels[i])) for i in np.flatnonzero((levels != NORMAL) & (levels != previous))]


def main() -> None:
    parser = argparse.ArgumentParser(description="랜드마크 표(CSV)에서 눈 감음 구간마다 주의·졸음이 된 시각을 낸다")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--best-params", type=Path, default=Path(config.BEST_PARAMS_PATH), help="임계값 파일. 없으면 EAR_CLOSED를 쓴다")
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t, ear = df["t"].to_numpy(), df["ear"].to_numpy()
    threshold, source = ear_threshold(ear, args.best_params)
    closed = under(ear, threshold)
    alerts = alert_times(t, closed)
    n_caution, n_drowsy = (sum(level == name for _, level in alerts) for name in (CAUTION, DROWSY))
    print(f"{args.csv}: 주의 {n_caution}회, 졸음 {n_drowsy}회, 기준 ear < {threshold:.3f} ({source})가 "
          f"{config.CAUTION_SEC:g}초 / {config.DROWSY_SEC:g}초 이상")

    runs = pd.DataFrame(closed_runs(t, closed), columns=["start", "end"])
    runs["length"] = runs["end"] - runs["start"]
    for column, name in (("caution_at", CAUTION), ("drowsy_at", DROWSY)):
        times = [a for a, level in alerts if level == name]
        runs[column] = [next((a for a in times if s <= a <= e), np.nan) for s, e in zip(runs["start"], runs["end"])]
    print(runs.sort_values("length", ascending=False).round(3).to_string(index=False, na_rep="-"))


if __name__ == "__main__":
    main()

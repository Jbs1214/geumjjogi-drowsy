# 눈 감음이 config.ALERT_SEC 이상 이어지면 경고한다 (기획서 3-4 ①의 2초 경고).
# 실시간 데모는 EyeClosureAlarm.update()를 프레임마다 부른다. 감은 시간은 eyes.closed_runs()와 같은 정의
# (첫 감음 프레임 시각 → 지금 프레임 시각)라서 구간 길이가 ALERT_SEC 이상인 구간에서만 경고가 난다.
#
# 실행: python -m rule_based.alarm <landmarks.extract CSV>

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks.features import add_features
from rule_based import config
from rule_based.eyes import closed_runs, is_closed


class EyeClosureAlarm:
    def __init__(self) -> None:
        self._closed_since: float | None = None

    def update(self, t: float, ear: float) -> bool:
        """프레임 하나(시각 t초, ear) → 지금 경고 중인지. 눈을 뜨거나 얼굴이 안 잡히면 감은 시간을 처음부터 다시 잰다."""
        if not is_closed(np.array([ear]))[0]:
            self._closed_since = None
            return False
        if self._closed_since is None:
            self._closed_since = t
        return t - self._closed_since >= config.ALERT_SEC


def alert_times(t: np.ndarray, ear: np.ndarray) -> list[float]:
    """표 전체 → 경고가 울리기 시작한 시각(초) 목록. 한 번 감은 구간에서 경고는 한 번만 센다."""
    alarm = EyeClosureAlarm()
    alerts = np.array([alarm.update(float(ti), float(ei)) for ti, ei in zip(t, ear)], dtype=bool)
    starts = np.flatnonzero(np.diff(np.r_[0, alerts.astype(int)]) == 1)
    return [float(t[i]) for i in starts]


def main() -> None:
    parser = argparse.ArgumentParser(description="랜드마크 표(CSV)에서 눈 감음 구간마다 경고가 울렸는지와 울린 시각을 낸다")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.csv))
    t, ear = df["t"].to_numpy(), df["ear"].to_numpy()
    alerts = alert_times(t, ear)
    print(f"{args.csv}: 경고 {len(alerts)}회, 기준 ear < {config.EAR_CLOSED}가 {config.ALERT_SEC:g}초 이상")

    runs = pd.DataFrame(closed_runs(t, is_closed(ear)), columns=["start", "end"])
    runs["length"] = runs["end"] - runs["start"]
    runs["alert_at"] = [next((a for a in alerts if s <= a <= e), np.nan) for s, e in zip(runs["start"], runs["end"])]
    print(runs.sort_values("length", ascending=False).round(3).to_string(index=False, na_rep="-"))


if __name__ == "__main__":
    main()

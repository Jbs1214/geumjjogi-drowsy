# 눈 감음이 config.ALERT_SEC 이상 이어지면 경고한다 (기획서 3-4의 졸음 3초 경고).
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


NORMAL, WARN, DROWSY = 0, 1, 2


class ClosureTimer:
    """프레임 시각 기준 눈 감음 타이머. 규칙과 모델이 같이 쓴다. 첫 감음 프레임 시각부터 지금 프레임 시각까지 잰다."""

    def __init__(self, warn_sec: float = config.WARN_SEC, alert_sec: float = config.ALERT_SEC, grace_sec: float = 0.0,
                 carry_max_sec: float | None = None) -> None:
        """grace_sec: 뜬 시간이 이보다 짧으면 타이머를 이어 간다(0이면 뜨자마자 다시 잰다). 지금까지의 프레임만 쓴다.
        carry_max_sec: 판정 불가·고개 숙임으로 이어 센 시간의 상한. 넘으면 다시 잰다. None이면 상한 없음."""
        self.warn_sec, self.alert_sec, self.grace_sec, self.carry_max_sec = warn_sec, alert_sec, grace_sec, carry_max_sec
        self._closed_since: float | None = None
        self._last_closed: float | None = None
        self._carry_since: float | None = None
        self.elapsed = 0.0

    def _stage(self) -> int:
        if self.elapsed >= self.alert_sec:
            return DROWSY
        return WARN if self.elapsed >= self.warn_sec else NORMAL

    def update(self, t: float, closed: bool, unknown: bool = False) -> int:
        """프레임 하나(시각 t초, 감음 여부) → 단계 NORMAL, WARN, DROWSY. 뜨면(끊김 허용을 넘으면) 처음부터 다시 잰다.
        unknown(판정 불가, 눈이 안 보임): 이미 CARRY_MIN_SEC 이상 감고 있었으면 감은 것으로 이어받아 계속 재고, 아니면 뜸으로 본다
        (수정 기획서 F 5-1)."""
        carrying = False
        if unknown:
            if self._closed_since is not None and self.elapsed >= config.CARRY_MIN_SEC - 1e-9:
                if self._carry_since is None:
                    self._carry_since = t
                carrying = self.carry_max_sec is None or t - self._carry_since <= self.carry_max_sec + 1e-9
            closed = carrying
        if not carrying:
            self._carry_since = None
        if not closed:
            # 마지막 감음 프레임에서 지금까지가 grace_sec보다 짧으면 끊김으로 보고 이어 간다. 15 FPS에서 0.2초면 2프레임까지다
            if self._closed_since is not None and t - self._last_closed < self.grace_sec - 1e-9:
                return self._stage()            # 끊김 동안은 감은 시간을 늘리지 않고 지금 단계를 유지한다
            self._closed_since, self._last_closed, self.elapsed = None, None, 0.0
            return NORMAL
        self._last_closed = t
        if self._closed_since is None:
            self._closed_since = t
        self.elapsed = t - self._closed_since
        return self._stage()


def timer_stages(t: np.ndarray, closed: np.ndarray, grace_sec: float = 0.0, unknown: np.ndarray | None = None,
                 carry_max_sec: float | None = None) -> np.ndarray:
    """프레임 시각과 프레임별 감음(과 판정 불가) → 프레임별 단계."""
    timer = ClosureTimer(grace_sec=grace_sec, carry_max_sec=carry_max_sec)
    unknown = np.zeros(len(closed), dtype=bool) if unknown is None else np.asarray(unknown, dtype=bool)
    return np.array([timer.update(float(ti), bool(ci), bool(ui)) for ti, ci, ui in zip(t, closed, unknown)], dtype=int)


def stage_starts(t: np.ndarray, stages: np.ndarray, stage: int = DROWSY) -> np.ndarray:
    """프레임별 단계 → 그 단계에 처음 닿은 시각들. 이어지는 동안에는 한 번만 센다."""
    active = np.asarray(stages) >= stage
    return np.asarray(t)[active & ~np.r_[False, active[:-1]]]


class EyeClosureAlarm:
    def __init__(self) -> None:
        self._timer = ClosureTimer()

    def update(self, t: float, ear: float) -> bool:
        """프레임 하나(시각 t초, ear) → 지금 경고 중인지. 눈을 뜨거나 얼굴이 안 잡히면 감은 시간을 처음부터 다시 잰다."""
        return self._timer.update(t, bool(is_closed(np.array([ear]))[0])) == DROWSY


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

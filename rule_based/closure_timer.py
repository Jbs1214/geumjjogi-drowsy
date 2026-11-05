# 프레임 시각 기준 연속 눈 감김 타이머 (기술서 3.1·3.2·3.3). 규칙과 모델이 같이 쓴다.
# 김민영 님 rule_based/alarm.py(EyeClosureAlarm)에 합칠 제안이다: 판정 불가·고개 숙임 이어받기(직전 감김 0.5초 이상),
# 하품 중에도 계속 셈(하품 구간을 타이머에서 빼지 않음). 합칠 위치는 PR에서 정한다. 데모·평가(eval/)는 이 파일을 쓴다.

import numpy as np

import config

NORMAL, WARN, DROWSY = 0, 1, 2


class ClosureTimer:
    """프레임 시각 기준 눈 감음 타이머. 첫 감음 프레임 시각부터 지금 프레임 시각까지 잰다."""

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
        unknown(판정 불가·고개 숙임): 이미 CARRY_MIN_SEC 이상 감고 있었으면 감은 것으로 이어받아 계속 재고, 아니면 뜸으로 본다
        (기술서 3.3)."""
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

# 수정 기획서 F의 별도 알림. 웹캠과 데모가 같이 쓴다. 졸음 단계(ClosureTimer)와는 따로 동작한다.
#   YawnTracker: 프레임마다 지금 하품 중인지 (rule_based.yawn.yawning과 같은 값, 지금까지 프레임만 씀)
#   YawnRepeat:  하품이 5분에 3회면 하품 주의, 알린 뒤 2분 쿨다운 (기획서 4, 5-3)
#   UnknownWatch: 판정 불가가 3초 이어지면 확인 필요 경고 (기획서 5-3 고개 숙임)
#   summary_markdown: 운전 후 요약 (기획서 4의 5·6)

from dataclasses import dataclass

import numpy as np

from rule_based import config


class YawnTracker:
    """mar를 하나씩 받아 하품 중인지 낸다. 입 벌림이 YAWN_MIN_SEC 이어진 프레임부터 다물 때까지 참이다."""

    def __init__(self) -> None:
        self._open_since: float | None = None

    def update(self, t: float, mar: float) -> bool:
        if mar is None or np.isnan(mar) or mar <= config.MAR_YAWN:
            self._open_since = None
            return False
        if self._open_since is None:
            self._open_since = t
        return t - self._open_since >= config.YAWN_MIN_SEC


class MouthClosedWatch:
    """mar를 하나씩 받아 입을 다문(MAR ≤ MAR_YAWN) 채 이어진 시간을 낸다. 얼굴이 안 보이면(NaN) 0이다.
    화면에서 하품을 끝내는 데 쓴다: 입을 다문 지 YAWN_DISPLAY_CLOSED_SEC가 지나면 하품을 정상으로 보인다."""

    def __init__(self) -> None:
        self._since: float | None = None

    def update(self, t: float, mar: float) -> float:
        if mar is None or np.isnan(mar) or mar > config.MAR_YAWN:
            self._since = None
            return 0.0
        if self._since is None:
            self._since = t
        return t - self._since


def mouth_closed_seconds(t: np.ndarray, mar: np.ndarray) -> np.ndarray:
    """프레임별 입 다문 채 이어진 시간 (MouthClosedWatch와 같은 값)."""
    watch = MouthClosedWatch()
    return np.array([watch.update(float(a), float(b)) for a, b in zip(t, mar)])


class YawnRepeat:
    """하품 1회 = 모델 하품 판정이 YAWN_MIN_SEC 이상 이어진 것. 최근 5분에 3회면 하품 주의를 낸다."""

    def __init__(self) -> None:
        self._since: float | None = None
        self._counted = False
        self.times: list[float] = []
        self.alerts: list[float] = []

    def update(self, t: float, yawning: bool) -> bool:
        """프레임 하나 → 이 프레임에서 하품 주의가 새로 났는지."""
        if not yawning:
            self._since, self._counted = None, False
            return False
        if self._since is None:
            self._since = t
        if self._counted or t - self._since < config.YAWN_MIN_SEC - 1e-9:
            return False
        self._counted = True
        self.times.append(self._since)
        recent = [x for x in self.times if t - x <= config.YAWN_REPEAT_WINDOW_SEC]
        cooled = not self.alerts or t - self.alerts[-1] >= config.YAWN_COOLDOWN_SEC
        if len(recent) >= config.YAWN_REPEAT and cooled:
            self.alerts.append(t)
            return True
        return False

    def active(self, t: float, show_sec: float = 5.0) -> bool:
        """화면에 하품 주의를 띄울지 (알림 뒤 show_sec초 동안)."""
        return bool(self.alerts) and t - self.alerts[-1] < show_sec


class UnknownWatch:
    """판정 불가가 UNKNOWN_ALERT_SEC 이어지면 확인 필요. 다시 판정되면 끝난다."""

    def __init__(self) -> None:
        self._since: float | None = None
        self._alerted = False
        self.alerts: list[float] = []
        self.active = False

    def update(self, t: float, unknown: bool) -> bool:
        """프레임 하나 → 이 프레임에서 확인 필요가 새로 났는지."""
        if not unknown:
            self._since, self._alerted, self.active = None, False, False
            return False
        if self._since is None:
            self._since = t
        self.active = t - self._since >= config.UNKNOWN_ALERT_SEC - 1e-9
        if self.active and not self._alerted:
            self._alerted = True
            self.alerts.append(t)
            return True
        return False


@dataclass(frozen=True)
class Event:
    t: float
    kind: str       # 주의, 졸음, 하품 주의, 확인 필요


def summary_markdown(events: list[Event], seconds: float, title: str) -> str:
    """이벤트 목록 → 시간순 요약 마크다운."""
    events = sorted(events, key=lambda e: e.t)
    counts = {kind: sum(e.kind == kind for e in events) for kind in ("주의", "졸음", "하품 주의", "확인 필요")}
    lines = [f"# {title}", "", f"- 길이 {seconds / 60:.1f}분",
             "- " + ", ".join(f"{k} {v}회" for k, v in counts.items()), "", "| 시각 | 이벤트 |", "| --- | --- |"]
    lines += [f"| {int(e.t // 60):02d}:{e.t % 60:05.2f} | {e.kind} |" for e in events] or ["| - | 없음 |"]
    return "\n".join(lines) + "\n"

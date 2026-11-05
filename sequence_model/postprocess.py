# 모델의 프레임별 확률 → 클래스. 눈감김 판정이 끊기거나 너무 민감한 것을 줄이는 후처리다 (README "성능 개선 판정 기준",
# "깜빡임·실눈 민감도 후처리").
#   enter: 눈감김 확률이 이 값 이상일 때만 눈감김에 들어간다. None이면 argmax가 눈감김일 때 들어간다.
#   exit(히스테리시스): 들어간 뒤에는 눈감김 확률이 exit 이상인 동안 유지하고, exit 미만이면 끈다(기술서 3.1, 김민영 님과 합의).
#         None이면 argmax가 눈감김인 동안 유지한다.
#   고개숙임·눈 상태: config.HEAD_FROM_PITCH이면 고개숙임은 모델 출력에서 빼고 pitch 규칙으로 정하고, 한쪽 눈 뜸·숙인 채 두 눈 감음
#         규칙을 적용한다(eye_flags, apply_rules, 기술서 2.4·7·8과 11.04 사용자 결정). 고개숙임은 정면 기준 상대 pitch다(PitchBaseline).
#   grace(끊김 허용): 눈감김 타이머(rule_based.closure_timer.ClosureTimer)가 뜬 시간이 grace보다 짧으면 이어 간다.
#   min_close: 이보다 짧은 눈감김 구간은 구간 채점에서 지운다. 경고 타이머는 그대로다.
# enter, exit, grace는 지금까지의 프레임만 쓰므로 파일 평가와 실시간(demo.webcam)이 같은 값을 낸다.

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

import config
from rule_based.common import runs


@dataclass(frozen=True)
class Post:
    exit: float | None = None
    grace: float = 0.0
    enter: float | None = None
    min_close: float = 0.0
    yawn_mode: str = "model"            # 하품 판정: model, gate, union, rule (README "하품 판정과 이어받기 상한")
    carry_max: float | None = None      # 고개 숙임·판정 불가 이어받기 상한(초). None이면 상한 없음

    def label(self) -> str:
        def show(v):
            return "없음" if v is None else v
        text = f"enter={show(self.enter)}, exit={show(self.exit)}, grace={self.grace}, min_close={self.min_close}"
        if self.yawn_mode != "model" or self.carry_max is not None:
            text += f", yawn={self.yawn_mode}, carry_max={show(self.carry_max)}"
        return text


YAWN_MODES = ("model", "gate", "union", "rule")
CARRY_CAPS = (None, 3.0, 2.0, 1.0)


def yawn_classes(classes: np.ndarray, mouth_closed: np.ndarray, rule_yawn: np.ndarray, mode: str) -> np.ndarray:
    """하품 판정 후보를 적용한 클래스. 눈감김·고개숙임·판정 불가는 바꾸지 않는다.
    gate: 모델 하품 ∧ 지금 입 벌림 / union: 모델 하품 ∨ MAR 하품 규칙 / rule: MAR 하품 규칙만."""
    out = np.asarray(classes).copy()
    rule_yawn = np.asarray(rule_yawn, dtype=bool)
    if mode == "gate":
        out[(out == config.YAWN) & (np.asarray(mouth_closed) > 0)] = config.NORMAL
    elif mode == "union":
        out[(out == config.NORMAL) & rule_yawn] = config.YAWN
    elif mode == "rule":
        out[out == config.YAWN] = config.NORMAL
        out[(out == config.NORMAL) & rule_yawn] = config.YAWN
    elif mode != "model":
        raise ValueError(f"없는 하품 판정: {mode}")
    return out


def shown_yawn(classes: np.ndarray, mouth_closed: np.ndarray, rule_yawn: np.ndarray, mode: str) -> np.ndarray:
    """화면과 채점에 쓰는 하품 여부. 하품 판정 후보를 적용한 뒤 입 다문 지 0.5초가 지나면 끝낸다."""
    shown = display_classes(yawn_classes(classes, mouth_closed, rule_yawn, mode), np.full(len(classes), np.inf), mouth_closed=mouth_closed)
    return shown == config.YAWN


NONE = Post()
EXITS = (None, 0.4, 0.3, 0.2)
GRACES = (0.0, 0.2, 0.33)
ENTERS = (None, 0.6, 0.7, 0.8)
MIN_CLOSES = (0.0, 0.5)
# 바꾸는 폭이 작은 순서 (같은 점수면 앞의 것). 앞 12개는 처음 후보와 같다
CANDIDATES = tuple(Post(e, g, en, m) for m in MIN_CLOSES for en in ENTERS for g in GRACES for e in EXITS
                   if not (e is not None and en is not None and e > en))
FILE_NAME = "post.json"


class Hysteresis:
    """프레임 확률을 하나씩 받아 클래스를 낸다. 판정하지 않는 프레임(None)에서는 눈감김 유지를 끊는다."""

    def __init__(self, post: Post = NONE) -> None:
        self.post, self.closed = post, False

    def step(self, prob: np.ndarray | None) -> int:
        if prob is None:
            self.closed = False
            return -1
        k = int(np.argmax(prob))
        p = float(prob[config.CLOSED])
        keep = k == config.CLOSED if self.post.exit is None else p >= self.post.exit
        if self.closed and keep:
            return config.CLOSED
        can_enter = k == config.CLOSED if self.post.enter is None else p >= self.post.enter
        if can_enter:
            self.closed = True
            return config.CLOSED
        self.closed = False
        # 눈감김이 아니면 나머지 클래스 중 가장 높은 것. 고개숙임은 pitch 규칙으로 정하므로 모델 출력에서 뺀다
        others = np.array(prob, dtype=float)
        others[config.CLOSED] = -np.inf
        if config.HEAD_FROM_PITCH:
            others[config.HEAD] = -np.inf
        return int(np.argmax(others))


class PitchBaseline:
    """정면 pitch 기준 (고개 숙임 상대 판정, config.HEAD_PITCH_RELATIVE). 세션 시작 config.PITCH_BASE_SEC 안에서 두 눈을 뜬
    (두 EAR ≥ rule_based/config.EAR_CLOSED) 검출 프레임의 pitch 중앙값이다. 감은 채 숙인 프레임과, 그 시점 기준으로 이미 고개 숙임인
    프레임(기준이 없을 때는 절대 HEAD_PITCH_DOWN, 있으면 상대 HEAD_PITCH_DOWN_REL)은 기준에 넣지 않는다.
    지금까지의 프레임만 쓰므로 파일 평가와 실시간이 같다. 아직 없으면 NaN이다."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self._values: list[float] = []
        self.value = float("nan")

    def update(self, t: float, vector: np.ndarray) -> float:
        """시각 t의 원시 특징 행(11개) → 지금까지의 정면 pitch 기준."""
        import bisect

        from rule_based.config import EAR_CLOSED

        if self._t0 is None:
            self._t0 = t
        row = np.asarray(vector, dtype=float)
        get = lambda name: row[config.FEATURES.index(name)]
        pitch = get("pitch")
        down = pitch > config.HEAD_PITCH_DOWN if np.isnan(self.value) else pitch - self.value > config.HEAD_PITCH_DOWN_REL
        if (t - self._t0 <= config.PITCH_BASE_SEC + 1e-9 and get(config.FACE_COLUMN) > 0.5
                and min(get("ear_l"), get("ear_r")) >= EAR_CLOSED and not down):
            bisect.insort(self._values, float(pitch))
            n = len(self._values)
            self.value = (self._values[(n - 1) // 2] + self._values[n // 2]) / 2
        return self.value


def pitch_baselines(t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """프레임 시각과 원시 특징 (N, 11) → 프레임별 정면 pitch 기준 (PitchBaseline과 같다)."""
    baseline = PitchBaseline()
    return np.array([baseline.update(float(ti), row) for ti, row in zip(t, X)], dtype=float)


def team_sides(weights) -> bool:
    """가중치 경로의 tag 폴더(runs/v8/…) → 팀 좌우 기준으로 학습했는지 (config.TEAM_SIDES_FROM_TAG 이상)."""
    import re

    found = re.fullmatch(r"v(\d+)", Path(weights).parent.name)
    return bool(found) and int(found.group(1)) >= config.TEAM_SIDES_FROM_TAG


def swap_ears(X: np.ndarray) -> np.ndarray:
    """특징 (…, 11)의 ear_l·ear_r을 바꾼다. 이전 좌우 기준(v1~v7)과 팀 기준(v8~)은 이 두 열만 다르다(README 11.04)."""
    out = np.array(X, copy=True)
    left, right = config.FEATURES.index("ear_l"), config.FEATURES.index("ear_r")
    out[..., [left, right]] = out[..., [right, left]]
    return out


def eye_flags(X: np.ndarray, base: np.ndarray | float | None = None) -> np.ndarray:
    """원시 특징 행 (N, 11) → (N, 3) bool: 고개 숙임, 한쪽 눈만 감음(다른 눈은 뜸), 두 눈 모두 감음. 얼굴 미검출 행은 모두 거짓.
    고개 숙임: base(행별 정면 pitch, PitchBaseline)를 주고 config.HEAD_PITCH_RELATIVE이면 pitch − base > HEAD_PITCH_DOWN_REL,
    아니면(또는 base가 NaN이면) pitch > HEAD_PITCH_DOWN. 기준은 rule_based/config.EAR_CLOSED, config.BLINK_CLOSED, config.WINK_BLINK_DIFF."""
    from rule_based.config import EAR_CLOSED

    X = np.atleast_2d(np.asarray(X, dtype=float))
    col = {name: X[:, config.FEATURES.index(name)] for name in ("pitch", "ear_l", "ear_r", "blink_l", "blink_r")}
    found = X[:, config.FEATURES.index(config.FACE_COLUMN)] > 0.5
    ear_max = np.maximum(col["ear_l"], col["ear_r"])
    down = col["pitch"] > config.HEAD_PITCH_DOWN
    if base is not None and config.HEAD_PITCH_RELATIVE:
        base = np.broadcast_to(np.asarray(base, dtype=float), col["pitch"].shape)
        down = np.where(np.isnan(base), down, col["pitch"] - np.nan_to_num(base) > config.HEAD_PITCH_DOWN_REL)
    head_down = found & down
    one_open = found & (ear_max >= EAR_CLOSED) & (np.abs(col["blink_l"] - col["blink_r"]) >= config.WINK_BLINK_DIFF)
    both_closed = found & (ear_max < EAR_CLOSED) & (np.minimum(col["blink_l"], col["blink_r"]) >= config.BLINK_CLOSED)
    return np.column_stack([head_down, one_open, both_closed])


def apply_rules(state: int, flags) -> int:
    """판정한 칸(state ≥ 0)에 눈 상태·고개 숙임 규칙을 적용한다. flags = (고개 숙임, 한쪽 눈만 감음, 두 눈 감음).
    ① 한쪽 눈만 감았으면 눈감김이 아니다(정상) ② 고개를 숙였으면 두 눈이 감겼을 때 눈감김, 아니면 고개숙임.
    우선순위 눈감김 > 고개숙임 > 하품 > 정상(기술서 2.4). 판정 불가(-1)는 그대로 둔다."""
    if state < 0:
        return state
    head_down, one_open, both_closed = (bool(f) for f in flags)
    if state == config.CLOSED and one_open:
        state = config.NORMAL
    if config.HEAD_FROM_PITCH and head_down:
        return config.CLOSED if (state == config.CLOSED or both_closed) else config.HEAD
    return state


def decide(probs: np.ndarray, judged: np.ndarray, post: Post = NONE, flags: np.ndarray | None = None) -> np.ndarray:
    """프레임별 확률 (N, 4)와 판정 여부 → 프레임별 클래스 (판정 안 함은 -1). flags: 프레임별 eye_flags 결과."""
    h = Hysteresis(post)
    states = [h.step(p if j else None) for p, j in zip(probs, judged)]
    if flags is not None:
        states = [apply_rules(s, f) for s, f in zip(states, flags)]
    return np.array(states, dtype=int)


def bridge(times: np.ndarray, mask: np.ndarray, grace: float) -> np.ndarray:
    """감음 사이의 짧은 끊김을 잇는다. ClosureTimer(grace_sec)가 타이머를 이어 가는 끊김과 같은 조건이다."""
    out = np.asarray(mask, dtype=bool).copy()
    if grace <= 0:
        return out
    for start, end in zip(*runs(~out)):
        if start > 0 and end < len(out) - 1 and times[end] - times[start - 1] < grace - 1e-9:
            out[start:end + 1] = True
    return out


def drop_short(mask: np.ndarray, step: float, min_close: float) -> np.ndarray:
    """min_close보다 짧은 감음 구간을 지운다. 구간 길이는 프레임 수 × step이다."""
    out = np.asarray(mask, dtype=bool).copy()
    if min_close <= 0:
        return out
    for start, end in zip(*runs(out)):
        if (end - start + 1) * step < min_close - 1e-9:
            out[start:end + 1] = False
    return out


def display_classes(classes: np.ndarray, elapsed: np.ndarray, min_sec: float = config.DISPLAY_MIN_CLOSE_SEC,
                    mouth_closed: np.ndarray | None = None) -> np.ndarray:
    """화면에 보일 클래스. 이어서 감은 시간이 min_sec보다 짧은 눈감김은 정상으로 보인다(깜빡임).
    mouth_closed(입 다문 채 이어진 초)를 주면, 입을 다문 지 YAWN_DISPLAY_CLOSED_SEC가 지난 하품도 정상으로 보인다."""
    from config import YAWN_DISPLAY_CLOSED_SEC

    out = np.asarray(classes).copy()
    out[(out == config.CLOSED) & (np.asarray(elapsed) < min_sec - 1e-9)] = config.NORMAL
    if mouth_closed is not None:
        out[(out == config.YAWN) & (np.asarray(mouth_closed) >= YAWN_DISPLAY_CLOSED_SEC - 1e-9)] = config.NORMAL
    return out


def from_dict(data: dict) -> Post:
    names = {f.name for f in fields(Post)}
    values = {k: v for k, v in data.items() if k in names}
    values["grace"] = float(values.get("grace", 0.0))
    values["min_close"] = float(values.get("min_close", 0.0))
    return Post(**values)


def save(path: Path, model: str, post: Post, extra: dict) -> None:
    path.write_text(json.dumps({"model": model, **asdict(post), **extra}, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> tuple[str, Post] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["model"], from_dict(data)


def for_weights(weights: Path, model_name: str) -> Post:
    """가중치 폴더의 post.json이 같은 모델 종류에 정한 후처리. 없으면 후처리 없음."""
    found = load(Path(weights).parent / FILE_NAME)
    return found[1] if found and found[0] == model_name else NONE

# 웹캠(또는 영상 파일)을 실시간으로 판정해 OpenCV 창에 4클래스 판정, 눈감김 타이머, 규칙 경고를 그린다.
# 영상과 프레임은 저장하지 않고 외부로 보내지 않는다. 끝나면 처리 속도와 판정 시각만 출력한다. 창에서 q나 Esc로 끝낸다.
#
# 모델 입력은 파일 평가(eval.evaluate.model_frames)와 같다: 15 FPS 격자마다 가장 가까운 프레임을 골라 3초(45칸) 창을 만든다.
# 격자 시각 g는 g 뒤의 프레임이 들어온 순간 정해진다(지연은 프레임 하나 이하).
# 규칙은 파일 평가와 달리 지금까지 들어온 검출 프레임의 분위를 평소 EAR로 쓴다(미래를 모르기 때문이다).
#
# 실행: python -m demo.webcam [--camera 0] [--weights runs/v1/cnn1d_v01.pt] [--mute] [--record webcam|test2]
#       python -m demo.webcam --input 영상.mp4 --no-window      화면 없이 파일 판정과 같은지 확인한다

import argparse
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config
from sequence_model.build_windows import feature_matrix
from sequence_model.postprocess import NONE, Hysteresis, Post, display_classes, for_weights, yawn_classes
from rule_based.alarm import DROWSY, NORMAL, WARN, ClosureTimer
from rule_based.alerts import Event, MouthClosedWatch, UnknownWatch, YawnRepeat, YawnTracker, summary_markdown
from rule_based.config import BASELINE_EVERY_SEC, RULE_MIN_SEC     # 다른 모듈이 demo.webcam에서 가져가는 이름도 유지한다
from rule_based.head import HeadDown
from demo.cfr import TimedWriter
from demo.demo_video import STAGE_RGB, blend, font, panel
from eval.evaluate import BASELINE_EAR_QUANTILE, EAR_RATIO

def default_weights() -> Path:
    """runs/final.json이 고른 tag·모델의 대표 시드(_v01). 없으면 v1 cnn1d."""
    import json
    final = config.RUNS / "final.json"
    if final.exists():
        data = json.loads(final.read_text(encoding="utf-8"))
        return config.RUNS / data["tag"] / f"{data['model']}_v01.pt"
    return config.RUNS / "v1" / "cnn1d_v01.pt"


DEFAULT_WEIGHTS = default_weights()
WINDOW_NAME = "drowsy v3 webcam"
QUIT_KEYS = (ord("q"), 27)
BEEP_HZ, BEEP_MS, BEEP_GAP_SEC = 1000, 300, 0.2


def alarm_stage(judge: "LiveJudge", rule: "LiveRule") -> int:
    """화면과 경고음의 단계. 최종 구성(기술서 0장)은 규칙 눈 감김 단계와 하품 알림 중 높은 쪽이고, 아니면 모델 단계다."""
    if not judge.final:
        return judge.stage
    return max(rule.stage, WARN) if judge.history and judge.yawn_repeat.active(judge.history[-1][0]) else rule.stage


def check_active(judge: "LiveJudge", rule: "LiveRule") -> bool:
    """확인 필요 경고가 켜져 있는지. 최종 구성은 규칙(얼굴 미검출·pitch 고개 숙임, 기술서 3.3), 아니면 모델의 판정 불가다."""
    return rule.unknown_watch.active if judge.final else judge.unknown_watch.active


def current_state(judge: "LiveJudge", rule: "LiveRule") -> str:
    """화면의 현재 상태. 최종 구성은 규칙·하품 전용 모델로 정한다(기술서 3.2·3.3·4.3, 우선순위 2.4):
    판정 불가 > 눈 감김(깜빡임보다 긴 것) > 고개 숙임 > 하품 감지 > 정상. 규칙이 평소 EAR을 정하기 전에는 준비 중이다.
    최종 구성이 아니면 4클래스 화면 클래스다."""
    if not judge.history or (judge.final and not rule.ready):
        return "준비 중"
    if not judge.final:
        return config.CLASSES[judge.shown] if judge.shown >= 0 else "판정 불가"
    if not rule.found:
        return "판정 불가"
    if rule.closed_sec >= config.DISPLAY_MIN_CLOSE_SEC - 1e-9:
        return "눈 감김"
    if rule.head_down:
        return "고개 숙임"
    if judge.yawning and not judge.mask:
        return "하품 감지"
    return "정상"


def frame_vector(raw: dict) -> np.ndarray:
    """LandmarkExtractor.process() 한 행 → 특징 11개 벡터 (feature_matrix와 같은 미검출 처리)."""
    from landmarks.extract import COLUMNS
    from landmarks.features import add_features

    return feature_matrix(add_features(pd.DataFrame([raw], columns=COLUMNS)))[0]


class LiveJudge:
    """프레임을 하나씩 받아 15 FPS 격자마다 눈감김 타이머를 잰다. 모델은 0.2초(격자 JUDGE_EVERY칸)마다 판정하고 사이는 유지한다.
    타이머 입력은 파일 평가(eval.evaluate.model_scores)와 같다: 판정 불가·고개숙임은 직전 눈 감김이 0.5초 이상이면 이어서 센다
    (최종 기획서 5-1·5-3). 세션 기준 가중치(v7~)면 특징을 session.SessionBaseline으로 바꾼다.
    yawn_predictor를 주면 최종 구성이다: 하품은 하품 전용 모델이 맡고, 입 모양과 4클래스 하품 출력은 쓰지 않는다.
    mask가 참이면 마스크 착용이다: 하품 표시와 하품 알림을 끈다 (기술서 3.2)."""

    def __init__(self, predictor, post: Post = NONE, yawn_predictor=None, mask: bool = False) -> None:
        from landmarks.session import SessionBaseline

        if yawn_predictor is not None and yawn_predictor.relative != getattr(predictor, "relative", False):
            raise SystemExit("하품 전용 모델과 4클래스 모델의 입력 기준(세션 기준 여부)이 다릅니다")
        self._predictor, self._yawn_predictor = predictor, yawn_predictor
        self.final, self.mask = yawn_predictor is not None, mask
        self._yawn_on, self._yawn_since = False, None
        self.yawning = False
        self._baseline = SessionBaseline() if getattr(predictor, "relative", False) else None
        self._post = post
        self._prev: tuple[float, np.ndarray, float, bool] | None = None
        self._t0: float | None = None
        self._k = 0
        self._grid: list[tuple[np.ndarray, float, bool]] = []
        self._held: np.ndarray | None = None
        self.probs: np.ndarray | None = None        # 마지막 판정의 4클래스 확률(판정 불가면 None). 판정 로그용(기술서 6.2)
        self._hysteresis = Hysteresis(post)
        self._timer = ClosureTimer(grace_sec=post.grace, carry_max_sec=post.carry_max)
        self.yawn_repeat, self.unknown_watch = YawnRepeat(0.0) if self.final else YawnRepeat(), UnknownWatch()
        self.events: list[Event] = []
        self.history: list[tuple[float, int, int, int, float]] = []     # (격자 시각, 클래스, 단계, 화면 클래스, 연속 눈 감김 초)
        self.state, self.stage, self.closed_sec, self.shown = -1, NORMAL, 0.0, -1

    def update(self, t: float, vector: np.ndarray, mouth_closed: float = 0.0, rule_yawn: bool = False) -> None:
        """프레임 시각 t, 특징 벡터, 입 다문 채 이어진 초(MouthClosedWatch), MAR 하품 규칙(YawnTracker) → 이 프레임까지 정해진 격자를 판정한다."""
        if self._t0 is None:
            self._t0 = t
        if self._baseline is not None:
            vector = self._baseline.transform(t, vector)
        ticks = []
        while (g := self._t0 + self._k / config.FPS) <= t + 1e-9:
            # 격자 시각에 더 가까운 프레임을 고른다. 같으면 앞 프레임 (build_windows.resample_index와 같다)
            use_prev = self._prev is not None and g - self._prev[0] <= t - g
            self._grid.append(self._prev[1:] if use_prev else (vector, mouth_closed, rule_yawn))
            self._grid = self._grid[-config.WINDOW_FRAMES:]
            j = self._k - (config.WINDOW_FRAMES - 1)          # 창이 찬 뒤 몇 번째 격자인지
            window = np.array([v for v, _, _ in self._grid]) if len(self._grid) == config.WINDOW_FRAMES else None
            ticks.append((g, j, window if j >= 0 and j % config.JUDGE_EVERY == 0 else None, self._grid[-1][1], self._grid[-1][2]))
            self._k += 1
        self._prev = (t, vector, mouth_closed, rule_yawn)
        due = [w for _, _, w, _, _ in ticks if w is not None]
        probs = iter(self._predictor.predict(np.stack(due)) if due else [])
        yawn_probs = iter(self._yawn_predictor.predict(np.stack(due))[:, 0] if due and self.final else [])
        face = config.FEATURES.index(config.FACE_COLUMN)
        for g, j, window, grid_mouth, grid_rule in ticks:
            if window is not None:
                prob = next(probs)
                ok = 1 - window[:, face].mean() <= config.MAX_MISSING and window[-1, face] > 0.5
                self._held = self.probs = prob if ok else None
                if self.final:
                    self._yawn_on = bool(ok and next(yawn_probs) >= config.YAWN_ENTER)
            state = self._hysteresis.step(self._held if j >= 0 else None)
            before = self.stage
            self.stage = self._timer.update(g, state == config.CLOSED, unknown=state < 0 or state == config.HEAD)
            self.state, self.closed_sec = state, self._timer.elapsed
            # 화면: 0.5초보다 짧은 눈감김(깜빡임)과 입을 다문 지 0.5초가 지난 하품은 정상으로 보인다. 경고 단계는 그대로다
            if self.final:
                self.yawning = self._yawn_step(g)
                shown = int(display_classes(np.array([config.NORMAL if state == config.YAWN else state]), np.array([self.closed_sec]))[0])
                self.shown = config.YAWN if shown == config.NORMAL and self.yawning else shown
            else:
                yawned = yawn_classes(np.array([state]), np.array([grid_mouth]), np.array([grid_rule]), self._post.yawn_mode)
                self.shown = int(display_classes(yawned, np.array([self.closed_sec]), mouth_closed=np.array([grid_mouth]))[0])
            self.history.append((g, state, self.stage, self.shown, self.closed_sec))
            if self.mask and self.shown == config.YAWN:
                self.shown = config.NORMAL
            if self.stage > before and not self.final:
                self.events.append(Event(g, "졸음" if self.stage == DROWSY else "주의"))
            if self.yawn_repeat.update(g, not self.mask and (self.yawning if self.final else self.shown == config.YAWN)):
                self.events.append(Event(g, "하품 주의"))
            # 최종 구성의 확인 필요는 규칙(LiveRule)이 얼굴 미검출·pitch 고개 숙임으로 센다 (기술서 3.3·4.3)
            if j >= 0 and not self.final and self.unknown_watch.update(g, state < 0):
                self.events.append(Event(g, "확인 필요"))


    def _yawn_step(self, g: float) -> bool:
        """격자 시각 → 하품 전용 모델의 켜짐이 YAWN_MODEL_MIN_SEC 이상 이어졌는지."""
        if not self._yawn_on:
            self._yawn_since = None
            return False
        if self._yawn_since is None:
            self._yawn_since = g
        return g - self._yawn_since >= config.YAWN_MODEL_MIN_SEC - 1e-9


class LiveRule:
    """규칙: best_params의 ear_ratio × 평소 EAR. 평소 EAR은 지금까지 들어온 검출 프레임의 분위다."""

    def __init__(self, events: list[Event] | None = None) -> None:
        self._events = events           # 최종 구성이면 규칙의 주의·졸음을 이 목록에 남긴다
        self._ears: list[float] = []
        self._t0: float | None = None
        self._baseline, self._baseline_t = np.nan, -np.inf
        self._timer = ClosureTimer()
        self._head = HeadDown()
        self.unknown_watch = UnknownWatch()
        self.ready, self.stage, self.closed_sec = False, NORMAL, 0.0
        self.found, self.head_down = False, False
        self.history: list[tuple[float, int, float]] = []       # (시각, 단계, 연속 눈 감김 초)

    def update(self, t: float, found: bool, ear: float, pitch: float = float("nan"), ear_l: float = float("nan"), ear_r: float = float("nan")) -> None:
        """판정 불가(얼굴이 안 보임)와 고개 숙임(pitch 규칙)은 직전 눈 감김이 0.5초 이상이면 이어서 센다 (최종 기획서 5-1·5-3)."""
        if self._t0 is None:
            self._t0 = t
        if found and not np.isnan(ear):
            self._ears.append(ear)
        if t - self._baseline_t >= BASELINE_EVERY_SEC and self._ears:
            self._baseline, self._baseline_t = float(np.quantile(self._ears, BASELINE_EAR_QUANTILE)), t
        self.ready = t - self._t0 >= RULE_MIN_SEC
        closed = self.ready and found and not np.isnan(ear) and ear < EAR_RATIO * self._baseline
        head_down = self._head.update(t, pitch, ear_l, ear_r)
        before = self.stage
        self.stage = self._timer.update(t, closed, unknown=(not found) or head_down)
        self.closed_sec = self._timer.elapsed
        self.found, self.head_down = found, head_down
        if self._events is not None and self.stage > before:
            self._events.append(Event(t, "졸음" if self.stage == DROWSY else "주의"))
        # 확인 필요(기술서 3.3): 얼굴 미검출·고개 숙임이 3초 이상이고 직전 눈 감김을 이어받는 중이 아닐 때
        if self.unknown_watch.update(t, ((not found) or head_down) and self.closed_sec == 0) and self._events is not None:
            self._events.append(Event(t, "확인 필요"))
        self.history.append((t, self.stage, self.closed_sec))


class Beeper:
    """졸음 단계 동안 경고음을 반복한다. 프레임 처리를 막지 않게 별도 스레드에서 낸다."""

    def __init__(self) -> None:
        import winsound
        self._beep = winsound.Beep
        self._on, self._stop = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._on.wait(timeout=0.1) and not self._stop.is_set():
                self._beep(BEEP_HZ, BEEP_MS)
                self._stop.wait(BEEP_GAP_SEC)

    def set(self, on: bool) -> None:
        self._on.set() if on else self._on.clear()

    def close(self) -> None:
        self._on.clear()
        self._stop.set()
        self._thread.join(timeout=1.0)


def info_box(text: str, scale: float) -> np.ndarray:
    size = max(round(18 * scale), 10)
    w = round(len(text) * size * 0.75) + 2 * size
    image = Image.new("RGBA", (w, round(size * 1.8)), (0, 0, 0, 170))
    ImageDraw.Draw(image).text((size, round(size * 0.9)), text, font=font(size), fill="white", anchor="lm")
    return np.asarray(image)


def draw(frame: np.ndarray, judge: LiveJudge, rule: LiveRule, ear: float, mar: float, rate: float) -> None:
    height, width = frame.shape[:2]
    scale = max(width / 1280, 0.6)
    margin = round(16 * scale)
    stage = alarm_stage(judge, rule)
    board = panel(judge.shown, stage, rule.closed_sec if judge.final else judge.closed_sec, rule.stage, ear, mar, scale)
    blend(frame, board, margin, margin)
    note = f"처리 {rate:.1f} FPS" + ("" if rule.ready else " / 규칙 준비 중") + " / q 종료"
    top = margin + board.shape[0] + round(6 * scale)
    blend(frame, info_box(note, scale), margin, top)
    alerts = [text for text, on in (("하품 주의 (5분 3회)", judge.yawn_repeat.active(judge.history[-1][0] if judge.history else 0.0)),
                                    ("확인 필요 (판정 불가·고개 숙임 3초)", check_active(judge, rule))) if on]
    for i, text in enumerate(alerts):
        blend(frame, info_box(text, scale * 1.2), margin, top + round((i + 1) * 40 * scale))
    if stage == DROWSY:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), STAGE_RGB[DROWSY][::-1], max(round(12 * scale), 2))


def starts(history: list, value: int, column: int = 1) -> list[float]:
    out, before = [], None
    for row in history:
        now = row[column] == value
        if now and not before:
            out.append(round(row[0], 2))
        before = now
    return out


def open_source(source: str) -> tuple[cv2.VideoCapture, bool, float]:
    live = source.isdigit()
    capture = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if live else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"입력을 열 수 없습니다: {source}. 웹캠이면 연결과 번호를 확인하세요")
    return capture, live, capture.get(cv2.CAP_PROP_FPS) or 30.0


def record_path(kind: str) -> Path:
    folder = config.TEST2_DIR if kind == "test2" else config.WEBCAM_DIR
    return folder / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


def record_blind(source: str, out: Path, show: bool, max_seconds: float | None) -> dict:
    """Test2 녹화: 모델과 규칙을 돌리지 않고 카메라 화면만 보이며 실제 시각에 맞춰 저장한다. 판정 결과는 만들지 않는다."""
    capture, live, file_fps = open_source(source)
    writer = TimedWriter(out)
    frames, began, first = 0, time.perf_counter(), None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            first = time.perf_counter() if first is None else first      # 시계는 첫 프레임에서 시작한다 (녹화 영상 시각과 같게)
            t = time.perf_counter() - first if live else frames / file_fps
            if max_seconds is not None and t > max_seconds:
                break
            writer.write(frame.copy(), t)
            frames += 1
            if show:
                height, width = frame.shape[:2]
                blend(frame, info_box(f"녹화 중 (Test2, 판정 표시 없음) {t:5.1f}초 · q 종료", max(width / 1280, 0.6)), 16, 16)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                    break
    finally:
        capture.release()
        writer.close()
        if show:
            cv2.destroyAllWindows()
    return {"frames": frames, "seconds": time.perf_counter() - began, "saved": out, "saved_frames": writer.frames}


def run(source: str, weights: Path, show: bool, sound: bool, max_seconds: float | None, record: Path | None = None,
        yawn_weights: Path | None = config.YAWN_WEIGHTS, mask: bool = False) -> dict:
    from landmarks.extract import LandmarkExtractor
    from sequence_model.models import Predictor
    from sequence_model.train_yawn import load_yawn

    capture, live, file_fps = open_source(source)
    writer = TimedWriter(record) if record else None
    predictor = Predictor(weights)
    post = for_weights(weights, predictor.name)
    yawn_predictor = load_yawn(yawn_weights)
    print(f"모델 {weights}, 후처리 {post.label()}, 하품 전용 {yawn_weights if yawn_predictor else '없음(4클래스 판정)'}", flush=True)
    judge = LiveJudge(predictor, post, yawn_predictor, mask)
    rule, mouth_watch, yawn_tracker = LiveRule(judge.events if judge.final else None), MouthClosedWatch(), YawnTracker()
    extractor = LandmarkExtractor()
    beeper = Beeper() if sound and show else None
    stamps: list[float] = []
    frames, began, first = 0, time.perf_counter(), None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            first = time.perf_counter() if first is None else first      # 시계는 첫 프레임에서 시작한다 (녹화 영상 시각과 같게)
            t = time.perf_counter() - first if live else frames / file_fps
            if max_seconds is not None and t > max_seconds:
                break
            if writer:
                writer.write(frame.copy(), t)       # 표시를 그리기 전의 원본 프레임
            raw = extractor.process(frame, t)
            vector = frame_vector(raw)
            found = bool(vector[config.FEATURES.index(config.FACE_COLUMN)] > 0.5)
            ear = float(vector[config.FEATURES.index("ear")]) if found else np.nan
            mar = float(vector[config.FEATURES.index("mar")]) if found else np.nan
            mouth_closed = mouth_watch.update(t, mar)
            pitch = float(vector[config.FEATURES.index("pitch")]) if found else np.nan
            judge.update(t, vector, mouth_closed, yawn_tracker.update(t, mar))
            rule.update(t, found, ear, pitch, *(float(vector[config.FEATURES.index(name)]) for name in ("ear_l", "ear_r")))
            frames += 1
            stamps.append(time.perf_counter())
            stamps = stamps[-60:]
            rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0
            if beeper:
                beeper.set(alarm_stage(judge, rule) == DROWSY or check_active(judge, rule))
            if show:
                draw(frame, judge, rule, ear, mar, rate)
                cv2.imshow(WINDOW_NAME, frame)
                wait_ms = 1 if live else max(round((t - (time.perf_counter() - began)) * 1000), 1)    # 파일은 영상 속도로 보여 준다
                if cv2.waitKey(wait_ms) & 0xFF in QUIT_KEYS:
                    break
    finally:
        capture.release()
        if writer:
            writer.close()
        extractor.close()
        if beeper:
            beeper.close()
        if show:
            cv2.destroyAllWindows()
    spent = time.perf_counter() - began
    judged = [h[1] for h in judge.history]
    duration = judge.history[-1][0] - judge.history[0][0] if judge.history else 0.0
    where = ("웹캠" if live else Path(source).name) + (f", 녹화 {record.name}" if record else "")
    summary = summary_markdown(judge.events, duration, f"운전 후 요약 ({time.strftime('%Y-%m-%d %H:%M')}, {where})")
    if live:     # 웹캠 세션만 저장한다. 영상·얼굴은 없고 이벤트 시각만 있다
        out = config.RUNS / "sessions" / f"{time.strftime('%Y%m%d_%H%M%S')}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary, encoding="utf-8")
    return {"frames": frames, "seconds": spent, "fps": frames / spent if spent else 0.0, "summary": summary,
            "events": judge.events,
            "ticks": len(judged), "judged": float(np.mean(np.array(judged) >= 0)) if judged else 0.0,
            "history": judge.history,
            "starts": {name: starts(judge.history, value, column=3) for name, value in
                       (("눈감김", config.CLOSED), ("하품", config.YAWN), ("고개숙임", config.HEAD))},
            "drowsy": starts(judge.history, DROWSY, column=2), "rule_drowsy": starts(rule.history, DROWSY)}


def main() -> None:
    parser = argparse.ArgumentParser(description="웹캠이나 영상 파일을 v3 모델로 실시간 판정한다")
    parser.add_argument("--camera", default="0", help="웹캠 번호 (DirectShow)")
    parser.add_argument("--input", help="웹캠 대신 영상 파일. 파일 판정과 같은지 확인할 때 쓴다")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--yawn-weights", type=Path, default=config.YAWN_WEIGHTS, help="하품 전용 모델 가중치. 파일이 없으면 4클래스 판정만 쓴다")
    parser.add_argument("--mask", action="store_true", help="마스크 착용: 하품 표시와 하품 알림을 끈다")
    parser.add_argument("--no-window", action="store_true", help="화면 없이 처리한다")
    parser.add_argument("--mute", action="store_true", help="경고음을 끈다")
    parser.add_argument("--max-seconds", type=float, help="입력 시각으로 이 시간까지만 처리한다")
    parser.add_argument("--record", choices=("webcam", "test2"),
                        help="영상을 이 PC에 저장한다. webcam = 판정 화면을 보며 녹화(Val2), test2 = 판정 없이 녹화(Test2)")
    args = parser.parse_args()
    source = args.input or args.camera
    if args.record == "test2":
        result = record_blind(source, record_path("test2"), not args.no_window, args.max_seconds)
        print(f"Test2 녹화: {result['saved']} ({result['saved_frames']}프레임, {result['saved_frames'] / config.RECORD_FPS:.1f}초). 판정 결과는 만들지 않았다")
        return
    record = record_path("webcam") if args.record == "webcam" else None
    result = run(source, args.weights, not args.no_window, not args.mute, args.max_seconds, record, args.yawn_weights, args.mask)
    if record:
        print(f"Val2 녹화: {record}")
    print(f"프레임 {result['frames']}개, {result['seconds']:.1f}초, 평균 처리 {result['fps']:.1f} FPS, "
          f"15 FPS 격자 {result['ticks']}칸 중 판정 {result['judged']:.1%}")
    for name, found in result["starts"].items():
        print(f"  모델 {name} 시작 시각: {found}")
    print(f"  모델 졸음 경고: {result['drowsy']}, 규칙 졸음 경고: {result['rule_drowsy']}")
    print(result["summary"])


if __name__ == "__main__":
    main()

# 웹캠(또는 영상 파일)을 실시간으로 판정해 OpenCV 창에 4클래스 판정, 눈감김 타이머, 규칙 경고를 그린다.
# 영상과 프레임은 저장하지 않고 외부로 보내지 않는다. 끝나면 처리 속도와 판정 시각을 출력하고, 웹캠이면 요약과 판정 로그
# (기술서 6.2, runs/sessions/<시각>_log.csv)를 남긴다. 창에서 q나 Esc로 끝낸다.
#
# 모델 입력은 파일 평가(eval.evaluate.model_frames)와 같다: 15 FPS 격자마다 가장 가까운 프레임을 골라 3초(45칸) 창을 만든다.
# 격자 시각 g는 g 뒤의 프레임이 들어온 순간 정해진다(지연은 프레임 하나 이하).
# 규칙 EAR 0.48은 파일 평가와 달리 지금까지 들어온 검출 프레임의 90% 분위를 평소 값으로 쓴다(미래를 모르기 때문이다).
#
# 실행: python -m demo.webcam [--camera 0] [--weights runs/v1/cnn1d_v01.pt] [--mute] [--record webcam|test2]
#       [--grace 0.2] [--head-rel 10] [--backend MSMF|DSHOW]      이번 실행에만 후처리·고개 숙임 기준·백엔드를 바꾼다
#       python -m demo.webcam --input 영상.mp4 --no-window      화면 없이 파일 판정과 같은지 확인한다

import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config
from sequence_model.frames import feature_matrix
from sequence_model.postprocess import (NONE, Hysteresis, PitchBaseline, Post, display_classes, for_weights, apply_rules, eye_flags,
                                        swap_ears, yawn_classes)
from rule_based.closure_timer import DROWSY, NORMAL, ClosureTimer
from rule_based.alerts import Event, MouthClosedWatch, UnknownWatch, YawnRepeat, YawnTracker, summary_markdown
from demo.cfr import TimedWriter
from demo.demo_video import STAGE_NAMES, STAGE_RGB, blend, font, panel
from eval.evaluate import BASELINE_EAR_QUANTILE, EAR_RATIO

# 판정 로그 열 (기술서 6.2): 실제 시각, 영상 시각, 단계, 연속 눈 감김(초), 클래스 확률 4개, 판정 클래스, 이벤트
LOG_COLUMNS = ["wall_time", "video_t", "stage", "closed_sec", *[f"p_{name}" for name in config.CLASSES], "class", "event"]

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
RULE_MIN_SEC = 5.0          # 평소 EAR을 정할 기록이 이보다 짧으면 규칙은 준비 중이다
BASELINE_EVERY_SEC = 0.5    # 평소 EAR을 다시 구하는 간격
BEEP_HZ, BEEP_MS, BEEP_GAP_SEC = 1000, 300, 0.2


def frame_vector(raw: dict) -> np.ndarray:
    """LandmarkExtractor.process() 한 행 → 특징 11개 벡터 (feature_matrix와 같은 미검출 처리)."""
    from landmarks.extract import COLUMNS
    from landmarks.features import add_features

    return feature_matrix(add_features(pd.DataFrame([raw], columns=COLUMNS)))[0]


class LiveJudge:
    """프레임을 하나씩 받아 15 FPS 격자마다 눈감김 타이머를 잰다. 모델은 0.2초(격자 JUDGE_EVERY칸)마다 판정하고 사이는 유지한다.
    타이머 입력은 파일 평가(eval.evaluate.model_scores)와 같다: 판정 불가·고개숙임은 직전 눈 감김이 0.5초 이상이면 이어서 센다
    (최종 기획서 5-1·5-3). 세션 기준 가중치(v7~)면 특징을 session.SessionBaseline으로 바꾼다.
    입력은 지금 추출 코드의 특징(팀 좌우 기준)이다. 이전 기준으로 학습한 가중치(team 속성 없음, v1~v7)면 ear_l·ear_r을 바꿔 넣는다.
    고개 숙임은 정면 기준 상대 pitch다(PitchBaseline). 마지막 프레임의 고개 숙임은 head_down에 둔다(LiveRule이 같이 쓴다)."""

    def __init__(self, predictor, post: Post = NONE, team_input: bool = True) -> None:
        from landmarks.session import SessionBaseline

        self._predictor = predictor
        self._swap = team_input != getattr(predictor, "team", False)
        self._pitch_base = PitchBaseline()
        self.head_down = False
        self._baseline = SessionBaseline() if getattr(predictor, "relative", False) else None
        self._post = post
        self._prev: tuple[float, np.ndarray, float, bool, np.ndarray] | None = None
        self._t0: float | None = None
        self._k = 0
        self._grid: list[tuple[np.ndarray, float, bool, np.ndarray]] = []     # (특징, 입 다문 초, MAR 하품, eye_flags)
        self._held: np.ndarray | None = None
        self._hysteresis = Hysteresis(post)
        self._timer = ClosureTimer(grace_sec=post.grace, carry_max_sec=post.carry_max)
        self.yawn_repeat, self.unknown_watch = YawnRepeat(), UnknownWatch()
        self.events: list[Event] = []
        self.history: list[tuple[float, int, int, int, float]] = []     # (격자 시각, 클래스, 단계, 화면 클래스, 연속 눈 감김 초)
        self.log: list[dict] = []       # 격자마다 판정 로그 한 줄 (LOG_COLUMNS, 기술서 6.2)
        self.state, self.stage, self.closed_sec, self.shown = -1, NORMAL, 0.0, -1

    def update(self, t: float, vector: np.ndarray, mouth_closed: float = 0.0, rule_yawn: bool = False) -> None:
        """프레임 시각 t, 특징 벡터, 입 다문 채 이어진 초(MouthClosedWatch), MAR 하품 규칙(YawnTracker) → 이 프레임까지 정해진 격자를 판정한다."""
        if self._t0 is None:
            self._t0 = t
        # 고개 숙임(pitch 규칙)과 눈 상태 규칙은 세션 기준으로 바꾸기 전의 원시 값으로 판단한다 (postprocess.eye_flags)
        flags = eye_flags(vector, self._pitch_base.update(t, vector))[0]
        self.head_down = bool(flags[0])
        if self._swap:
            vector = swap_ears(vector)
        if self._baseline is not None:
            vector = self._baseline.transform(t, vector)
        ticks = []
        while (g := self._t0 + self._k / config.FPS) <= t + 1e-9:
            # 격자 시각에 더 가까운 프레임을 고른다. 같으면 앞 프레임 (frames.resample_index와 같다)
            use_prev = self._prev is not None and g - self._prev[0] <= t - g
            self._grid.append(self._prev[1:] if use_prev else (vector, mouth_closed, rule_yawn, flags))
            self._grid = self._grid[-config.WINDOW_FRAMES:]
            j = self._k - (config.WINDOW_FRAMES - 1)          # 창이 찬 뒤 몇 번째 격자인지
            window = np.array([cell[0] for cell in self._grid]) if len(self._grid) == config.WINDOW_FRAMES else None
            ticks.append((g, j, window if j >= 0 and j % config.JUDGE_EVERY == 0 else None, *self._grid[-1][1:]))
            self._k += 1
        self._prev = (t, vector, mouth_closed, rule_yawn, flags)
        due = [tick[2] for tick in ticks if tick[2] is not None]
        probs = iter(self._predictor.predict(np.stack(due)) if due else [])
        face = config.FEATURES.index(config.FACE_COLUMN)
        for g, j, window, grid_mouth, grid_rule, grid_flags in ticks:
            if window is not None:
                prob = next(probs)
                ok = 1 - window[:, face].mean() <= config.MAX_MISSING and window[-1, face] > 0.5
                self._held = prob if ok else None
            state = apply_rules(self._hysteresis.step(self._held if j >= 0 else None), grid_flags)
            before = self.stage
            self.stage = self._timer.update(g, state == config.CLOSED, unknown=state < 0 or state == config.HEAD)
            self.state, self.closed_sec = state, self._timer.elapsed
            # 화면: 0.5초보다 짧은 눈감김(깜빡임)과 입을 다문 지 0.5초가 지난 하품은 정상으로 보인다. 경고 단계는 그대로다
            yawned = yawn_classes(np.array([state]), np.array([grid_mouth]), np.array([grid_rule]), self._post.yawn_mode)
            self.shown = int(display_classes(yawned, np.array([self.closed_sec]), mouth_closed=np.array([grid_mouth]))[0])
            self.history.append((g, state, self.stage, self.shown, self.closed_sec))
            n_events = len(self.events)
            if self.stage > before:
                self.events.append(Event(g, "졸음" if self.stage == DROWSY else "주의"))
            if self.yawn_repeat.update(g, self.shown == config.YAWN):
                self.events.append(Event(g, "하품 주의"))
            # 확인 필요: 판정 불가·고개 숙임이 3초 이어지고 그 사이 눈 감김을 이어 세지 않을 때(기술서 3.3). 평가 지표에는 넣지 않는다
            if j >= 0 and self.unknown_watch.update(g, (state < 0 or state == config.HEAD) and self.closed_sec == 0):
                self.events.append(Event(g, "확인 필요"))
            prob = self._held if j >= 0 and self._held is not None else np.full(config.N_CLASSES, np.nan)
            self.log.append(dict(zip(LOG_COLUMNS, [
                datetime.now().isoformat(timespec="milliseconds"), round(g, 3), STAGE_NAMES[self.stage], round(self.closed_sec, 3),
                *[round(float(p), 4) for p in prob], config.CLASSES[state] if state >= 0 else "판정 불가",
                ";".join(e.kind for e in self.events[n_events:])])))


class LiveRule:
    """규칙 EAR 0.48. 평소 EAR은 지금까지 들어온 검출 프레임의 90% 분위다."""

    def __init__(self) -> None:
        self._ears: list[float] = []
        self._t0: float | None = None
        self._baseline, self._baseline_t = np.nan, -np.inf
        self._timer = ClosureTimer()
        self.ready, self.stage = False, NORMAL
        self.history: list[tuple[float, int]] = []

    def update(self, t: float, found: bool, ear: float, head_down: bool = False) -> None:
        """판정 불가(얼굴이 안 보임)와 고개 숙임(LiveJudge.head_down, 정면 기준 상대 pitch)은 직전 눈 감김이 0.5초 이상이면 이어서 센다
        (최종 기획서 5-1·5-3)."""
        if self._t0 is None:
            self._t0 = t
        if found and not np.isnan(ear):
            self._ears.append(ear)
        if t - self._baseline_t >= BASELINE_EVERY_SEC and self._ears:
            self._baseline, self._baseline_t = float(np.quantile(self._ears, BASELINE_EAR_QUANTILE)), t
        self.ready = t - self._t0 >= RULE_MIN_SEC
        closed = self.ready and found and not np.isnan(ear) and ear < EAR_RATIO * self._baseline
        self.stage = self._timer.update(t, closed, unknown=(not found) or head_down)
        self.history.append((t, self.stage))


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
    board = panel(judge.shown, judge.stage, judge.closed_sec, rule.stage, ear, mar, scale)
    blend(frame, board, margin, margin)
    note = f"처리 {rate:.1f} FPS" + ("" if rule.ready else " / 규칙 준비 중") + " / q 종료"
    top = margin + board.shape[0] + round(6 * scale)
    blend(frame, info_box(note, scale), margin, top)
    alerts = [text for text, on in (("하품 주의 (5분 3회)", judge.yawn_repeat.active(judge.history[-1][0] if judge.history else 0.0)),
                                    ("확인 필요 (판정 불가·고개 숙임 3초)", judge.unknown_watch.active)) if on]
    for i, text in enumerate(alerts):
        blend(frame, info_box(text, scale * 1.2), margin, top + round((i + 1) * 40 * scale))
    if judge.stage == DROWSY:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), STAGE_RGB[DROWSY][::-1], max(round(12 * scale), 2))


def save_log(rows: list[dict], path: Path) -> None:
    """판정 로그(LiveJudge.log) → CSV. 얼굴·영상은 없고 시각·단계·확률만 있다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


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
    capture = cv2.VideoCapture(int(source), getattr(cv2, f"CAP_{config.WEBCAM_BACKEND}")) if live else cv2.VideoCapture(source)
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


def apply_settings(post: Post, grace: float | None = None, head_rel: float | None = None, backend: str | None = None) -> Post:
    """실행 옵션(--grace·--head-rel·--backend)을 이번 실행에만 적용한다. post.json과 config.py 파일은 바꾸지 않는다.
    11.05 학원 검증의 설정 A(합의값)·B(grace 0.2, 상대 10도)를 명령만으로 바꾸기 위해 둔다."""
    from dataclasses import replace

    if head_rel is not None:
        config.HEAD_PITCH_DOWN_REL = head_rel
    if backend is not None:
        config.WEBCAM_BACKEND = backend
    return post if grace is None else replace(post, grace=grace)


def run(source: str, weights: Path, show: bool, sound: bool, max_seconds: float | None, record: Path | None = None,
        grace: float | None = None) -> dict:
    from landmarks.extract import LandmarkExtractor
    from eval.evaluate import load_predictor

    capture, live, file_fps = open_source(source)
    writer = TimedWriter(record) if record else None
    predictor = load_predictor(weights)
    post = apply_settings(for_weights(weights, predictor.name), grace=grace)
    setting = f"후처리 {post.label()}, 고개 숙임 상대 {config.HEAD_PITCH_DOWN_REL:g}도, 웹캠 {config.WEBCAM_BACKEND}"
    print(f"모델 {weights}, {setting}", flush=True)
    judge, rule, mouth_watch, yawn_tracker = LiveJudge(predictor, post), LiveRule(), MouthClosedWatch(), YawnTracker()
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
            judge.update(t, vector, mouth_closed, yawn_tracker.update(t, mar))
            rule.update(t, found, ear, judge.head_down)
            frames += 1
            stamps.append(time.perf_counter())
            stamps = stamps[-60:]
            rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0
            if beeper:
                beeper.set(judge.stage == DROWSY or judge.unknown_watch.active)
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
    where = ("웹캠" if live else Path(source).name) + (f", 녹화 {record.name}" if record else "") + f", {Path(weights).parent.name} {setting}"
    summary = summary_markdown(judge.events, duration, f"운전 후 요약 ({time.strftime('%Y-%m-%d %H:%M')}, {where})")
    if live:     # 웹캠 세션만 저장한다. 영상·얼굴은 없고 이벤트 시각과 판정 로그만 있다
        stem = config.RUNS / "sessions" / time.strftime('%Y%m%d_%H%M%S')
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".md").write_text(summary, encoding="utf-8")
        save_log(judge.log, stem.with_name(stem.name + "_log.csv"))
    return {"frames": frames, "seconds": spent, "fps": frames / spent if spent else 0.0, "summary": summary,
            "events": judge.events,
            "ticks": len(judged), "judged": float(np.mean(np.array(judged) >= 0)) if judged else 0.0,
            "history": judge.history,
            "starts": {name: starts(judge.history, value, column=3) for name, value in
                       (("눈감김", config.CLOSED), ("하품", config.YAWN), ("고개숙임", config.HEAD))},
            "drowsy": starts(judge.history, DROWSY, column=2), "rule_drowsy": starts(rule.history, DROWSY)}


def main() -> None:
    parser = argparse.ArgumentParser(description="웹캠이나 영상 파일을 v3 모델로 실시간 판정한다")
    parser.add_argument("--camera", default="0", help="웹캠 번호 (백엔드는 config.WEBCAM_BACKEND)")
    parser.add_argument("--input", help="웹캠 대신 영상 파일. 파일 판정과 같은지 확인할 때 쓴다")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--no-window", action="store_true", help="화면 없이 처리한다")
    parser.add_argument("--mute", action="store_true", help="경고음을 끈다")
    parser.add_argument("--max-seconds", type=float, help="입력 시각으로 이 시간까지만 처리한다")
    parser.add_argument("--record", choices=("webcam", "test2"),
                        help="영상을 이 PC에 저장한다. webcam = 판정 화면을 보며 녹화(Val2), test2 = 판정 없이 녹화(Test2)")
    parser.add_argument("--grace", type=float, help="눈 감김 짧은 끊김 허용(초). 이번 실행에만 post.json 값을 덮어쓴다 (설정 B: 0.2)")
    parser.add_argument("--head-rel", type=float, help="고개 숙임 상대 pitch 기준(도). 이번 실행에만 config 값을 덮어쓴다 (설정 B: 10)")
    parser.add_argument("--backend", choices=("MSMF", "DSHOW"), help="웹캠 백엔드. 기본 config.WEBCAM_BACKEND")
    args = parser.parse_args()
    apply_settings(NONE, head_rel=args.head_rel, backend=args.backend)
    source = args.input or args.camera
    if args.record == "test2":
        result = record_blind(source, record_path("test2"), not args.no_window, args.max_seconds)
        print(f"Test2 녹화: {result['saved']} ({result['saved_frames']}프레임, {result['saved_frames'] / config.RECORD_FPS:.1f}초). 판정 결과는 만들지 않았다")
        return
    record = record_path("webcam") if args.record == "webcam" else None
    result = run(source, args.weights, not args.no_window, not args.mute, args.max_seconds, record, grace=args.grace)
    if record:
        print(f"Val2 녹화: {record}")
    print(f"프레임 {result['frames']}개, {result['seconds']:.1f}초, 평균 처리 {result['fps']:.1f} FPS, "
          f"15 FPS 격자 {result['ticks']}칸 중 판정 {result['judged']:.1%}")
    for name, found in result["starts"].items():
        print(f"  모델 {name} 시작 시각: {found}")
    print(f"  모델 졸음 경고: {result['drowsy']}, 규칙 EAR 0.48 졸음 경고: {result['rule_drowsy']}")
    print(result["summary"])


if __name__ == "__main__":
    main()

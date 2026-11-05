# 영상 파일 → 최종 구성(규칙 눈 감김, 하품 전용 모델) 또는 4클래스 판정, 눈감김 타이머, 경고, 현재 EAR·MAR을 겹쳐 그린 결과 영상. 프레임마다 그 시각까지 끝난 판정만 그린다.
# 결과 영상에는 얼굴이 나오므로 이 PC에만 둔다.
#
# 실행: python -m demo.demo_video --input 영상.mp4 --weights runs/v1/cnn1d_v01.pt [--features 특징.parquet] [--out 결과.mp4] [--yawn-weights runs/yawn1/cnn1d_v01.pt] [--mask]

import argparse
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config
from sequence_model.models import Predictor
from sequence_model.postprocess import NONE, Post, display_classes, for_weights, yawn_classes
from sequence_model.train_yawn import YawnPredictor, load_yawn
from rule_based.alarm import DROWSY, WARN, ClosureTimer, stage_starts, timer_stages
from rule_based.alerts import YawnRepeat
from rule_based.common import runs
from eval.evaluate import ear48_closed, grid_mouth_closed, grid_yawning, model_frames, model_probs, model_timer_unknown, rule_timer_inputs

FONT = Path("C:/Windows/Fonts/malgun.ttf")
CLASS_RGB = {-1: (110, 110, 110), config.NORMAL: (46, 160, 67), config.CLOSED: (218, 54, 51), config.YAWN: (156, 98, 222), config.HEAD: (230, 140, 30)}
STAGE_RGB = ((46, 160, 67), (224, 165, 30), (218, 54, 51))
STAGE_NAMES = ("정상", "주의", "졸음")


def elapsed(times: np.ndarray, closed: np.ndarray, grace: float = 0.0, unknown: np.ndarray | None = None,
            carry_max: float | None = None) -> np.ndarray:
    """프레임별 감음(과 판정 불가) → 연속 감은 시간(초). ClosureTimer와 같은 정의다."""
    timer, out = ClosureTimer(grace_sec=grace, carry_max_sec=carry_max), np.zeros(len(times))
    unknown = np.zeros(len(times), dtype=bool) if unknown is None else unknown
    for i, (t, c, u) in enumerate(zip(times, closed, unknown)):
        timer.update(float(t), bool(c), bool(u))
        out[i] = timer.elapsed
    return out


def yawn_flags(df: pd.DataFrame, predictor: YawnPredictor) -> np.ndarray:
    """특징 표 → 15 FPS 격자별 하품 여부. 켜짐이 YAWN_MODEL_MIN_SEC 이어진 뒤부터 참이다 (demo.webcam.LiveJudge와 같다)."""
    times, probs, judged = model_probs(df, predictor)
    on = (probs[:, 0] >= config.YAWN_ENTER) & judged
    out = np.zeros(len(on), dtype=bool)
    for start, end in zip(*runs(on)):
        out[start:end + 1] = times[start:end + 1] - times[start] >= config.YAWN_MODEL_MIN_SEC - 1e-9
    return out


def yawn_alerts(times: np.ndarray, yawning: np.ndarray) -> np.ndarray:
    """격자별 하품 여부 → 하품 주의를 띄우는 격자."""
    repeat, out = YawnRepeat(0.0), np.zeros(len(times), dtype=bool)
    for i, (t, on) in enumerate(zip(times, yawning)):
        repeat.update(float(t), bool(on))
        out[i] = repeat.active(float(t))
    return out


def at(times: np.ndarray, values: np.ndarray, t: float, default):
    i = int(np.searchsorted(times, t, side="right")) - 1
    return default if i < 0 else values[i]


@lru_cache(maxsize=8)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def panel(state: int, stage: int, seconds: float, rule_stage: int, ear: float, mar: float, scale: float) -> np.ndarray:
    """왼쪽 위 상태판 → RGBA."""
    w, h = round(440 * scale), round(236 * scale)
    image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=round(12 * scale), fill=(16, 16, 16, 195))
    small, large = font(max(round(20 * scale), 10)), font(max(round(28 * scale), 12))
    x = round(18 * scale)

    def badge(top: int, text: str, rgb) -> None:
        left = w - round(150 * scale)
        draw.rounded_rectangle((left, top, w - round(16 * scale), top + round(40 * scale)), radius=round(8 * scale), fill=(*rgb, 255))
        draw.text(((left + w - round(16 * scale)) / 2, top + round(20 * scale)), text, font=large, fill="white", anchor="mm")

    y = round(12 * scale)
    draw.text((x, y + round(8 * scale)), "모델 판정", font=small, fill=(210, 210, 210))
    badge(y, "준비 중" if state < 0 else config.CLASSES[state], CLASS_RGB[state])
    y += round(52 * scale)
    draw.text((x, y + round(8 * scale)), f"눈 감은 시간 {seconds:.1f}초", font=small, fill="white")
    badge(y, STAGE_NAMES[stage], STAGE_RGB[stage])
    y += round(52 * scale)
    draw.text((x, y + round(8 * scale)), "규칙 경고", font=small, fill=(210, 210, 210))
    badge(y, STAGE_NAMES[rule_stage], STAGE_RGB[rule_stage])
    y += round(56 * scale)
    text = "얼굴 미검출" if np.isnan(ear) else f"현재 EAR {ear:.3f}   MAR {mar:.3f}"
    draw.text((x, y), text, font=small, fill=(255, 200, 120) if np.isnan(ear) else "white")
    return np.asarray(image)


def blend(frame: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    h, w = min(overlay.shape[0], frame.shape[0] - y), min(overlay.shape[1], frame.shape[1] - x)
    part = overlay[:h, :w]
    alpha = part[..., 3:] / 255.0
    area = frame[y:y + h, x:x + w]
    area[:] = (area * (1 - alpha) + part[..., 2::-1] * alpha).astype(np.uint8)


def strip(times: np.ndarray, values: np.ndarray, colors: dict, width: int, height: int, duration: float, default) -> np.ndarray:
    """시간 띠 → BGR. x축은 영상 시각 / 영상 길이다."""
    picks = (np.arange(width) + 1) / width * duration
    index = np.searchsorted(times, picks, side="right") - 1
    out = np.full((height, width, 3), 60, dtype=np.uint8)
    for value, rgb in colors.items():
        chosen = np.where(index >= 0, values[np.clip(index, 0, None)], default) == value
        out[:, chosen] = rgb[::-1]
    return out


def render(video: Path, df: pd.DataFrame, predictor: Predictor, out: Path, post: Post = NONE,
           yawn_predictor: YawnPredictor | None = None, mask: bool = False) -> dict:
    times, classes = model_frames(df, predictor, post)
    # 타이머 입력은 파일 평가와 같다: 판정 불가·고개 숙임은 직전 눈 감김이 0.5초 이상이면 이어서 센다 (최종 기획서 5-1·5-3)
    closed, unknown = classes == config.CLOSED, model_timer_unknown(classes)
    stages = timer_stages(times, closed, grace_sec=post.grace, unknown=unknown, carry_max_sec=post.carry_max)
    seconds = elapsed(times, closed, post.grace, unknown, post.carry_max)
    # 화면에는 0.5초보다 짧은 눈감김(깜빡임)과 입을 다문 지 0.5초가 지난 하품을 정상으로 보인다. 경고 단계는 위에서 이미 구했다
    frame_t = df["t"].to_numpy(dtype=float)
    fps_src = 1 / float(np.median(np.diff(frame_t)))
    rule_closed, rule_inputs = ear48_closed(df, fps_src), rule_timer_inputs(df)
    rule = timer_stages(frame_t, rule_closed, **rule_inputs)
    stage_t = times
    if yawn_predictor is None:
        mouth = grid_mouth_closed(df)
        classes = display_classes(yawn_classes(classes, mouth, grid_yawning(df), post.yawn_mode), seconds, mouth_closed=mouth)
        if mask:
            classes[classes == config.YAWN] = config.NORMAL
    else:       # 최종 구성: 경고는 규칙 눈 감김 단계와 하품 알림 중 높은 쪽 (기술서 3.2)
        yawning = yawn_flags(df, yawn_predictor) & (not mask)
        classes = display_classes(np.where(classes == config.YAWN, config.NORMAL, classes), seconds)
        classes[(classes == config.NORMAL) & yawning] = config.YAWN
        alerts = yawn_alerts(times, yawning)[np.clip(np.searchsorted(times, frame_t, side="right") - 1, 0, None)]
        stage_t, stages = frame_t, np.where(alerts, np.maximum(rule, WARN), rule)
        seconds = elapsed(frame_t, rule_closed, unknown=rule_inputs["unknown"])
    ear, mar = df["ear"].to_numpy(dtype=float), df["mar"].to_numpy(dtype=float)

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = len(df) / fps_src
    scale = max(width / 1280, 0.5)
    row = max(round(20 * scale), 8)
    strips = [strip(times, classes, CLASS_RGB, width, row, duration, -1), strip(stage_t, stages, dict(enumerate(STAGE_RGB)), width, row, duration, 0)]
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for backend, code in ((cv2.CAP_MSMF, "H264"), (cv2.CAP_FFMPEG, "mp4v")):
        writer = cv2.VideoWriter(str(out), backend, cv2.VideoWriter_fourcc(*code), fps, (width, height))
        if writer.isOpened():
            break
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = (index + 1) / fps
            i = min(int(np.searchsorted(frame_t, t, side="right")) - 1, len(df) - 1)
            stage = int(at(stage_t, stages, t, 0))
            blend(frame, panel(int(at(times, classes, t, -1)), stage, float(at(stage_t, seconds, t, 0.0)), int(at(frame_t, rule, t, 0)),
                               float(ear[max(i, 0)]), float(mar[max(i, 0)]), scale), round(16 * scale), round(16 * scale))
            if stage == DROWSY:
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), STAGE_RGB[DROWSY][::-1], max(round(12 * scale), 2))
            done = min(round(t / duration * width), width)
            for k, image in enumerate(strips):
                top = height - (len(strips) - k) * (row + 2)
                frame[top:top + row] = (frame[top:top + row] * 0.35).astype(np.uint8)
                frame[top:top + row, :done] = image[:, :done]
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()
    found = {name: np.round(times[(classes == value) & ~np.r_[False, (classes == value)[:-1]]], 2).tolist() for name, value in
             (("눈감김", config.CLOSED), ("하품", config.YAWN), ("고개숙임", config.HEAD))}
    return {"frames": index, "starts": found, "drowsy": np.round(stage_starts(stage_t, stages, DROWSY), 2).tolist(),
            "rule_drowsy": np.round(stage_starts(frame_t, rule, DROWSY), 2).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="영상 파일로 판정 데모를 만든다")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="특징 parquet. 없으면 data/features/<이름>.parquet")
    parser.add_argument("--out", type=Path, help="결과 영상. 없으면 runs/demo/<이름>_v3.mp4")
    parser.add_argument("--yawn-weights", type=Path, default=config.YAWN_WEIGHTS, help="하품 전용 모델 가중치. 파일이 없으면 4클래스 판정만 쓴다")
    parser.add_argument("--mask", action="store_true", help="마스크 착용: 하품 표시와 하품 알림을 끈다")
    args = parser.parse_args()
    df = pd.read_parquet(args.features or config.FEATURE_DIR / f"{args.input.stem}.parquet")
    out = args.out or config.RUNS / "demo" / f"{args.input.stem}_v3.mp4"
    predictor = Predictor(args.weights)
    post = for_weights(args.weights, predictor.name)
    yawn_predictor = load_yawn(args.yawn_weights)
    result = render(args.input, df, predictor, out, post, yawn_predictor, args.mask)
    print(f"저장: {out}, {result['frames']}프레임, 후처리 {post.label()}, 하품 전용 {args.yawn_weights if yawn_predictor else '없음(4클래스 판정)'}")
    for name, starts in result["starts"].items():
        print(f"  모델 {name} 시작 시각: {starts}")
    print(f"  {'최종' if yawn_predictor else '모델'} 졸음 경고: {result['drowsy']}, 규칙 졸음 경고: {result['rule_drowsy']}")


if __name__ == "__main__":
    main()

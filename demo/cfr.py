# 실제 시각에 맞춘 일정 FPS 영상 쓰기. 웹캠 녹화와 가변 FPS(휴대폰) 영상 변환에 쓴다.
# 칸 k(시각 k / fps)에는 그 시각까지 들어온 가장 최근 프레임을 쓴다. 그래서 "프레임 번호 ÷ FPS" = 실제 시각(오차 1 / fps 이내)이고,
# 특징 추출·라벨 도구·채점이 지금처럼 프레임 번호로 시각을 계산해도 맞는다. 영상은 이 PC에만 둔다.
#
# 실행: python -m demo.cfr <가변 FPS 영상.mp4> <결과.mp4>      기록된 프레임 시각으로 일정 FPS 영상을 만든다

import argparse
from pathlib import Path

import cv2
import numpy as np

import config


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """브라우저에서 재생되는 H264(MSMF)를 먼저 시도하고, 안 되면 mp4v로 쓴다 (demo/demo_video.py와 같다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for backend, code in ((cv2.CAP_MSMF, "H264"), (cv2.CAP_FFMPEG, "mp4v")):
        writer = cv2.VideoWriter(str(path), backend, cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer
    raise SystemExit(f"영상을 쓸 수 없습니다: {path}")


class TimedWriter:
    """프레임과 실제 시각(초)을 받아 fps 칸마다 그 시각의 최신 프레임을 쓴다. 모자라면 앞 프레임을 반복하고 남으면 버린다."""

    def __init__(self, path: Path, fps: float = config.RECORD_FPS) -> None:
        self.path, self.fps = Path(path), fps
        self._writer: cv2.VideoWriter | None = None
        self._t0: float | None = None
        self._last: np.ndarray | None = None
        self._last_t = 0.0
        self.frames = 0

    def write(self, frame: np.ndarray, t: float) -> None:
        if self._writer is None:
            self._writer = open_writer(self.path, self.fps, (frame.shape[1], frame.shape[0]))
            self._t0 = t
        rel = t - self._t0
        while self._last is not None and self.frames / self.fps < rel - 1e-9:    # 이 프레임 전의 칸은 직전 프레임
            self._writer.write(self._last)
            self.frames += 1
        self._last, self._last_t = frame, rel

    def close(self) -> None:
        if self._writer is None:
            return
        while self._last is not None and self.frames / self.fps <= self._last_t + 1e-9:
            self._writer.write(self._last)
            self.frames += 1
        self._writer.release()
        self._writer = None


def convert(src: Path, dst: Path, fps: float = config.RECORD_FPS) -> dict:
    """가변 FPS 영상 → 기록된 프레임 시각(CAP_PROP_POS_MSEC)대로 일정 FPS 영상. (원본 프레임 수, 길이, 결과 프레임 수)."""
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {src}")
    writer = TimedWriter(dst, fps)
    count, last = 0, 0.0
    while capture.grab():
        last = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, frame = capture.retrieve()
        if not ok:
            break
        writer.write(frame, last)
        count += 1
    capture.release()
    writer.close()
    return {"source_frames": count, "seconds": last, "frames": writer.frames}


def main() -> None:
    parser = argparse.ArgumentParser(description="가변 FPS 영상을 실제 시각에 맞춘 일정 FPS 영상으로 바꾼다")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--fps", type=float, default=config.RECORD_FPS)
    args = parser.parse_args()
    result = convert(args.src, args.dst, args.fps)
    print(f"{args.dst}: 원본 {result['source_frames']}프레임 {result['seconds']:.2f}초 → {result['frames']}프레임 ({args.fps:g} FPS)")


if __name__ == "__main__":
    main()

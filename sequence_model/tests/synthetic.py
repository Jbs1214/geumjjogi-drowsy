# 테스트용 합성 라벨. 실제 데이터 없이 기획서 3-5의 자동 라벨 형식(X, y, segments, params)으로 파일을 만든다.
# 패턴이 뚜렷한 가짜 데이터라 여기서 나온 점수는 파이프라인 동작 확인일 뿐 성능 수치가 아니다.

import json
from pathlib import Path

import numpy as np

from sequence_model import config

NAMES = ("정상", "눈감김·고개숙임", "하품")


def clip_features(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """프레임별 상태 → 특징 (프레임, 11). 눈감김은 ear가, 하품은 mar가 달라진다."""
    n = len(states)
    closed, yawn = states == 1, states == 2
    ear = np.where(closed, 0.05, 0.30) + rng.normal(0, 0.01, n)
    mar = np.where(yawn, 0.60, 0.05) + rng.normal(0, 0.01, n)
    pitch, yaw, roll = rng.normal(0, 2, (3, n))
    blink = np.where(closed, 0.9, 0.1) + rng.normal(0, 0.02, n)
    jaw = np.where(yawn, 0.8, 0.05) + rng.normal(0, 0.02, n)
    return np.column_stack([np.ones(n), ear, ear, ear, mar, pitch, yaw, roll, blink, blink, jaw])


def clip_segments(rng: np.random.Generator, seconds: int) -> list[tuple[float, float, int]]:
    """1.5초 간격으로 눈감김·하품 구간을 번갈아 넣은 (시작, 끝, 상태) 목록. 사이는 정상이다."""
    segments = []
    t = 0.0
    while t < seconds:
        length = float(rng.choice([1.5, 2.0]))
        state = int(rng.integers(1, 3))
        segments += [(t, t + 1.5, 0), (t + 1.5, t + 1.5 + length, state)]
        t += 1.5 + length
    return segments


def frame_states(segments: list[tuple[float, float, int]], n_frames: int) -> np.ndarray:
    t = np.arange(n_frames) / config.FPS
    states = np.zeros(n_frames, dtype=int)
    for start, end, state in segments:
        states[(t >= start) & (t < end)] = state
    return states


def write_clips(label_dir: Path, reviewed_dir: Path, n_groups: int = 12, clips_per_group: int = 6,
                seconds: int = 12, seed: int = 0) -> list[str]:
    """그룹 × 클립만큼 자동 라벨과 검수 라벨 파일을 만들고 클립 번호 목록을 돌려준다."""
    label_dir.mkdir(parents=True, exist_ok=True)
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clip_ids = []
    for group in range(n_groups):
        for clip in range(clips_per_group):
            clip_id = f"SGA21{group:05d}S{clip:04d}"
            segments = clip_segments(rng, seconds)
            n = seconds * config.FPS
            states = frame_states(segments, n)
            auto = {
                "X": clip_features(states, rng).tolist(),
                "y": states.tolist(),
                "segments": [],
                "params": {},
            }
            reviewed = {"segments": [{"start": s, "end": e, "state": NAMES[state]} for s, e, state in segments]}
            (label_dir / f"{clip_id}_자동라벨.json").write_text(json.dumps(auto), encoding="utf-8")
            (reviewed_dir / f"{clip_id}.json").write_text(json.dumps(reviewed, ensure_ascii=False), encoding="utf-8")
            clip_ids.append(clip_id)
    return clip_ids

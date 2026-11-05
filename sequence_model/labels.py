# 자동·검수 라벨 JSON을 읽고, 클립을 SGA 그룹 단위로 학습·검증·평가에 나눈다 (기획서 3-3, 3-5).
# 라벨 JSON은 샘플을 아직 받지 못해 기획서의 필드 이름(X, y, segments)을 기준으로 읽는다.
# 실제 형식이 다르면 이 파일의 load_auto_label()·load_reviewed_segments()만 고친다.

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model import config

# 검수 라벨 상태 5종 → y 값. 손·팔 동작은 얼굴을 가려 판정할 수 없어 불확실로 둔다 (자동 라벨 일치도 계산과 같음)
STATE_BY_NAME = {
    "정상": 0,
    "눈감김·고개숙임": 1,
    "하품": 2,
    "손·팔 동작": config.UNCERTAIN,
    "불확실": config.UNCERTAIN,
}


@dataclass
class ClipLabel:
    clip_id: str
    X: np.ndarray       # (프레임, 특징 11)
    y: np.ndarray       # (프레임,) 상태 0~3
    fps: float

    def to_frame(self) -> pd.DataFrame:
        """프레임별 표. 열: t, 특징 11개, state. 시각은 프레임 번호 / fps (기획서 3-5)."""
        df = pd.DataFrame(self.X, columns=list(config.FEATURE_COLUMNS))
        df.insert(0, "t", np.arange(len(df)) / self.fps)
        df["state"] = self.y
        return df


def clip_id_of(path: Path) -> str:
    """라벨 파일 경로 → 클립 번호. '<클립>_자동라벨.json'과 '<클립>.json'을 모두 받는다."""
    return path.stem.split("_")[0]


def clip_group(clip_id: str) -> str:
    """클립 번호 → 분할 그룹. 탑승자 식별자가 없어 SGA 번호를 쓴다 (SGA2100002S0124 → SGA2100002)."""
    match = re.match(r"SGA\d+(?=S\d+$)", clip_id)
    return match[0] if match else clip_id


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_auto_label(path: Path) -> ClipLabel:
    """자동 라벨 JSON → ClipLabel. X는 null(얼굴 미검출)을 NaN으로 읽는다."""
    data = _read_json(path)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=int)
    if X.ndim != 2 or X.shape[1] != len(config.FEATURE_COLUMNS) or len(X) != len(y):
        raise ValueError(f"{path}: X {X.shape}, y {y.shape}는 (프레임, {len(config.FEATURE_COLUMNS)})과 (프레임,)이어야 합니다")
    return ClipLabel(clip_id_of(path), X, y, float(data.get("fps", config.FPS)))


def load_reviewed_segments(path: Path) -> list[tuple[float, float, int]]:
    """검수 라벨 JSON의 segments → (시작 초, 끝 초, 상태 값) 목록."""
    return [(float(s["start"]), float(s["end"]), STATE_BY_NAME[s["state"]]) for s in _read_json(path)["segments"]]


def segments_to_frames(segments: list[tuple[float, float, int]], n_frames: int, fps: float) -> np.ndarray:
    """구간 목록 → 프레임별 상태. 어느 구간에도 없는 프레임은 라벨이 없으므로 불확실이다."""
    t = np.arange(n_frames) / fps
    states = np.full(n_frames, config.UNCERTAIN, dtype=int)
    for start, end, state in segments:
        states[(t >= start) & (t < end)] = state
    return states


def split_clips(clip_ids: list[str], calibration_clips: tuple[str, ...] = (),
                seed: int = config.SPLIT_SEED) -> dict[str, str]:
    """클립 번호 → 'train' / 'val' / 'test'. 같은 그룹은 한 분할에만 들어간다.

    임계값 보정에 쓴 클립의 그룹은 평가에서 빠지도록 train에 넣는다 (기획서 9절).
    그룹 크기가 치우쳐 있어(1~327클립) 무작위 배정을 SPLIT_TRIALS번 해 보고 목표 비율에 가장 가까운 것을 고른다.
    오차는 분할별 상대 오차의 합이라 빈 분할은 크게 불리하다.
    """
    sizes = pd.Series([clip_group(c) for c in clip_ids], dtype=object).value_counts().sort_index()
    calibration_groups = {clip_group(c) for c in calibration_clips}
    free = [group for group in sizes.index if group not in calibration_groups]
    free_sizes = sizes[free].to_numpy()

    names = list(config.SPLIT_RATIOS)
    ratios = np.array(list(config.SPLIT_RATIOS.values()))
    fixed = np.zeros(len(names))
    fixed[names.index("train")] = len(clip_ids) - free_sizes.sum()

    rng = np.random.default_rng(seed)
    best_choice, best_error = None, np.inf
    for _ in range(config.SPLIT_TRIALS):
        choice = rng.choice(len(names), size=len(free), p=ratios)
        counts = fixed + np.bincount(choice, weights=free_sizes, minlength=len(names))
        error = np.abs(counts / (ratios * len(clip_ids)) - 1).sum()
        if error < best_error:
            best_choice, best_error = choice, error

    assigned = {group: names[i] for group, i in zip(free, best_choice)}
    assigned.update(dict.fromkeys(calibration_groups, "train"))
    return {clip: assigned[clip_group(clip)] for clip in clip_ids}

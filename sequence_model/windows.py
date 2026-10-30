# 특징 시계열을 일정한 FPS로 맞추고 고정 길이 창으로 잘라 학습 데이터를 만든다 (기획서 3-3).
# 실시간 데모도 같은 resample()·fill_missing()·make_windows()를 쓴다.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model import config
from sequence_model.labels import clip_group, clip_id_of, load_auto_label, split_clips


@dataclass
class WindowSet:
    X: np.ndarray           # (N, 창 프레임, 특징 11)
    y: np.ndarray           # (N,) 클래스 0~2
    info: pd.DataFrame      # 창별 정보. 열: clip_id, group, split, label, start_sec

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        keep = (self.info["split"] == name).to_numpy()
        return self.X[keep], self.y[keep]


def window_frames(sec: float = config.WINDOW_SEC, fps: float = config.FPS) -> int:
    """초 → 프레임 수."""
    return round(sec * fps)


def resample(df: pd.DataFrame, fps: float = config.FPS) -> pd.DataFrame:
    """영상 시각(t, 초) 기준으로 간격을 fps에 맞춘다. 칸마다 가장 가까운 원본 프레임을 쓴다.

    보간하면 없던 반쯤 감은 눈이 생기고 상태·얼굴 검출 값이 소수가 되므로 원본 값을 그대로 가져온다.
    """
    t = df["t"].to_numpy(dtype=float)
    if len(t) < 2:
        return df.reset_index(drop=True)
    grid = t[0] + np.arange(round((t[-1] - t[0] + np.median(np.diff(t))) * fps)) / fps
    right = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    left = right - 1
    out = df.iloc[np.where(grid - t[left] <= t[right] - grid, left, right)].reset_index(drop=True)
    out["t"] = grid
    return out


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """얼굴 미검출 프레임의 특징을 0으로 채우고 face 열로 표시한다 (기획서 3-3).

    미검출인데 정상인 프레임은 불확실로 바꾼다. 미검출이어도 눈감김으로 표시된 프레임(lost_close)은 그대로 둔다.
    """
    out = df.copy()
    features = [c for c in config.FEATURE_COLUMNS if c != config.FACE_COLUMN]
    missing = out[features].isna().any(axis=1) | ~(out[config.FACE_COLUMN] > 0.5)
    out.loc[missing, features] = 0.0
    out[config.FACE_COLUMN] = (~missing).astype(float)
    if "state" in out:
        out.loc[missing & (out["state"] == 0), "state"] = config.UNCERTAIN
    return out


def window_label(states: np.ndarray) -> int:
    """창 안 프레임별 상태 → 창의 상태. 눈감김, 하품, 불확실 순으로 LABEL_RATIO 이상인 것, 없으면 정상."""
    for state in (1, 2, config.UNCERTAIN):
        if np.count_nonzero(states == state) >= config.LABEL_RATIO * len(states):
            return state
    return 0


def make_windows(df: pd.DataFrame, window_sec: float = config.WINDOW_SEC, stride_sec: float = config.STRIDE_SEC,
                 fps: float = config.FPS, label_col: str = "state") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """resample·fill_missing을 거친 표 → (창 (N, 창 프레임, 특징 수), 창 상태 (N,), 창 시작 초 (N,)).

    창보다 짧은 클립은 빈 배열을 돌려준다. 불확실 창도 그대로 내고 쓰는 쪽에서 뺀다.
    """
    size = window_frames(window_sec, fps)
    starts = np.arange(0, len(df) - size + 1, window_frames(stride_sec, fps))
    features = df[list(config.FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    states = df[label_col].to_numpy(dtype=int)
    X = np.empty((0, size, features.shape[1]), dtype=np.float32)
    if len(starts):
        X = np.stack([features[s:s + size] for s in starts])
    return X, np.array([window_label(states[s:s + size]) for s in starts], dtype=int), starts / fps


def build_dataset(label_dir: Path, calibration_clips: tuple[str, ...] = (),
                  window_sec: float = config.WINDOW_SEC) -> WindowSet:
    """자동 라벨 폴더 → 불확실 창을 뺀 학습 데이터. 분할은 클립 단위라 창이 겹쳐도 분할 사이에 새지 않는다."""
    paths = sorted(label_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"자동 라벨 JSON이 없습니다: {label_dir}")
    splits = split_clips([clip_id_of(p) for p in paths], calibration_clips)

    Xs, ys, rows = [], [], []
    short = uncertain = 0
    for path in paths:
        df = fill_missing(resample(load_auto_label(path).to_frame()))
        X, y, starts = make_windows(df, window_sec)
        short += not len(X)
        uncertain += int(np.count_nonzero(y == config.UNCERTAIN))
        keep = y != config.UNCERTAIN
        clip_id = clip_id_of(path)
        Xs.append(X[keep])
        ys.append(y[keep])
        rows += [(clip_id, clip_group(clip_id), splits[clip_id], label, start)
                 for label, start in zip(y[keep], starts[keep])]

    info = pd.DataFrame(rows, columns=["clip_id", "group", "split", "label", "start_sec"])
    print(f"클립 {len(paths)}개, 창 {len(info)}개. 짧은 클립 제외 {short}개, 불확실 창 제외 {uncertain}개")
    print(info.groupby(["split", "label"]).size().to_string())
    return WindowSet(np.concatenate(Xs), np.concatenate(ys), info)


def fit_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train 창 → 특징별 (평균, 표준편차). 0으로 채운 미검출 프레임이 값을 끌어내리지 않게 검출 프레임만 쓴다."""
    frames = X.reshape(-1, X.shape[-1])
    face = config.FEATURE_COLUMNS.index(config.FACE_COLUMN)
    detected = frames[frames[:, face] > 0.5]
    mean, std = detected.mean(axis=0), detected.std(axis=0)
    mean[face], std[face] = 0.0, 1.0
    return mean, np.where(std < 1e-6, 1.0, std)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """창 (…, 프레임, 특징 수)을 정규화한다. 미검출 프레임은 평균값인 0으로 둔다."""
    out = (X - mean) / std
    face = config.FEATURE_COLUMNS.index(config.FACE_COLUMN)
    out[X[..., face] <= 0.5] = 0.0
    return out.astype(np.float32)

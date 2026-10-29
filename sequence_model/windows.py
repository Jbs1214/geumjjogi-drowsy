# 특징 시계열 표를 15 FPS로 맞추고 10초(150프레임) 창으로 잘라 학습 데이터를 만든다 (기획서 3-3).
# 실시간 데모도 같은 resample()·fill_missing()을 쓴다.

from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model import config


def resample(df: pd.DataFrame, fps: int = config.FPS) -> pd.DataFrame:
    """영상 시각(t, 초) 기준으로 프레임 간격을 fps에 맞춘다."""
    raise NotImplementedError


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """얼굴 미검출 구간 중 config.FILL_MAX_SEC 이하로 짧은 것만 앞뒤 값으로 채운다."""
    raise NotImplementedError


def make_windows(df: pd.DataFrame, stride_sec: int = config.STRIDE_SEC) -> tuple[np.ndarray, np.ndarray]:
    """특징 표 → 창 배열 (N, WINDOW_FRAMES, 특징 수)과 길이 마스크 (N, WINDOW_FRAMES).

    클립이 10초보다 짧으면 클립 길이에 맞추고 나머지는 패딩한 뒤 마스크로 표시한다.
    미검출 비율이 config.DROP_MISSING_RATIO 이상인 창은 빼고, 뺀 비율을 기록한다.
    """
    raise NotImplementedError


def build_dataset(feature_dir: Path, label_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """특징 파일 폴더 + 라벨 폴더 → (창, 마스크, 창별 정보 표). 창은 장면 라벨을 물려받는다.

    창별 정보 열: scene_id, occupant_id, split, label, start_sec.
    """
    raise NotImplementedError

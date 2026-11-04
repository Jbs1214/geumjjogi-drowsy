# 좌표 표 → EAR·MAR 열 추가 (기획서 3-1 ④, 3-2). 고개 각도(pitch·yaw·roll)는 extract.py가 이미 넣는다.
# PERCLOS처럼 이동창으로 구하는 비율은 rule_based/에서 계산한다.

import numpy as np
import pandas as pd

# sequence_model 입력으로 쓰는 특징 열 (config.FEATURE_COLUMNS와 맞춘다)
FEATURE_COLUMNS = ("ear", "mar", "pitch", "yaw", "roll")


def _dist(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    return np.hypot(df[f"{a}_x"] - df[f"{b}_x"], df[f"{a}_y"] - df[f"{b}_y"])


def eye_aspect_ratio(df: pd.DataFrame, prefix: str) -> pd.Series:
    """눈 하나의 EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|). prefix는 'le' 또는 're'. Soukupová & Čech (2016)."""
    vertical = _dist(df, f"{prefix}2", f"{prefix}6") + _dist(df, f"{prefix}3", f"{prefix}5")
    return vertical / (2 * _dist(df, f"{prefix}1", f"{prefix}4"))


def mouth_aspect_ratio(df: pd.DataFrame) -> pd.Series:
    """MAR = 입술 세로 간격 / 입꼬리 가로 간격."""
    return _dist(df, "mouth_t", "mouth_b") / _dist(df, "mouth_l", "mouth_r")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """extract.py 표 → ear_l, ear_r, ear(좌우 평균), mar 열을 더한 새 표. 얼굴 미검출 행은 NaN."""
    out = df.copy()
    out["ear_l"] = eye_aspect_ratio(df, "le")
    out["ear_r"] = eye_aspect_ratio(df, "re")
    out["ear"] = (out["ear_l"] + out["ear_r"]) / 2
    out["mar"] = mouth_aspect_ratio(df)
    return out

# 학습한 가중치를 불러와 시간창의 클래스 확률을 낸다.
# 실시간 데모는 0.5초마다 최근 창을 resample()·fill_missing()으로 맞춰 predict_proba()에 넣는다.

from pathlib import Path

import numpy as np
import torch

from sequence_model.models import build_model
from sequence_model.windows import normalize, window_frames


class SeqPredictor:
    def __init__(self, path: Path, device: str = "cpu") -> None:
        """가중치 파일(path)과 함께 저장한 설정(모델 이름·창 길이·정규화 값)을 불러온다."""
        saved = torch.load(path, map_location=device)
        self.window_sec = saved["window_sec"]
        self.fps = saved["fps"]
        self.feature_columns = saved["feature_columns"]
        self._mean = saved["mean"].cpu().numpy()
        self._std = saved["std"].cpu().numpy()
        self._device = device
        self._model = build_model(saved["model"], saved["n_features"], saved["n_classes"]).to(device)
        self._model.load_state_dict(saved["state_dict"])
        self._model.eval()

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """창 (N, 창 프레임, 특징 수) → 클래스 확률 (N, 클래스 수). 특징은 fill_missing()을 거친 값이다."""
        size = window_frames(self.window_sec, self.fps)
        if windows.ndim != 3 or windows.shape[1:] != (size, len(self.feature_columns)):
            raise ValueError(f"창 모양 {windows.shape}: (N, {size}, {len(self.feature_columns)})이어야 합니다")
        x = torch.from_numpy(normalize(windows, self._mean, self._std)).to(self._device)
        with torch.no_grad():
            return torch.softmax(self._model(x), dim=1).cpu().numpy()

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        """창 하나 (창 프레임, 특징 수) → 클래스 확률 (클래스 수,)."""
        return self.predict_batch(window[np.newaxis])[0]

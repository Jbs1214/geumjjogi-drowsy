# 4클래스 softmax 분류기: 1D-CNN, LSTM (기획서 2번). 입력 (배치, 45, 11), 출력 클래스별 점수 (배치, 4).
# 눈감김·고개숙임 정답은 창 끝 프레임, 하품은 창 전체 비율이라, 창 전체 요약과 끝 0.5초 요약을 함께 쓴다.

import torch
from torch import nn

import config

MODEL_NAMES = ("cnn1d", "lstm")
TAIL_FRAMES = round(config.FPS * 0.5)     # 끝 0.5초 (8프레임)


class CNN1D(nn.Module):
    def __init__(self, n_features: int = len(config.FEATURES), n_classes: int = config.N_CLASSES, channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x.transpose(1, 2))
        return self.head(torch.cat([out.mean(dim=-1), out[..., -TAIL_FRAMES:].mean(dim=-1)], dim=1))


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int = len(config.FEATURES), n_classes: int = config.N_CLASSES, hidden: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(torch.cat([out.mean(dim=1), out[:, -TAIL_FRAMES:].mean(dim=1)], dim=1))


def build_model(name: str) -> nn.Module:
    if name == "cnn1d":
        return CNN1D()
    if name == "lstm":
        return LSTMClassifier()
    raise ValueError(f"모델 이름은 {MODEL_NAMES} 중 하나여야 합니다: {name}")


class Predictor:
    """가중치 파일 → 창 (N, 45, 11, 정규화 전) 클래스 확률. 평균·표준편차는 가중치에 함께 저장돼 있다."""

    def __init__(self, path, device: str = "cpu") -> None:
        saved = torch.load(path, map_location=device, weights_only=True)     # 텐서·숫자·문자·list·dict만 연다 (CWE-502)
        self.name, self.norm, self.device = saved["model"], saved["norm"], device
        self.relative = bool(self.norm.get("relative", False))     # 세션 기준 특징으로 학습한 가중치인지 (session.py, v7부터)
        self.model = build_model(saved["model"]).to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def predict(self, windows, batch: int = 4096):
        import numpy as np
        from sequence_model.build_windows import normalize

        out = []
        with torch.no_grad():
            for i in range(0, len(windows), batch):
                x = torch.from_numpy(normalize(windows[i:i + batch], self.norm)).to(self.device)
                out.append(torch.softmax(self.model(x), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, config.N_CLASSES))

# 시퀀스 분류기를 만든다. 샘플이 적어 과적합을 막으려고 층과 채널을 작게 둔다 (기획서 5-2).

import torch
from torch import nn

MODEL_NAMES = ("cnn1d", "lstm")


class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x.transpose(1, 2)).squeeze(-1))


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 창 정답이 창 안 상태 비율로 정해지므로 마지막 상태 대신 전체 출력의 평균을 쓴다
        output, _ = self.lstm(x)
        return self.head(output.mean(dim=1))


def build_model(name: str, n_features: int, n_classes: int) -> nn.Module:
    """모델 이름 → 분류기. 입력 (배치, 창 프레임, 특징 수), 출력 클래스별 점수 (배치, n_classes)."""
    if name == "cnn1d":
        return CNN1D(n_features, n_classes)
    if name == "lstm":
        return LSTMClassifier(n_features, n_classes)
    raise ValueError(f"모델 이름은 {MODEL_NAMES} 중 하나여야 합니다: {name}")

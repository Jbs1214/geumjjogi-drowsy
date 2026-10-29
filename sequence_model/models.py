# 시퀀스 분류기를 만든다. 학습할 모델은 아직 정하지 않았다 (논문 검토 후 결정).

from torch import nn


def build_model(name: str, n_features: int, n_classes: int) -> nn.Module:
    """모델 이름 → 분류기.

    입력 (배치, WINDOW_FRAMES, 특징 수)와 길이 마스크를 받아 클래스별 점수 (배치, n_classes)를 낸다.
    TODO: 후보 모델을 정한 뒤 이름별로 추가한다.
    """
    raise NotImplementedError

# 10초 창 데이터로 시퀀스 분류기를 학습하고 가중치를 저장한다.
# 실행: python -m sequence_model.train --model <이름>

import argparse

from torch import nn


def parse_args() -> argparse.Namespace:
    """학습 옵션. TODO: 모델이 정해지면 epoch·batch·학습률 등 기본값을 채운다."""
    parser = argparse.ArgumentParser(description="시퀀스 분류기 학습")
    parser.add_argument("--model", required=True, help="models.build_model()에 넘길 모델 이름")
    return parser.parse_args()


def train(model: nn.Module, data: dict, args: argparse.Namespace) -> dict:
    """train 분할로 학습하고 val 분할로 고른 최고 성능 결과를 돌려준다. 결과: 지표·임계치·가중치 경로."""
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

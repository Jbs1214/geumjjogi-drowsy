# AI Hub 라벨을 정상/주의/졸음 클래스로 대응시키고, 탑승자 단위로 학습·검증·평가를 나눈다 (기획서 3-5).

from pathlib import Path

import pandas as pd


def scene_label(actions: list[str]) -> int | None:
    """장면의 행동 라벨 목록 → 클래스 번호(config.CLASSES 순서). 학습에 쓰지 않는 장면은 None.

    - 졸음: 이상행동 '졸음운전'
    - 주의: 행동 라벨 중 하품·눈 깜빡임처럼 졸음 전조로 볼 수 있는 것 (config.N_CLASSES == 3일 때만)
    - 정상: TODO 정의 확정 필요. 이 데이터에는 '운전하다'만 있는 장면이 없다.
      후보안: 졸음 관련 행동 0개 + 키프레임 5장 중 '운전하다' 3장 이상
    - 졸음이 아닌 다른 이상행동(통화·휴대폰 조작 등): None (오경보 평가 전용)
    """
    raise NotImplementedError


def load_labels(label_dir: Path) -> pd.DataFrame:
    """라벨 JSON 폴더를 읽어 장면 단위 표로 만든다. 열: scene_id, video_id, occupant_id, actions, label."""
    raise NotImplementedError


def split_by_occupant(df: pd.DataFrame) -> dict[str, str]:
    """탑승자 ID → 'train' / 'val' / 'test'. 같은 사람의 장면이 학습과 평가에 섞이지 않게 나눈다."""
    raise NotImplementedError

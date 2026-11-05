# 미니 데이터 묶음 → data/labels, data/features, data/split.csv. 라벨은 내용을 바꾸지 않고 복사한다.
#   NITYMED: 검수 라벨 41개, 묶음의 특징 CSV, split_nitymed.csv의 분할
#   DMD: 변환 라벨 labels_v2 8개(close + closing), 묶음의 특징 JSON. 변환 규칙을 다시 적용하지 않는다
#   DMD 분할은 운전자 단위로 시드를 고정해 정한다. 신규 7명은 라벨이 없어 들어오지 않는다
#
# 실행: python -m dataset.import_mini

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config


def check_hashes(root: Path) -> int:
    """SHA256SUMS.txt의 파일이 모두 같은지 확인한다. 다르면 멈춘다."""
    lines = [line for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if line]
    for line in lines:
        digest, name = line.split(maxsplit=1)
        if hashlib.sha256((root / name.strip()).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"체크섬이 다릅니다: {name}")
    return len(lines)


def save_features(table: pd.DataFrame, clip: str) -> None:
    """특징 표 → data/features/<clip>.parquet. 미검출은 NaN이고 feature_matrix가 0으로 바꾼다."""
    table = table.astype(float)
    table.insert(1, "face_found", table.pop(config.FACE_COLUMN) > 0.5)
    table.to_parquet(config.FEATURE_DIR / f"{clip}.parquet", index=False)


def copy_label(source: Path, frames: int) -> dict:
    """라벨을 그대로 복사한다. 프레임 수가 특징과 다르면 멈춘다."""
    label = json.loads(source.read_text(encoding="utf-8"))
    if int(label["frames"]) != frames:
        raise SystemExit(f"{source.name}: 라벨 {label['frames']}프레임, 특징 {frames}프레임")
    shutil.copyfile(source, config.LABEL_DIR / source.name)
    return label


def import_nitymed() -> pd.DataFrame:
    split = pd.read_csv(config.MINI_NITYMED_LABELS / "split_nitymed.csv", encoding="utf-8-sig")
    rows = []
    for row in split.itertuples(index=False):
        table = pd.read_csv(config.MINI_BUNDLE / "features_nitymed" / f"{row.clip}.csv")
        label = copy_label(config.MINI_NITYMED_LABELS / "labels" / f"{row.clip}.json", len(table))
        if label["driver_id"] != row.driver_id:
            raise SystemExit(f"{row.clip}: 라벨과 분할표의 운전자가 다릅니다")
        save_features(table, row.clip)
        rows.append({"clip": row.clip, "source": "nitymed", "driver_id": row.driver_id, "split": row.split, "kind": row.kind})
    return pd.DataFrame(rows)


def dmd_splits(drivers: list[str], seed: int = config.SEEDS[0]) -> dict[str, str]:
    """DMD 운전자 → 분할. 정렬한 뒤 시드로 섞어 앞의 MINI_DMD_VAL_DRIVERS명을 val로 둔다."""
    order = np.random.default_rng(seed).permutation(sorted(drivers))
    return {str(d): "val" if i < config.MINI_DMD_VAL_DRIVERS else "train" for i, d in enumerate(order)}


def import_dmd() -> pd.DataFrame:
    paths = sorted((config.MINI_BUNDLE / "dmd" / "labels_v2").glob("*.json"))
    drivers = sorted({json.loads(p.read_text(encoding="utf-8"))["driver_id"] for p in paths})
    splits = dmd_splits(drivers)
    rows = []
    for path in paths:
        raw = json.loads((config.MINI_BUNDLE / "dmd" / "features" / f"{path.stem}_rgb_face.json").read_text(encoding="utf-8"))
        table = pd.DataFrame(raw["frames"], columns=raw["columns"]).drop(columns="frame").rename(columns={"t_s": "t"})
        label = copy_label(path, len(table))
        save_features(table, path.stem)
        rows.append({"clip": path.stem, "source": "dmd", "driver_id": label["driver_id"], "split": splits[label["driver_id"]], "kind": "dmd"})
    return pd.DataFrame(rows)


def main() -> None:
    if config.SPLIT.exists():
        raise SystemExit(f"{config.SPLIT}이 있습니다. 다시 만들면 뒤에 더한 DMD 운전자 분할이 사라지니 처음 받을 때만 실행하세요")
    print(f"체크섬 {check_hashes(config.MINI_BUNDLE)}개 일치")
    config.LABEL_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    split = pd.concat([import_nitymed(), import_dmd()], ignore_index=True)
    grouped = split.groupby("driver_id")
    if (grouped["split"].nunique() > 1).any() or (grouped["source"].nunique() > 1).any():
        raise SystemExit("운전자가 분할이나 출처를 넘습니다")
    split.to_csv(config.SPLIT, index=False, encoding="utf-8-sig")
    print(split.groupby(["source", "split"]).agg(클립=("clip", "size"), 운전자=("driver_id", "nunique")).to_string())
    print("DMD:", split[split["source"] == "dmd"].drop_duplicates("driver_id")[["driver_id", "split"]].to_dict("records"))


if __name__ == "__main__":
    main()

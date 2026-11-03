# 특징 표와 라벨 JSON → 3초 창 windows.npz, 정규화 값 norm.npz (기술서 2.2, 2.4, 2.5, 5.6).
# split.json은 읽기만 한다. 값이 옛 기준인 config.py와 섞이지 않도록 창 설정은 이 파일에 둔다.
#
# 실행: python -m sequence_model.build_windows --labels <labels 폴더> --split <split.json>
#       [--features data/processed/features_nitymed] [--out data/processed/windows]
#       [--include-dmd --dmd-labels <폴더> --dmd-features <폴더>]

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model.config import FEATURE_COLUMNS
from sequence_model.windows import fit_normalizer

FPS = 15
WINDOW_FRAMES = 45
WINDOW_SEC = WINDOW_FRAMES / FPS
STRIDE_SEC = 0.5
YAWN_RATIO = 0.5
MAX_MISSING = 0.3

NORMAL, EYE_CLOSED, YAWN, HEAD_DOWN = 0, 1, 2, 3
UNCERTAIN = -1
CLASS_NAMES = ("정상", "눈 감김", "하품", "고개 숙임")
UNASSIGNED = "unassigned"       # DMD는 팀이 분할을 정할 때까지 어느 분할에도 넣지 않는다
EXCLUDE_END_UNCERTAIN, EXCLUDE_END_MISSING, EXCLUDE_RATIO = "끝 프레임 불확실", "끝 프레임 미검출", "불확실+미검출 비율"


def to_frame(sec: float, fps: float) -> int:
    """구간 경계(초) → t >= sec인 첫 프레임 번호. 라벨 시각은 소수 셋째 자리로 반올림돼 있어 그 오차만큼 허용한다."""
    return math.ceil(sec * fps - 0.001 * fps)


def _spans(label: dict, key: str) -> list[tuple[float, float]]:
    return [(s["start"], s["end"]) if isinstance(s, dict) else (s[0], s[1]) for s in label[key]]


def frame_states(label: dict, n_frames: int) -> np.ndarray:
    """라벨 → 원본 프레임별 상태 하나. 우선순위가 낮은 채널부터 칠해 불확실 > 눈 감김 > 고개 숙임 > 하품 > 정상이 되게 한다."""
    states = np.full(n_frames, NORMAL, dtype=int)
    for key, state in (("yawn", YAWN), ("head_down", HEAD_DOWN), ("eye_closed", EYE_CLOSED), ("uncertain", UNCERTAIN)):
        for start, end in _spans(label, key):
            states[to_frame(start, label["fps"]):to_frame(end, label["fps"])] = state
    return states


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """특징 CSV(시각 열 t) 또는 추출 JSON(시각 열 t_s) → (시각 (n,), 특징 (n, 11)). 미검출 프레임은 특징 0, face 0 (기술서 1.2)."""
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data["frames"], columns=data["columns"]).astype(float).rename(columns={"t_s": "t"})
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    face = FEATURE_COLUMNS.index("face")
    missing = np.isnan(X).any(axis=1) | (X[:, face] <= 0.5)
    X[missing] = 0.0
    X[~missing, face] = 1.0
    return df["t"].to_numpy(dtype=float), X


def nearest_frames(t: np.ndarray, times: np.ndarray) -> np.ndarray:
    """시각마다 가장 가까운 원본 프레임 번호 (기술서 2.2)."""
    right = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
    left = right - 1
    return np.where(times - t[left] <= t[right] - times, left, right)


def exclude_reason(states: np.ndarray, face: np.ndarray, max_missing: float = MAX_MISSING) -> str | None:
    """창을 뺄 이유, 없으면 None (기술서 2.5).

    기술서의 "얼굴 미검출이거나"는 프레임 기준이 없어 임시로 끝 프레임으로 본다. 불확실 비율의 분자에는 미검출을 더한다.
    """
    if states[-1] == UNCERTAIN:
        return EXCLUDE_END_UNCERTAIN
    if face[-1] <= 0.5:
        return EXCLUDE_END_MISSING
    if np.mean((states == UNCERTAIN) | (face <= 0.5)) >= max_missing:
        return EXCLUDE_RATIO
    return None


def window_label(states: np.ndarray) -> int:
    """창 안 프레임별 상태 → 정답. 눈 감김·고개 숙임은 끝 프레임, 하품은 창의 50% 이상 (기술서 2.4).

    상태를 하나로 정한 뒤 세므로 눈 감김과 겹친 프레임은 하품에 들어가지 않는다 (라벨 README 2·4번).
    """
    if states[-1] in (EYE_CLOSED, HEAD_DOWN):
        return int(states[-1])
    return YAWN if np.mean(states == YAWN) >= YAWN_RATIO else NORMAL


def clip_windows(t: np.ndarray, X: np.ndarray, states: np.ndarray, fps: float, max_missing: float = MAX_MISSING):
    """클립 하나 → (창 목록, 정답 목록, 시작 초 목록, 제외 Counter[(이유, 빼지 않았을 때의 정답)]).

    0.5초는 15 FPS에서 7.5프레임이라 전역 격자로는 나눠지지 않는다. 그래서 창마다 시작 시각에서 시각을 다시 계산한다.
    창보다 짧은 클립은 창이 나오지 않는다.
    """
    face = FEATURE_COLUMNS.index("face")
    windows, labels, starts, excluded = [], [], [], Counter()
    n_windows = int((len(t) / fps - WINDOW_SEC) / STRIDE_SEC + 1e-9) + 1 if len(t) / fps >= WINDOW_SEC else 0
    for i in range(n_windows):
        start = i * STRIDE_SEC
        idx = nearest_frames(t, start + np.arange(WINDOW_FRAMES) / FPS)
        label = window_label(states[idx])
        reason = exclude_reason(states[idx], X[idx, face], max_missing)
        if reason:
            excluded[(reason, label)] += 1
            continue
        windows.append(X[idx])
        labels.append(label)
        starts.append(start)
    return windows, labels, starts, excluded


def find_features(folder: Path, clip: str) -> Path | None:
    for name in (f"{clip}.csv", f"{clip}.json", f"{clip}_rgb_face.json"):
        if (folder / name).exists():
            return folder / name
    return None


def driver_overlap(rows: pd.DataFrame) -> dict[str, list[str]]:
    """둘 이상의 분할에 나오는 운전자 → 그 분할들. 분할이 안 정해진 창은 검사에서 뺀다."""
    assigned = rows[rows["split"] != UNASSIGNED]
    per_driver = assigned.groupby("driver_id")["split"].unique()
    return {driver: sorted(splits) for driver, splits in per_driver.items() if len(splits) > 1}


def collect(label_dir: Path, feature_dir: Path, split_of: dict[str, str] | None, drivers: dict[str, str] | None,
            max_missing: float, pattern: str = "*.json"):
    """폴더 하나 → (창 목록, 창 정보 행, 제외 Counter[(분할, 이유, 정답)], 건너뛴 클립 [(클립, 이유)]). split_of가 None이면 UNASSIGNED."""
    windows, rows, excluded, skipped = [], [], Counter(), []
    for path in sorted(label_dir.glob(pattern)):
        label = json.loads(path.read_text(encoding="utf-8"))
        clip = label["clip"]
        if label["label_status"] != "reviewed" and label["source"] != "dmd":
            skipped.append((clip, f"label_status={label['label_status']}"))     # 기술서 5.3: 검수본과 DMD 변환본만
            continue
        if split_of is not None and clip not in split_of:
            skipped.append((clip, "split.json에 없음"))
            continue
        if drivers is not None and drivers.get(clip) != label["driver_id"]:
            raise ValueError(f"{clip}: 라벨 driver_id {label['driver_id']}와 split.json {drivers.get(clip)}가 다릅니다")
        feature_path = find_features(feature_dir, clip)
        if feature_path is None:
            skipped.append((clip, "특징 파일 없음"))
            continue
        t, X = load_features(feature_path)
        if len(t) != label["frames"]:
            skipped.append((clip, f"특징 {len(t)}행, 라벨 {label['frames']}프레임"))
            continue
        split = UNASSIGNED if split_of is None else split_of[clip]
        clip_X, clip_y, starts, clip_excluded = clip_windows(t, X, frame_states(label, len(t)), label["fps"], max_missing)
        if not clip_X and not clip_excluded:
            skipped.append((clip, f"{WINDOW_SEC:g}초보다 짧음"))
            continue
        windows += clip_X
        rows += [(clip, str(label["driver_id"]), label["source"], split, y, s) for y, s in zip(clip_y, starts)]
        excluded.update({(split, *key): count for key, count in clip_excluded.items()})
    return windows, rows, excluded, skipped


def build(label_dir: Path, split_path: Path, feature_dir: Path, out_dir: Path, max_missing: float = MAX_MISSING,
          dmd_label_dir: Path | None = None, dmd_feature_dir: Path | None = None):
    """폴더 → windows.npz, norm.npz 저장 후 (창 정보 표, 제외 Counter, 건너뛴 클립, 운전자 겹침)을 돌려준다.

    test 창도 windows.npz에 split 표시와 함께 들어가지만 정규화 값은 train 창으로만 구한다.
    """
    split_file = json.loads(split_path.read_text(encoding="utf-8"))
    split_of = {clip: name for name, clips in split_file["splits"].items() for clip in clips}
    windows, rows, excluded, skipped = collect(label_dir, feature_dir, split_of, split_file["drivers"], max_missing, "P*.json")
    if dmd_label_dir is not None:
        dmd = collect(dmd_label_dir, dmd_feature_dir, None, None, max_missing)
        windows += dmd[0]
        rows += dmd[1]
        excluded.update(dmd[2])
        skipped += dmd[3]
    if not windows:
        raise ValueError("만들어진 창이 없습니다. 특징 폴더와 라벨 폴더를 확인하세요")

    info = pd.DataFrame(rows, columns=["clip", "driver_id", "source", "split", "y", "start_sec"])
    overlap = driver_overlap(info)
    if overlap:
        raise ValueError(f"운전자가 여러 분할에 있습니다: {overlap}")
    X = np.stack(windows).astype(np.float32)
    is_train = (info["split"] == "train").to_numpy()
    if not is_train.any():
        raise ValueError("train 창이 없어 정규화 값을 구할 수 없습니다")

    mean, std = fit_normalizer(X[is_train])
    keep = [i for i, name in enumerate(FEATURE_COLUMNS) if name != "face"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", X=X, y=info["y"].to_numpy(dtype=np.int64),
                        clip=info["clip"].to_numpy(dtype=str), driver_id=info["driver_id"].to_numpy(dtype=str),
                        source=info["source"].to_numpy(dtype=str), split=info["split"].to_numpy(dtype=str),
                        start_sec=info["start_sec"].to_numpy(dtype=np.float64))
    np.savez(out_dir / "norm.npz", mean=mean[keep], std=std[keep], columns=np.array([FEATURE_COLUMNS[i] for i in keep]))
    return info, excluded, skipped, overlap


def main() -> None:
    parser = argparse.ArgumentParser(description="특징 표와 라벨 JSON으로 windows.npz, norm.npz를 만든다")
    parser.add_argument("--labels", type=Path, required=True, help="NITYMED labels 폴더 (P*.json)")
    parser.add_argument("--split", type=Path, required=True, help="split.json (읽기만 한다)")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_nitymed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/windows"))
    parser.add_argument("--max-missing", type=float, default=MAX_MISSING)
    parser.add_argument("--include-dmd", action="store_true", help=f"DMD 창을 split={UNASSIGNED}로 더한다. 기본은 제외")
    parser.add_argument("--dmd-labels", type=Path)
    parser.add_argument("--dmd-features", type=Path)
    args = parser.parse_args()
    if args.include_dmd and not (args.dmd_labels and args.dmd_features):
        parser.error("--include-dmd에는 --dmd-labels와 --dmd-features가 필요합니다")

    info, excluded, skipped, overlap = build(args.labels, args.split, args.features, args.out, args.max_missing,
                                             args.dmd_labels if args.include_dmd else None,
                                             args.dmd_features if args.include_dmd else None)
    table = pd.crosstab(info["split"], info["y"].map(dict(enumerate(CLASS_NAMES))), margins=True, margins_name="합계")
    print(table.reindex(columns=[*CLASS_NAMES, "합계"], fill_value=0).to_string())
    print(f"\n클립 수: {info.groupby('split')['clip'].nunique().to_dict()}")
    print(f"운전자 수: {info.groupby('split')['driver_id'].nunique().to_dict()}, 겹침: {overlap or '없음'}")
    print(f"제외 창 {sum(excluded.values())}개")
    for (split, reason, label), count in sorted(excluded.items()):
        print(f"  {split} / {reason} / {CLASS_NAMES[label]}: {count}")
    print(f"건너뛴 클립 {len(skipped)}개: {skipped or '없음'}")
    print(f"저장: {args.out / 'windows.npz'}, {args.out / 'norm.npz'}")


if __name__ == "__main__":
    main()

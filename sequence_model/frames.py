# v1~v7 창 생성과 판정에 쓰는 함수(육주영). 팀 공통 창 묶음(v8~)은 sequence_model/build_windows.py(김민영)가 만든다.
# 라벨 JSON과 특징 → 15 FPS 프레임별 y와 3초 창 정답 (기획서 6·7·8번). 정답은 여기서만 계산한다.
#   data/windows/windows.npz  X (창, 45, 11) 정규화 전 값, y, 창 정보는 windows.csv
#   data/windows/skipped.csv  클립별 창 수와 뺀 이유(불확실, 미검출)
#   data/norm.json            train 창 검출 프레임의 특징별 평균·표준편차
#
# 실행: python -m sequence_model.frames

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

FEATURE_SOURCE = {"nitymed": config.FEATURE_DIR, "aihub089": config.AIHUB_FEATURES, "dmd": config.FEATURE_DIR}


def channel_masks(label: dict, n: int, fps: float) -> dict[str, np.ndarray]:
    """라벨 JSON → 원본 프레임별 채널 표시."""
    t = np.arange(n) / fps
    masks = {}
    for name in ("eye_closed", "uncertain", "head_down"):
        mask = np.zeros(n, dtype=bool)
        for start, end in label.get(name, []):
            mask |= (t >= start) & (t < end)
        masks[name] = mask
    yawn = np.zeros(n, dtype=bool)
    for event in label.get("yawn", []):
        yawn |= (t >= event["start"]) & (t < event["end"])
    masks["yawn"] = yawn
    return masks


def frame_states(channels: dict[str, np.ndarray]) -> np.ndarray:
    """프레임별 채널 → 상태. 겹치면 불확실 > 눈감김 > 고개숙임 > 하품 > 정상 (기획서 4번)."""
    n = len(next(iter(channels.values())))
    state = np.full(n, config.NORMAL)
    for name, value in reversed(config.PRIORITY):
        state[channels[name]] = value
    return state


def resample_index(t: np.ndarray, fps: float = config.FPS) -> tuple[np.ndarray, np.ndarray]:
    """원본 프레임 시각 → (15 FPS 격자 시각, 격자마다 가장 가까운 원본 프레임 번호)."""
    t = np.asarray(t, dtype=float)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1 / fps
    count = int(np.floor((t[-1] - t[0] + step) * fps + 1e-9))
    grid = t[0] + np.arange(count) / fps
    right = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    left = right - 1
    return grid, np.where(grid - t[left] <= t[right] - grid, left, right)


def feature_matrix(df: pd.DataFrame, relative: bool = False) -> np.ndarray:
    """특징 표 → (프레임, 11). 미검출 프레임은 face 0, 나머지 0 (기획서 3-3).
    relative면 세션 기준으로 바꾼다(session.py, v7부터)."""
    found = df["face_found"].to_numpy(dtype=bool)
    columns = [found.astype(float) if name == config.FACE_COLUMN else df[name].to_numpy(dtype=float) for name in config.FEATURES]
    X = np.column_stack(columns)
    missing = ~found | np.isnan(X).any(axis=1)
    X[missing] = 0.0
    X[:, config.FEATURES.index(config.FACE_COLUMN)] = (~missing).astype(float)
    X = X.astype(np.float32)
    if relative:
        from landmarks.session import relative_matrix
        X = relative_matrix(df["t"].to_numpy(dtype=float), X)
    return X


def window_label(states: np.ndarray) -> int:
    """창 안 프레임 상태 → 정답. 끝 프레임이 불확실이면 UNCERTAIN, 눈감김·고개숙임은 끝 프레임, 하품은 50% 이상."""
    end = int(states[-1])
    if end in (config.UNCERTAIN, config.CLOSED, config.HEAD):
        return end
    return config.YAWN if np.mean(states == config.YAWN) >= config.YAWN_RATIO else config.NORMAL


def end_indices(n: int, fps: float = config.FPS) -> np.ndarray:
    """창 끝 프레임 번호. 끝 시각이 STRIDE_SEC 간격이 되도록 고른다 (15 FPS에서 0.5초는 7.5프레임이라 시각으로 정한다)."""
    ends = np.arange(config.WINDOW_SEC, n / fps + 1e-9, config.STRIDE_SEC)
    index = np.round(ends * fps).astype(int) - 1
    return index[(index >= config.WINDOW_FRAMES - 1) & (index < n)]


def clip_windows(df: pd.DataFrame, label: dict, relative: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """클립 하나 → (창 X, 창 정답, 창 끝 시각, 뺀 수). 15 FPS로 맞춘 뒤 자른다."""
    fps = float(label["fps"])
    n = min(len(df), int(label["frames"]))
    df = df.iloc[:n]
    states = frame_states(channel_masks(label, n, fps))
    grid, index = resample_index(df["t"].to_numpy())
    X, states = feature_matrix(df, relative)[index], states[index]
    face = X[:, config.FEATURES.index(config.FACE_COLUMN)]
    keep_X, keep_y, keep_t = [], [], []
    counts = {"창": 0, "불확실": 0, "미검출": 0}
    for end in end_indices(len(grid)):
        span = slice(end - config.WINDOW_FRAMES + 1, end + 1)
        counts["창"] += 1
        y = window_label(states[span])
        if y == config.UNCERTAIN:
            counts["불확실"] += 1
            continue
        if 1 - face[span].mean() > config.MAX_MISSING:
            counts["미검출"] += 1
            continue
        keep_X.append(X[span])
        keep_y.append(y)
        keep_t.append(grid[end] - grid[0] + 1 / config.FPS)
    empty = np.zeros((0, config.WINDOW_FRAMES, len(config.FEATURES)), dtype=np.float32)
    return (np.stack(keep_X) if keep_X else empty), np.array(keep_y, dtype=int), np.array(keep_t), counts


def fit_norm(X: np.ndarray) -> dict:
    """train 창 → 특징별 평균·표준편차. 검출 프레임만 쓰고 face는 0, 1로 둔다 (기획서 8번)."""
    frames = X.reshape(-1, X.shape[-1])
    face = config.FEATURES.index(config.FACE_COLUMN)
    detected = frames[frames[:, face] > 0.5]
    mean, std = detected.mean(axis=0), detected.std(axis=0)
    mean[face], std[face] = 0.0, 1.0
    std = np.where(std < 1e-6, 1.0, std)
    return {"features": list(config.FEATURES), "mean": mean.tolist(), "std": std.tolist()}


def normalize(X: np.ndarray, norm: dict) -> np.ndarray:
    """창 (…, 프레임, 11)을 정규화한다. 미검출 프레임은 평균값인 0으로 둔다."""
    mean, std = np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["std"], dtype=np.float32)
    out = (X - mean) / std
    face = config.FEATURES.index(config.FACE_COLUMN)
    out[X[..., face] <= 0.5] = 0.0
    return out.astype(np.float32)


def limit_aihub(info: pd.DataFrame, share: float = config.AIHUB_SHARE, seed: int = config.SEEDS[0]) -> np.ndarray:
    """train에서 AI Hub 창이 share를 넘지 않도록 무작위로 고른 행 표시."""
    keep = np.ones(len(info), dtype=bool)
    train = (info["split"] == "train").to_numpy()
    aihub = train & (info["source"] == "aihub089").to_numpy()
    allowed = int(np.count_nonzero(train & ~aihub) * share / (1 - share))
    if aihub.sum() > allowed:
        rng = np.random.default_rng(seed)
        drop = rng.choice(np.flatnonzero(aihub), int(aihub.sum()) - allowed, replace=False)
        keep[drop] = False
    return keep


def export_spec(out_dir: Path, split: pd.DataFrame, norm: dict) -> None:
    """기획서 이름의 파일을 함께 쓴다 (내용은 split.csv·norm.json과 같다).
    split.json = {"unit", "splits": {분할: [클립]}, "drivers": {클립: 운전자}}, norm.npz = features, mean, std."""
    table = split[["clip", "driver_id", "split"]].sort_values(["split", "clip"])
    (out_dir / "split.json").write_text(json.dumps({"unit": "driver_id", "splits": {s: g["clip"].tolist() for s, g in table.groupby("split")},
                                                    "drivers": dict(zip(table["clip"], table["driver_id"]))}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    np.savez(out_dir / "norm.npz", features=np.array(norm["features"]), mean=np.asarray(norm["mean"], dtype=np.float32),
             std=np.asarray(norm["std"], dtype=np.float32), relative=bool(norm.get("relative", False)))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="라벨과 특징으로 3초 창과 정규화 값을 만든다")
    parser.add_argument("--sources", nargs="+", help="이 출처만 쓴다 (예: nitymed dmd). 없으면 split.csv 전부")
    parser.add_argument("--out", type=Path, help="결과 폴더. 주면 정규화 값도 그 폴더의 norm.json에 쓴다")
    parser.add_argument("--relative", action="store_true", help="특징을 세션 기준으로 바꾼다 (session.py, v7)")
    parser.add_argument("--export", type=Path, help="이미 만든 창 폴더에 split.json·norm.npz만 더 쓴다")
    args = parser.parse_args()
    if args.export:
        meta = json.loads((args.export / "meta.json").read_text(encoding="utf-8"))
        split = pd.read_csv(config.SPLIT, encoding="utf-8-sig")
        if meta.get("sources"):
            split = split[split["source"].isin(meta["sources"])]
        norm = json.loads((args.export / "norm.json").read_text(encoding="utf-8"))
        norm["relative"] = meta.get("relative", False)
        export_spec(args.export, split, norm)
        print(f"저장: {args.export / 'split.json'}, {args.export / 'norm.npz'}")
        return
    out_dir = args.out or config.WINDOW_DIR
    norm_path = out_dir / "norm.json" if args.out else config.NORM
    if args.relative and not args.out:
        raise SystemExit("--relative는 --out 폴더와 함께 쓴다 (기존 data/windows를 덮지 않게)")

    split = pd.read_csv(config.SPLIT, encoding="utf-8-sig")
    if args.sources:
        split = split[split["source"].isin(args.sources)]
    Xs, ys, rows, skipped = [], [], [], []
    for row in split.itertuples(index=False):
        label = json.loads((config.LABEL_DIR / f"{row.clip}.json").read_text(encoding="utf-8"))
        df = pd.read_parquet(FEATURE_SOURCE[row.source] / f"{row.clip}.parquet")
        X, y, ends, counts = clip_windows(df, label, args.relative)
        Xs.append(X)
        ys.append(y)
        rows += [{"clip": row.clip, "source": row.source, "driver_id": row.driver_id, "split": row.split, "end_sec": round(float(t), 3), "label": int(v)}
                 for t, v in zip(ends, y)]
        skipped.append({"clip": row.clip, "source": row.source, "split": row.split, **counts, "남은 창": len(y)})
    info = pd.DataFrame(rows)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    keep = limit_aihub(info)
    X, y, info = X[keep], y[keep], info[keep].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", X=X, y=y)
    info.to_csv(out_dir / "windows.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(skipped)
    table.to_csv(out_dir / "skipped.csv", index=False, encoding="utf-8-sig")
    norm = fit_norm(X[(info["split"] == "train").to_numpy()])
    norm_path.write_text(json.dumps(norm, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({"relative": bool(args.relative), "sources": args.sources}, ensure_ascii=False), encoding="utf-8")
    export_spec(out_dir, split, {**norm, "relative": bool(args.relative)})
    print(f"저장: {out_dir}, 정규화 {norm_path}, 세션 기준 {args.relative}")

    print(f"창 {len(y)}개 (AI Hub 창 {int((~keep).sum())}개를 줄였다)")
    print(pd.crosstab([info["split"], info["source"]], info["label"].map(dict(enumerate(config.CLASSES)))).to_string())
    total = table[["창", "불확실", "미검출"]].sum()
    print(f"뺀 창: 불확실 {total['불확실']}개, 미검출 {total['미검출']}개 / 전체 {total['창']}개 ({total['미검출'] / total['창']:.1%}가 미검출로 빠짐)")


if __name__ == "__main__":
    main()

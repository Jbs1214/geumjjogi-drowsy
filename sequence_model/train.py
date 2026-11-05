# 창 폴더(sequence_model/frames.py 또는 김민영 build_windows.py 결과)로 1D-CNN·LSTM을 GPU로 학습한다. 시드마다 가중치 하나, 표는 runs/<tag>/experiments.csv.
# 조기 종료는 NITYMED Val(사람 라벨)의 창 macro F1이다. Test 창은 읽지 않는다.
#
# 실행: python -m sequence_model.train --tag v1 [--models cnn1d lstm] [--seeds 42 43 44]

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import config
from sequence_model.frames import normalize
from eval.metrics import classification_metrics
from sequence_model.models import MODEL_NAMES, TAIL_FRAMES, build_model

DEVICE = "cuda"     # 학습은 GPU로만 한다
# 증강 (--augment, train 배치에만). README "성능 개선 판정 기준"
EAR_SCALE, MAR_SCALE = (0.85, 1.15), (0.9, 1.1)     # 사람마다 다른 눈·입 크기
NOISE_STD = 0.05                                    # 정규화 뒤 잡음
DROP_PROB, DROP_LEN = 0.1, (3, 6)                   # 연속 프레임 미검출 흉내
EAR_COLUMNS = tuple(config.FEATURES.index(c) for c in ("ear_l", "ear_r", "ear"))
MAR_COLUMN = config.FEATURES.index("mar")
FACE = config.FEATURES.index(config.FACE_COLUMN)


def augment(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """정규화된 창 (B, 45, 11) → 증강한 새 창. 미검출 프레임(face 0)은 0 그대로 두고, 미검출 비율이 MAX_MISSING을 넘지 않게 한다.
    끝 프레임은 정답을 정하는 프레임이라 지우지 않는다."""
    out = z.clone()
    found = out[:, :, FACE:FACE + 1] > 0.5
    b, n, _ = out.shape
    raw = out * std + mean
    ear_scale = torch.empty(b, 1, 1, device=z.device).uniform_(*EAR_SCALE, generator=gen)
    mar_scale = torch.empty(b, 1, device=z.device).uniform_(*MAR_SCALE, generator=gen)
    for c in EAR_COLUMNS:
        raw[:, :, c] = raw[:, :, c] * ear_scale[:, :, 0]
    raw[:, :, MAR_COLUMN] = raw[:, :, MAR_COLUMN] * mar_scale
    scaled = (raw - mean) / std
    scaled = scaled + torch.randn(scaled.shape, device=z.device, generator=gen) * NOISE_STD
    scaled[:, :, FACE] = out[:, :, FACE]
    out = torch.where(found, scaled, torch.zeros_like(scaled))
    drop = torch.rand(b, device=z.device, generator=gen) < DROP_PROB
    limit = int(config.MAX_MISSING * n)
    for i in torch.nonzero(drop).flatten().tolist():
        missing = int((out[i, :, FACE] < 0.5).sum())
        length = int(torch.randint(DROP_LEN[0], DROP_LEN[1] + 1, (1,), device=z.device, generator=gen))
        length = min(length, limit - missing)
        if length <= 0:
            continue
        start = int(torch.randint(0, n - length, (1,), device=z.device, generator=gen))     # 끝 프레임(n-1)은 남긴다
        out[i, start:start + length] = 0
    return out


def norm_from_npz(path: Path) -> dict:
    """김민영 build_windows의 norm.npz(face를 뺀 10개) → 11개 정규화 값. face 자리는 fit_norm과 같이 평균 0, 표준편차 1이다."""
    with np.load(path) as saved:
        columns, mean, std = saved["columns"].tolist(), saved["mean"], saved["std"]
    expected = [name for name in config.FEATURES if name != config.FACE_COLUMN]
    if columns != expected:
        raise SystemExit(f"{path}의 열 순서가 config.FEATURES와 다릅니다: {columns}")
    face = config.FEATURES.index(config.FACE_COLUMN)
    return {"features": list(config.FEATURES), "mean": np.insert(mean, face, 0.0).tolist(),
            "std": np.insert(std, face, 1.0).tolist(), "relative": False}


def load_windows(folder: Path = config.WINDOW_DIR) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """창 폴더 → (X, y, 창 정보, 정규화 값). 폴더에 norm.json이 있으면 그것을, 없으면 data/norm.json을 쓴다.
    김민영 build_windows 형식(창 정보가 windows.npz 안에, 정규화 값이 norm.npz)도 읽는다 (v8부터)."""
    data = np.load(folder / "windows.npz")
    if not (folder / "windows.csv").exists():
        info = pd.DataFrame({key: data[key] for key in ("clip", "driver_id", "source", "split", "start_sec")})
        return data["X"], data["y"], info, norm_from_npz(folder / "norm.npz")
    info = pd.read_csv(folder / "windows.csv", encoding="utf-8-sig")
    norm_path = folder / "norm.json" if (folder / "norm.json").exists() else config.NORM
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    meta = folder / "meta.json"
    # 세션 기준 특징으로 만든 창이면 가중치(norm)에 표시해 평가·실시간이 같은 변환을 쓰게 한다
    norm["relative"] = bool(json.loads(meta.read_text(encoding="utf-8")).get("relative")) if meta.exists() else False
    return data["X"], data["y"], info, norm


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=config.N_CLASSES).clip(min=1)
    return torch.tensor(len(y) / (config.N_CLASSES * counts), dtype=torch.float32)


def predict(model: nn.Module, X: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch]).argmax(dim=1) for i in range(0, len(X), batch)]).cpu().numpy()


def next_path(folder: Path, name: str) -> Path:
    versions = [int(m[1]) for p in folder.glob(f"{name}_v*.pt") if (m := re.fullmatch(rf"{name}_v(\d+)\.pt", p.name))]
    return folder / f"{name}_v{max(versions, default=0) + 1:02d}.pt"


def train_one(name: str, seed: int, data: dict, norm: dict, out_dir: Path, use_augment: bool = False, tail: int = TAIL_FRAMES) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    mean = torch.tensor(norm["mean"], dtype=torch.float32, device=DEVICE)
    std = torch.tensor(norm["std"], dtype=torch.float32, device=DEVICE)
    (X_train, y_train), (X_val, y_val) = data["train"], data["val"]
    inputs, targets = torch.from_numpy(X_train).to(DEVICE), torch.from_numpy(y_train).long().to(DEVICE)
    val_inputs = torch.from_numpy(X_val).to(DEVICE)
    model = build_model(name, tail).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    n_batches = len(inputs) // config.BATCH_SIZE or 1
    best = {"macro_f1": -1.0, "epoch": 0}
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        order = torch.randperm(len(inputs), device=DEVICE)
        for i in range(n_batches):
            batch = order[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE]
            optimizer.zero_grad()
            x = augment(inputs[batch], mean, std, gen) if use_augment else inputs[batch]
            loss = loss_fn(model(x), targets[batch])
            loss.backward()
            optimizer.step()
        metrics = classification_metrics(y_val, predict(model, val_inputs, config.BATCH_SIZE))
        print(f"{name} seed {seed} epoch {epoch}: loss {loss.item():.4f}, val macro F1 {metrics['macro_f1']:.3f}", flush=True)
        if metrics["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": metrics["macro_f1"], "epoch": epoch, "metrics": metrics,
                    "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= config.PATIENCE:
            break
    path = next_path(out_dir, name)
    torch.save({"model": name, "state_dict": best["state_dict"], "norm": norm, "seed": seed, "epoch": best["epoch"],
                "val_metrics": best["metrics"], "features": list(config.FEATURES), "classes": list(config.CLASSES),
                "fps": config.FPS, "window_sec": config.WINDOW_SEC, "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024 ** 2, "augment": use_augment, "tail": tail}, path)
    per_class = best["metrics"].get("f1_per_class", {})
    return {"모델": name, "seed": seed, "weights": str(path), "augment": use_augment, "tail": tail, "epoch": best["epoch"], "val_macro_f1": best["macro_f1"],
            **{f"val_F1_{config.CLASSES[int(c)]}": v for c, v in per_class.items()}, "초": round(time.perf_counter() - began, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="1D-CNN·LSTM 학습")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    parser.add_argument("--augment", action="store_true", help="train 배치에 증강을 쓴다 (눈·입 크기, 잡음, 미검출 흉내)")
    parser.add_argument("--windows", type=Path, default=config.WINDOW_DIR, help="창 폴더 (frames --out 또는 build_windows --out)")
    parser.add_argument("--tail", type=int, default=TAIL_FRAMES, help="1D-CNN 끝 요약 프레임 수 (v9 실험, 기본 8은 v1~v8과 같음)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU(CUDA)를 찾지 못했습니다. 학습은 GPU로만 합니다")
    out_dir = config.RUNS / args.tag
    if (out_dir / "experiments.csv").exists():
        raise SystemExit(f"{out_dir}에 결과가 있습니다. --tag에 새 이름을 쓰세요")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, info, norm = load_windows(args.windows)
    print(f"창 폴더 {args.windows}", flush=True)
    data = {}
    for split in ("train", "val"):
        keep = (info["split"] == split).to_numpy()
        if split == "val":
            keep = keep & (info["source"] == "nitymed").to_numpy()
        data[split] = (normalize(X[keep], norm), y[keep])
    print(f"학습 장치 {torch.cuda.get_device_name(0)}, train 창 {len(data['train'][1])}개, val 창 {len(data['val'][1])}개", flush=True)
    print("train 클래스별", np.bincount(data["train"][1], minlength=config.N_CLASSES).tolist(), "val", np.bincount(data["val"][1], minlength=config.N_CLASSES).tolist(), flush=True)
    rows = []
    for name in args.models:
        for seed in args.seeds:
            rows.append(train_one(name, seed, data, norm, out_dir, args.augment, args.tail))
            pd.DataFrame(rows).to_csv(out_dir / "experiments.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

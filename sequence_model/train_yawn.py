# 하품 하나만 답하는 이진 1D-CNN·LSTM을 GPU로 학습한다. 창·분할·정규화·학습 설정은 softmax 학습과 같다.
# 정답은 창의 하품 비율이 YAWN_RATIO 이상인지이고 눈 감김과 겹쳐도 하품이다. 최고 epoch는 모든 출처 val의 하품 F1로 고른다.
# Test 창은 읽지 않는다.
#
# 실행: python -m sequence_model.train_yawn --tag yawn1 --windows data/windows_sig3

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from sequence_model.build_windows import normalize
from sequence_model.models import CNN1D, MODEL_NAMES, LSTMClassifier
from sequence_model.train import DEVICE, load_windows, next_path


def build_yawn_model(name: str) -> torch.nn.Module:
    if name == "cnn1d":
        return CNN1D(n_classes=1)
    if name == "lstm":
        return LSTMClassifier(n_classes=1)
    raise ValueError(f"모델 이름은 {MODEL_NAMES} 중 하나여야 합니다: {name}")


class YawnPredictor:
    """가중치 파일 → 창 (N, 45, 11, 정규화 전) 하품 확률 (N, 1)."""

    outputs = 1

    def __init__(self, path, device: str = "cpu") -> None:
        saved = torch.load(path, map_location=device, weights_only=False)
        self.name, self.norm, self.device = saved["model"], saved["norm"], device
        self.relative = bool(self.norm.get("relative", False))
        self.model = build_yawn_model(saved["model"]).to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def predict(self, windows: np.ndarray, batch: int = 4096) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(windows), batch):
                x = torch.from_numpy(normalize(windows[i:i + batch], self.norm)).to(self.device)
                out.append(torch.sigmoid(self.model(x)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


def load_yawn(path: Path | None) -> YawnPredictor | None:
    """하품 전용 가중치. 파일이 없으면 None이고 4클래스 판정만 쓴다."""
    return YawnPredictor(path) if path is not None and Path(path).exists() else None


def f1(truth: np.ndarray, guess: np.ndarray) -> float:
    tp = float((truth & guess).sum())
    return 2 * tp / max(float(truth.sum() + guess.sum()), 1.0)


def train_one(name: str, seed: int, data: dict, norm: dict, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    (X_train, y_train), (X_val, y_val) = data["train"], data["val"]
    inputs, targets = torch.from_numpy(X_train).to(DEVICE), torch.from_numpy(y_train).float().to(DEVICE)
    val_inputs = torch.from_numpy(X_val).to(DEVICE)
    model = build_yawn_model(name).to(DEVICE)
    positive = max(float(y_train.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y_train) - positive) / positive], device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    n_batches = len(inputs) // config.BATCH_SIZE or 1
    best = {"score": -1.0, "epoch": 0}
    began = time.perf_counter()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        order = torch.randperm(len(inputs), device=DEVICE)
        for i in range(n_batches):
            batch = order[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(inputs[batch])[:, 0], targets[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = [torch.sigmoid(model(val_inputs[i:i + config.BATCH_SIZE]))[:, 0] for i in range(0, len(val_inputs), config.BATCH_SIZE)]
        score = f1(y_val, torch.cat(scores).cpu().numpy() >= config.YAWN_SELECT_THRESHOLD)
        print(f"{name} seed {seed} epoch {epoch}: loss {loss.item():.4f}, val 하품 F1 {score:.3f}", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= config.PATIENCE:
            break
    path = next_path(out_dir, name)
    torch.save({"model": name, "output": "yawn", "state_dict": best["state_dict"], "norm": norm, "seed": seed, "epoch": best["epoch"],
                "val_f1": best["score"], "features": list(config.FEATURES), "fps": config.FPS, "window_sec": config.WINDOW_SEC}, path)
    return {"모델": name, "seed": seed, "weights": str(path), "epoch": best["epoch"], "val_하품_F1": best["score"], "초": round(time.perf_counter() - began, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="하품 전용 이진 모델을 학습한다")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    parser.add_argument("--windows", type=Path, default=config.WINDOW_DIR, help="창 폴더 (build_windows --out)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU(CUDA)를 찾지 못했습니다. 학습은 GPU로만 합니다")
    out_dir = config.RUNS / args.tag
    if (out_dir / "experiments.csv").exists():
        raise SystemExit(f"{out_dir}에 결과가 있습니다. --tag에 새 이름을 쓰세요")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, _, info, norm = load_windows(args.windows)
    with np.load(args.windows / "windows.npz") as saved:
        targets = saved["yawn"]
    data = {}
    for split in ("train", "val"):
        keep = (info["split"] == split).to_numpy()
        data[split] = (normalize(X[keep], norm), targets[keep])
    print(f"학습 장치 {torch.cuda.get_device_name(0)}, train 창 {len(data['train'][1])}개(하품 {int(data['train'][1].sum())}), val 창 {len(data['val'][1])}개(하품 {int(data['val'][1].sum())})", flush=True)
    rows = []
    for name in args.models:
        for seed in args.seeds:
            rows.append(train_one(name, seed, data, norm, out_dir))
            pd.DataFrame(rows).to_csv(out_dir / "experiments.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

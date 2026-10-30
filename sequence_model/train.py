# 시간창 데이터로 시퀀스 분류기를 학습하고 가중치를 checkpoints/best_model_vNN.pt로 저장한다.
# 정답은 자동 라벨이므로 이 점수는 규칙을 얼마나 따라 배웠는지만 보여 준다. 평가는 eval/compare.py가 검수 라벨로 한다.
# 실행: python -m sequence_model.train --model cnn1d --calibration-clips <보정 클립 목록.txt>

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval.metrics import classification_metrics
from sequence_model import config
from sequence_model.models import MODEL_NAMES, build_model
from sequence_model.windows import build_dataset, fit_normalizer, normalize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="시퀀스 분류기 학습")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--labels-dir", type=Path, default=Path("data/processed/labels_auto"), help="자동 라벨 JSON 폴더")
    parser.add_argument("--calibration-clips", type=Path, help="보정에 쓴 클립 번호 목록(한 줄에 하나). 이 클립의 그룹은 평가에서 뺀다")
    parser.add_argument("--window-sec", type=float, default=config.WINDOW_SEC)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8, help="val macro F1이 이 epoch 수만큼 안 오르면 멈춘다")
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args(argv)


def read_clip_list(path: Path | None) -> tuple[str, ...]:
    """클립 번호 목록 파일 → 튜플. 파일을 주지 않으면 빈 튜플."""
    return tuple(path.read_text(encoding="utf-8-sig").split()) if path else ()


def class_weights(y: np.ndarray) -> torch.Tensor:
    """클래스 불균형을 맞추는 손실 가중치. 없는 클래스는 1개로 친다."""
    counts = np.bincount(y, minlength=config.N_CLASSES).clip(min=1)
    return torch.tensor(len(y) / (config.N_CLASSES * counts), dtype=torch.float32)


def predict_classes(model: nn.Module, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(X[i:i + batch_size]).to(device)).argmax(dim=1).cpu().numpy()
                               for i in range(0, len(X), batch_size)])


def train(model: nn.Module, data: dict, args: argparse.Namespace) -> dict:
    """train 분할로 학습하고 val macro F1이 가장 높은 결과를 돌려준다. 결과: state_dict, epoch, val_metrics."""
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    model.to(args.device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(X_train) > args.batch_size,    # BatchNorm은 크기 1인 배치를 처리하지 못한다
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train).to(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {"val_metrics": {"macro_f1": -1.0}}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(args.device)), yb.to(args.device))
            loss.backward()
            optimizer.step()

        metrics = classification_metrics(y_val, predict_classes(model, X_val, args.device, args.batch_size))
        print(f"epoch {epoch}: loss {loss.item():.4f}, val macro F1 {metrics['macro_f1']:.3f}")
        if metrics["macro_f1"] > best["val_metrics"]["macro_f1"]:
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best = {"state_dict": state, "epoch": epoch, "val_metrics": metrics}
        elif epoch - best["epoch"] >= args.patience:
            break
    return best


def next_weight_path(out_dir: Path) -> Path:
    """다음 가중치 경로. 학습할 때마다 버전을 올린다 (best_model_v01.pt, v02, …)."""
    versions = [int(m[1]) for p in out_dir.glob("best_model_v*.pt") if (m := re.fullmatch(r"best_model_v(\d+)\.pt", p.name))]
    return out_dir / f"best_model_v{max(versions, default=0) + 1:02d}.pt"


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_dataset(args.labels_dir, read_clip_list(args.calibration_clips), args.window_sec)
    X_train, y_train = dataset.split("train")
    X_val, y_val = dataset.split("val")
    if not len(X_train) or not len(X_val):
        raise ValueError(f"train 창 {len(X_train)}개, val 창 {len(X_val)}개: 분할할 클립 그룹이 부족합니다")

    mean, std = fit_normalizer(X_train)
    data = {"train": (normalize(X_train, mean, std), y_train), "val": (normalize(X_val, mean, std), y_val)}
    model = build_model(args.model, X_train.shape[-1], config.N_CLASSES)
    best = train(model, data, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = next_weight_path(args.out_dir)
    torch.save({
        "model": args.model,
        "state_dict": best["state_dict"],
        "n_features": X_train.shape[-1],
        "n_classes": config.N_CLASSES,
        "feature_columns": config.FEATURE_COLUMNS,
        "window_sec": args.window_sec,
        "fps": config.FPS,
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }, path)
    print(f"{path}: epoch {best['epoch']}, val {best['val_metrics']}")
    return path


if __name__ == "__main__":
    main()

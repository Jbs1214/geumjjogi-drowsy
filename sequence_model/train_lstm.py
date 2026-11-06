# 4클래스 LSTM을 windows.npz·norm.npz로 학습한다 (기술서 2.2, 2.6, 4절).
# 창 상수는 build_windows.py, 학습 설정(EPOCHS·BATCH_SIZE·LR·PATIENCE)은 1D-CNN과 같은 config.py 값을 쓴다 (기술서 4.2).
# Test는 --final일 때만 읽고, 모델 선택은 Val로만 한다. --final은 다시 학습하지 않고 이미 학습한 가중치로 한 번만 잰다.
#
# 실행: python -m sequence_model.train_lstm --tag v01 [--bundle <폴더>] [--seeds 42 43 44] [--device cuda|cpu]
#       python -m sequence_model.train_lstm --tag v01 --final
#       python -m sequence_model.train_lstm --tag diag --seeds 42 43 44 --diagnose     (학습 설정은 그대로, 진단 내용만 더 적는다)
#       python -m sequence_model.train_lstm --tag v01_last --pool last                 (마지막 시점 출력을 쓰는 참고 실험)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from eval.window_eval import markdown_table
from eval.metrics import classification_metrics
from sequence_model.build_windows import CLASS_NAMES, EYE_CLOSED, FEATURE_COLUMNS, HEAD_DOWN, WINDOW_FRAMES

DEFAULT_BUNDLE = Path("data/processed/bundle_20261103_84e7bf4")
HIDDEN = 32
DROPOUT = 0.3
EPOCHS, BATCH_SIZE, LR, PATIENCE = config.EPOCHS, config.BATCH_SIZE, config.LR, config.PATIENCE
SEEDS = config.SEEDS                    # 1D-CNN(train.py)과 같은 시드 (기술서 4.2)
POOLS = {"mean": "lstm", "last": "lstm_last"}   # 풀링 → 체크포인트 model 값. lstm_last는 predictor.py에서 오류로 멈춘다 (평균 풀링으로 잘못 돌지 않게)

N_CLASSES = len(CLASS_NAMES)
FACE = FEATURE_COLUMNS.index("face")
NORM_COLUMNS = [i for i in range(len(FEATURE_COLUMNS)) if i != FACE]


class LSTMClassifier(nn.Module):
    """이 스크립트의 LSTM. 전체 출력 평균 하나만 쓴다. models.LSTMClassifier(평균 + 끝 0.5초)와 가중치가 호환되지 않는다."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output.mean(dim=1))


class LastStepLSTM(LSTMClassifier):
    """참고 실험용. 시간축 평균 대신 마지막 시점 출력만 쓴다."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


def make_model(pool: str, n_features: int, n_classes: int, hidden: int, dropout: float) -> nn.Module:
    model = LastStepLSTM if pool == "last" else LSTMClassifier
    return model(n_features, n_classes, hidden=hidden, dropout=dropout)


def load_bundle(bundle: Path, splits: tuple[str, ...]) -> dict:
    """번들 폴더 → 고른 분할의 (X, y)와 정규화 값. 고르지 않은 분할의 행은 여기서 버린다."""
    with np.load(bundle / "windows.npz", allow_pickle=False) as data:
        X, y, split = data["X"], data["y"], data["split"]
    with np.load(bundle / "norm.npz", allow_pickle=False) as data:
        mean, std, columns = data["mean"], data["std"], data["columns"]

    if X.shape[1:] != (WINDOW_FRAMES, len(FEATURE_COLUMNS)):
        raise SystemExit(f"X 모양은 (N, {WINDOW_FRAMES}, {len(FEATURE_COLUMNS)})여야 합니다: {X.shape}")
    expected = [FEATURE_COLUMNS[i] for i in NORM_COLUMNS]
    if [str(c) for c in columns] != expected:
        raise SystemExit(f"norm.npz 열이 {expected}가 아닙니다: {[str(c) for c in columns]}")
    if mean.shape != (len(expected),) or std.shape != (len(expected),) or (std <= 0).any():
        raise SystemExit(f"norm.npz 값이 잘못됐습니다: mean {mean.shape}, std {std.shape}, 최솟값 {std.min()}")

    data = {"mean": mean, "std": std}
    for name in splits:
        keep = split == name
        if not keep.any():
            raise SystemExit(f"windows.npz에 split={name} 행이 없습니다")
        if y[keep].min() < 0 or y[keep].max() >= N_CLASSES:
            raise SystemExit(f"split={name}의 y는 0~{N_CLASSES - 1}이어야 합니다")
        data[name] = (X[keep], y[keep].astype(np.int64))
    return data


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(X-mean)/std를 face를 뺀 열에만 적용하고, 미검출 프레임(face=0)은 전부 0으로 둔다."""
    out = X.astype(np.float32).copy()
    out[..., NORM_COLUMNS] = (out[..., NORM_COLUMNS] - mean) / std
    out[X[..., FACE] <= 0.5] = 0.0
    return out


def class_weights(y: np.ndarray) -> torch.Tensor:
    """클래스 불균형을 맞추는 손실 가중치 N/(4×개수). 없는 클래스는 1개로 친다."""
    counts = np.bincount(y, minlength=N_CLASSES).clip(min=1)
    return torch.tensor(len(y) / (N_CLASSES * counts), dtype=torch.float32)


def norm_check(raw_train: np.ndarray, raw_val: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """정규화 전 창 → 정규화가 맞는 열에 들어갔는지 확인할 값. windows.npz에 열 이름이 없어 숫자로 대조한다."""
    def stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = X.reshape(-1, X.shape[-1])
        detected = frames[frames[:, FACE] > 0.5].astype(np.float64)
        return detected.mean(axis=0), detected.std(axis=0)

    train = normalize(raw_train, mean, std)
    raw_mean, raw_std = stats(raw_train)
    train_mean, train_std = stats(train)
    val_mean, val_std = stats(normalize(raw_val, mean, std))
    columns = [{"column": FEATURE_COLUMNS[i], "raw_mean": float(raw_mean[i]), "raw_std": float(raw_std[i]),
                "saved_mean": float(mean[k]), "saved_std": float(std[k]),
                "train_mean": float(train_mean[i]), "train_std": float(train_std[i]),
                "val_mean": float(val_mean[i]), "val_std": float(val_std[i])}
               for k, i in enumerate(NORM_COLUMNS)]
    missing = raw_train[..., FACE] <= 0.5
    return {
        "columns": columns,
        "matches_saved": bool(np.allclose(raw_mean[NORM_COLUMNS], mean, rtol=1e-4, atol=1e-4)
                              and np.allclose(raw_std[NORM_COLUMNS], std, rtol=1e-4, atol=1e-4)),
        "face_unchanged": bool(np.array_equal(train[..., FACE], raw_train[..., FACE])),
        "face_values": [float(v) for v in np.unique(raw_train[..., FACE])],
        "missing_frames": int(missing.sum()),
        "total_frames": int(missing.size),
        "missing_all_zero": bool((train[missing] == 0).all()),
        "nan_raw": int(np.isnan(raw_train).sum()),
        "nan_normalized": int(np.isnan(train).sum()),
    }


def fixed_averages(f1_per_class: dict) -> dict:
    """클래스별 F1 → 참고용 고정 평균. 정답·예측에 없는 클래스는 0으로 넣는다."""
    f1 = [f1_per_class.get(c, 0.0) for c in range(N_CLASSES)]
    return {"macro_f1_4class": float(np.mean(f1)),
            "macro_f1_3class": float(np.mean([v for c, v in enumerate(f1) if c != HEAD_DOWN]))}


def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(torch.from_numpy(X[i:i + batch_size]).to(device)), dim=1).cpu().numpy()
                               for i in range(0, len(X), batch_size)])


def confusion_matrix(y: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    """행 정답, 열 예측."""
    return np.bincount(N_CLASSES * y + pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES).tolist()


def _epoch_row(model: nn.Module, epoch: int, loss_fn: nn.Module, splits: dict, args: argparse.Namespace) -> dict:
    """epoch 끝의 진단 값. 드롭아웃을 끈 eval 모드로 재므로 난수를 쓰지 않고 학습에 영향이 없다."""
    model.eval()
    row = {"epoch": epoch}
    for name, (X, y) in splits.items():
        with torch.no_grad():
            logits = torch.cat([model(torch.from_numpy(X[i:i + args.batch_size]).to(args.device))
                                for i in range(0, len(X), args.batch_size)])
            loss = loss_fn(logits, torch.from_numpy(y).to(args.device)).item()
        metrics = classification_metrics(y, logits.argmax(dim=1).cpu().numpy())
        row |= {f"{name}_loss": loss, f"{name}_macro_f1": metrics["macro_f1"], f"{name}_eye_f1": metrics["f1_per_class"].get(EYE_CLOSED, 0.0)}
    return row


def fit(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
        args: argparse.Namespace, seed: int) -> dict:
    """Val macro F1이 가장 높은 epoch를 고른다. PATIENCE epoch 동안 오르지 않으면 멈춘다."""
    model.to(args.device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train).to(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {"metrics": {"macro_f1": -1.0}}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(args.device)), yb.to(args.device))
            loss.backward()
            optimizer.step()

        metrics = classification_metrics(y_val, predict_proba(model, X_val, args.batch_size, args.device).argmax(axis=1))
        print(f"시드 {seed} epoch {epoch}: loss {loss.item():.4f}, val macro F1 {metrics['macro_f1']:.3f}")
        if args.diagnose:
            history.append(_epoch_row(model, epoch, loss_fn, {"train": (X_train, y_train), "val": (X_val, y_val)}, args))
        if metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best = {"state_dict": state, "best_epoch": epoch, "metrics": metrics}
        elif epoch - best["best_epoch"] >= args.patience:
            break
    return {**best, "stopped_epoch": epoch, "history": history}


def _suffix(args: argparse.Namespace) -> str:
    """CPU 결과 표시. 태그에 이미 _cpu가 있으면 다시 붙이지 않는다."""
    return "_cpu" if args.device == "cpu" and not args.tag.endswith("_cpu") else ""


def weight_path(args: argparse.Namespace, seed: int) -> Path:
    return args.checkpoints_dir / f"lstm_{args.tag}_seed{seed}{_suffix(args)}.pt"


def probs_path(args: argparse.Namespace, seed: int, split: str) -> Path:
    return args.runs_dir / f"lstm_{args.tag}_seed{seed}{_suffix(args)}_{split}_probs.npy"


def results_path(args: argparse.Namespace, kind: str, extension: str) -> Path:
    return args.runs_dir / f"lstm_{args.tag}{_suffix(args)}_{kind}.{extension}"


def _row_metrics(metrics: dict) -> dict:
    """classification_metrics 결과 → JSON에 쓸 값. 클래스별 F1 키는 클래스 이름이다."""
    return {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
            "f1_per_class": {CLASS_NAMES[c]: f1 for c, f1 in metrics["f1_per_class"].items()},
            **fixed_averages(metrics["f1_per_class"])}


def _class_counts(y: np.ndarray) -> dict:
    return {name: int(count) for name, count in zip(CLASS_NAMES, np.bincount(y, minlength=N_CLASSES))}


def to_report(title: str, rows: list[dict], counts: dict) -> str:
    """시드별 결과 → 마크다운. 고개 숙임은 창이 적어 참고값이다. 값이 '-'이면 정답과 예측 모두에 그 클래스가 없다."""
    main_headers = ["시드", "멈춘 epoch", "macro F1", *CLASS_NAMES[:-1], f"{CLASS_NAMES[-1]}(참고)", "풀링"]
    main_rows = [[r["seed"], r.get("stopped_epoch"), r["macro_f1"], *[r["f1_per_class"].get(name) for name in CLASS_NAMES], r["pool"]] for r in rows]
    note_rows = [[r["seed"], r.get("best_epoch"), r["macro_f1_4class"], r["macro_f1_3class"]] for r in rows]
    return "\n\n".join([
        f"# {title}",
        markdown_table(main_headers, main_rows),
        "참고 평균 (macro F1은 정답·예측에 나온 클래스만 평균, 아래는 클래스를 고정한 평균)",
        markdown_table(["시드", "가장 좋았던 epoch", "4클래스 고정 평균", "3클래스 평균(고개 숙임 제외)"], note_rows),
        "클래스별 창 수",
        markdown_table(["분할", *CLASS_NAMES], [[name, *[count[c] for c in CLASS_NAMES]] for name, count in counts.items()]),
    ]) + "\n"


def seed_spread(rows: list[dict]) -> list[list]:
    """시드별 F1 → 항목별 평균, 표준편차(모표준편차), 최솟값, 최댓값."""
    items = {"macro F1": [r["macro_f1"] for r in rows],
             **{name: [r["f1_per_class"].get(name, 0.0) for r in rows] for name in CLASS_NAMES}}
    return [[name, float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v))] for name, v in items.items()]


def diagnose_report(check: dict, weights: list[float], rows: list[dict]) -> str:
    """--diagnose 내용 → 마크다운."""
    keys = ["column", "raw_mean", "saved_mean", "raw_std", "saved_std", "train_mean", "train_std", "val_mean", "val_std"]
    parts = [
        "## 정규화 점검",
        f"norm.npz와 원본 train 값 일치: {check['matches_saved']}, face 열 그대로: {check['face_unchanged']} (값 {check['face_values']}), "
        f"미검출 프레임 {check['missing_frames']}/{check['total_frames']} 전부 0: {check['missing_all_zero']}, "
        f"NaN 원본 {check['nan_raw']}, 정규화 후 {check['nan_normalized']}",
        markdown_table(["열", "원본 평균", "저장 평균", "원본 표준편차", "저장 표준편차", "train 평균", "train 표준편차", "val 평균", "val 표준편차"],
                       [[c[k] for k in keys] for c in check["columns"]]),
        "## 클래스 가중치",
        markdown_table(list(CLASS_NAMES), [weights]),
        "## 시드별 변동",
        markdown_table(["항목", "평균", "표준편차", "최솟값", "최댓값"], seed_spread(rows)),
    ]
    history_keys = ["epoch", "train_loss", "val_loss", "train_macro_f1", "val_macro_f1", "train_eye_f1", "val_eye_f1"]
    for r in rows:
        parts += [
            f"## 시드 {r['seed']} 혼동행렬 (행 정답, 열 예측, 가장 좋았던 epoch {r['best_epoch']})",
            markdown_table(["정답 \\ 예측", *CLASS_NAMES], [[name, *line] for name, line in zip(CLASS_NAMES, r["confusion_matrix"])]),
            f"## 시드 {r['seed']} epoch 기록",
            markdown_table(["epoch", "train loss", "val loss", "train macro F1", "val macro F1", "train 눈 감김 F1", "val 눈 감김 F1"],
                           [[h[k] for k in history_keys] for h in r["history"]]),
        ]
    return "\n\n".join(parts) + "\n"


def _write_results(path_json: Path, path_md: Path, result: dict, report: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    path_md.write_text(report, encoding="utf-8")
    print(report)
    print(f"저장: {path_json}, {path_md}")


def resolve_device(device: str) -> None:
    """cuda를 고르면 없을 때 멈춘다. CPU는 --device cpu로 명시했을 때만 쓴다."""
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("GPU(CUDA)를 찾지 못했습니다. 코드 확인만 하려면 --device cpu를 명시하세요")
        print(f"학습 장치: {torch.cuda.get_device_name(0)}")
    else:
        print("학습 장치: cpu (코드 확인용)")


def run_training(args: argparse.Namespace) -> dict:
    outputs = [results_path(args, "results", "json"), results_path(args, "results", "md")]
    for seed in args.seeds:
        outputs += [weight_path(args, seed), probs_path(args, seed, "val")]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise SystemExit(f"--tag {args.tag}의 결과가 이미 있습니다. 재학습은 새 --tag를 쓰세요: {existing}")

    data = load_bundle(args.bundle, ("train", "val"))
    X_train, y_train = normalize(data["train"][0], data["mean"], data["std"]), data["train"][1]
    X_val, y_val = normalize(data["val"][0], data["mean"], data["std"]), data["val"][1]

    check = norm_check(data["train"][0], data["val"][0], data["mean"], data["std"]) if args.diagnose else None
    rows = []
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = make_model(args.pool, len(FEATURE_COLUMNS), N_CLASSES, HIDDEN, DROPOUT)
        best = fit(model, X_train, y_train, X_val, y_val, args, seed)
        model.load_state_dict(best["state_dict"])

        torch.save({"model": POOLS[args.pool], "pool": args.pool, "state_dict": best["state_dict"], "hidden": HIDDEN, "dropout": DROPOUT,
                    "n_features": len(FEATURE_COLUMNS), "n_classes": N_CLASSES, "class_names": list(CLASS_NAMES),
                    "feature_columns": list(FEATURE_COLUMNS), "mean": torch.from_numpy(data["mean"]),
                    "std": torch.from_numpy(data["std"]), "tag": args.tag, "seed": seed, "device": args.device,
                    "best_epoch": best["best_epoch"], "stopped_epoch": best["stopped_epoch"]}, weight_path(args, seed))
        val_probs = predict_proba(model, X_val, args.batch_size, args.device)
        np.save(probs_path(args, seed, "val"), val_probs)
        rows.append({"seed": seed, "pool": args.pool, "stopped_epoch": best["stopped_epoch"], "best_epoch": best["best_epoch"], **_row_metrics(best["metrics"]),
                     "weights": str(weight_path(args, seed)), "val_probs": str(probs_path(args, seed, "val"))})
        if args.diagnose:
            rows[-1] |= {"confusion_matrix": confusion_matrix(y_val, val_probs.argmax(axis=1)), "history": best["history"]}

    counts = {"train": _class_counts(y_train), "val": _class_counts(y_val)}
    result = {"tag": args.tag, "device": args.device, "bundle": str(args.bundle), "class_counts": counts, "seeds": rows,
              "settings": {"hidden": HIDDEN, "dropout": DROPOUT, "epochs": args.epochs, "batch_size": args.batch_size,
                           "lr": args.lr, "patience": args.patience, "pool": args.pool}}
    report = to_report(f"LSTM {args.tag} Val 결과 (장치: {args.device}, 풀링: {args.pool})", rows, counts)
    if args.diagnose:
        weights = class_weights(y_train).tolist()
        result["diagnostics"] = {"norm_check": check, "class_weights": dict(zip(CLASS_NAMES, weights))}
        report += "\n" + diagnose_report(check, weights, rows)
    _write_results(results_path(args, "results", "json"), results_path(args, "results", "md"), result, report)
    return result


def run_final(args: argparse.Namespace) -> dict:
    """학습 때 저장한 Val 결과로 시드를 고르고, 그 가중치로 Test를 한 번만 잰다."""
    source = results_path(args, "results", "json")
    if not source.exists():
        raise SystemExit(f"{source}가 없습니다. --final은 다시 학습하지 않으니 같은 --tag와 --device로 먼저 학습하세요")
    final_json, final_md = results_path(args, "final", "json"), results_path(args, "final", "md")
    if final_json.exists():
        raise SystemExit(f"{final_json}가 이미 있습니다. Test는 한 번만 씁니다")

    trained = json.loads(source.read_text(encoding="utf-8"))
    seed = max(trained["seeds"], key=lambda r: r["macro_f1"])["seed"]     # 모델 선택은 Val로만 한다
    checkpoint = torch.load(weight_path(args, seed), map_location=args.device, weights_only=True)
    X_test, y_test = load_bundle(args.bundle, ("test",))["test"]
    pool = checkpoint.get("pool", "mean")
    model = make_model(pool, checkpoint["n_features"], checkpoint["n_classes"], checkpoint["hidden"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)

    probs = predict_proba(model, normalize(X_test, checkpoint["mean"].numpy(), checkpoint["std"].numpy()), args.batch_size, args.device)
    np.save(probs_path(args, seed, "test"), probs)
    row = {"seed": seed, "pool": pool, **_row_metrics(classification_metrics(y_test, probs.argmax(axis=1))),
           "weights": str(weight_path(args, seed)), "test_probs": str(probs_path(args, seed, "test"))}
    counts = {"test": _class_counts(y_test)}
    result = {"tag": args.tag, "device": args.device, "bundle": str(args.bundle), "class_counts": counts, "seed": row}
    _write_results(final_json, final_md, result, to_report(f"LSTM {args.tag} Test 결과 (장치: {args.device}, 풀링: {pool})", [row], counts))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4클래스 LSTM 학습")
    parser.add_argument("--tag", required=True, help="결과 이름. 재학습은 새 이름을 쓴다")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="windows.npz와 norm.npz가 있는 폴더")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cpu는 코드 확인용이고 결과에 cpu로 표시한다")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE, help="val macro F1이 이 epoch 수만큼 안 오르면 멈춘다")
    parser.add_argument("--final", action="store_true", help="학습한 가중치로 Test를 한 번 잰다. 다시 학습하지 않는다")
    parser.add_argument("--pool", choices=tuple(POOLS), default="mean", help="시간축 풀링. last는 참고 실험용이다. --final은 체크포인트 값을 쓴다")
    parser.add_argument("--diagnose", action="store_true", help="학습 설정은 그대로 두고 정규화 점검, epoch 기록, 혼동행렬, 클래스 가중치를 더 적는다")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.final and args.diagnose:
        raise SystemExit("--diagnose는 Val 진단용이라 --final과 함께 쓸 수 없습니다")
    resolve_device(args.device)
    return run_final(args) if args.final else run_training(args)


if __name__ == "__main__":
    main()

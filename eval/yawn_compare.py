# 하품 사건 검출을 MAR 규칙, 4클래스 softmax의 하품 출력, 하품 전용 모델로 비교한다. val만 읽는다.
# 기준은 README "하품 전용 모델 비교 기준"에 있다. 세 방법 모두 같은 인정 시간으로 채점한다.
#
# 실행: python -m eval.yawn_compare --softmax sig4 --yawn yawn1

import argparse
import json

import numpy as np
import pandas as pd

import config
from eval.evaluate import load_clip, model_probs
from rule_based.common import runs
from rule_based.yawn import yawning
from sequence_model.models import Predictor
from sequence_model.train_yawn import YawnPredictor

TOLERANCES = (1.0, 3.0)         # 하품이 끝난 뒤 이 시간까지의 검출을 인정한다
THRESHOLDS = (config.YAWN_ENTER, 0.8)
GROUPS = (("nitymed", False, "NITYMED_맨입"), ("dmd", False, "DMD_맨입"), ("dmd", True, "DMD_손"))


def detections(times: np.ndarray, on: np.ndarray, min_sec: float = 0.0) -> list[tuple[float, float]]:
    return [(times[a], times[b]) for a, b in zip(*runs(on)) if times[b] - times[a] >= min_sec]


def score(rows: list[dict], lag: float) -> dict:
    """클립별 (사건, 검출, 시간, 출처) → 묶음별 재현율과 출처별 시간당 오탐."""
    out, hits = {}, {name: [] for *_, name in GROUPS}
    wrong, hours = {"nitymed": 0, "dmd": 0}, {"nitymed": 0.0, "dmd": 0.0}
    for row in rows:
        for event in row["events"]:
            hit = any(a <= event["end"] + lag and b >= event["start"] for a, b in row["found"])
            for source, hand, name in GROUPS:
                if row["source"] == source and event["hand"] == hand:
                    hits[name].append(hit)
        wrong[row["source"]] += sum(not any(a <= e["end"] + lag and b >= e["start"] for e in row["events"]) for a, b in row["found"])
        hours[row["source"]] += row["hours"]
    for name, values in hits.items():
        out[f"{name}_재현율"] = float(np.mean(values)) if values else np.nan
    for source in wrong:
        out[f"{source}_오탐"] = wrong[source] / hours[source]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="하품 사건 검출 비교 (val만)")
    parser.add_argument("--softmax", required=True)
    parser.add_argument("--yawn", required=True)
    args = parser.parse_args()
    split = pd.read_csv(config.SPLIT, encoding="utf-8-sig", dtype={"driver_id": str})
    split = split[split["split"] == "val"]
    clips = []
    for row in split.itertuples():
        clip = load_clip(row.clip)
        label = json.loads((config.LABEL_DIR / f"{row.clip}.json").read_text(encoding="utf-8"))
        clips.append({"clip": clip, "source": row.source, "events": label["yawn"]})

    rows = []

    def add(name: str, model: str, threshold: float | None, seed: int, found_by_clip: list) -> None:
        table = [{"events": c["events"], "found": f, "hours": c["clip"]["hours"], "source": c["source"]} for c, f in zip(clips, found_by_clip)]
        for lag in TOLERANCES:
            rows.append({"방법": name, "모델": model, "기준": threshold, "인정초": lag, "seed": seed, **score(table, lag)})

    found = []
    for c in clips:
        df = c["clip"]["df"]
        t = df["t"].to_numpy()
        found.append(detections(t, yawning(t, df["mar"].to_numpy())))
    add("규칙 MAR", "-", None, 0, found)

    for name, tag, loader, output in (("4클래스 softmax", args.softmax, Predictor, config.YAWN), ("하품 전용", args.yawn, YawnPredictor, 0)):
        table = pd.read_csv(config.RUNS / tag / "experiments.csv", encoding="utf-8-sig")
        for run in table.itertuples(index=False):
            predictor = loader(run.weights)
            outputs = [model_probs(c["clip"]["df"], predictor) for c in clips]
            for threshold in THRESHOLDS:
                add(name, run.모델, threshold, run.seed, [detections(t, (p[:, output] >= threshold) & j, config.YAWN_MODEL_MIN_SEC) for t, p, j in outputs])
            print(f"채점: {run.weights}", flush=True)

    detail = pd.DataFrame(rows)
    out = config.RUNS / args.yawn / "yawn_compare.csv"
    detail.to_csv(out, index=False, encoding="utf-8-sig")
    columns = [c for c in detail.columns if c.endswith("_재현율") or c.endswith("_오탐")]
    summary = detail.groupby(["인정초", "방법", "모델", "기준"], dropna=False, sort=False)[columns].mean().round(3)
    print(f"저장: {out}")
    print(summary.to_string())


if __name__ == "__main__":
    main()

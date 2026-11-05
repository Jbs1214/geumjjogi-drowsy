# 자체 촬영 영상으로 규칙, 4클래스 모델, 최종 구성을 비교한다. 기준은 README "자체 영상 예비 비교 기준"에 있다.
#   팀원이 각자 같은 기준으로 돌릴 수 있게 사람별 표와 개수 파일을 runs/self/<이름>/에 쓴다.
#   결과를 보고 가중치나 임계값을 바꾸지 않는다. 라벨은 사람이 검수한 reviewed만 쓴다.
#
# 실행: python -m eval.self_compare --name 장범수
#       python -m eval.self_compare --merge runs/self/장범수 runs/self/김민영 runs/self/육주영   (개수를 합쳐 전체 표를 낸다)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from demo.demo_video import yawn_flags
from demo.webcam import RULE_MIN_SEC, BASELINE_EVERY_SEC
from eval.evaluate import (BASELINE_EAR_QUANTILE, EAR_RATIO, TARGETS, clip_score, grid_mouth_closed, grid_yawning, head_down, load_clip, model_probs,
                           model_timer_unknown, rule_timer_inputs)
from eval.metrics import ALERT_TOLERANCE_SEC, prf
from eval.yawn_compare import detections
from rule_based.yawn import yawning
from sequence_model.models import Predictor
from sequence_model.postprocess import Post, decide, shown_yawn
from sequence_model.train_yawn import YawnPredictor

POST = Post(enter=0.6, exit=0.4)
LAG_SEC = 1.0           # 하품 사건 인정 시간. eval.yawn_compare의 주 기준과 같다
COUNTS = "counts.csv"
MODELS = {"4클래스 1D-CNN": ("sig5", "cnn1d"), "4클래스 LSTM": ("sig5", "lstm")}
YAWN_MODEL = ("yawn2", "cnn1d")


def live_closed(df: pd.DataFrame) -> np.ndarray:
    """demo.webcam.LiveRule과 같은 감음: 지금까지의 평소 EAR × ear_ratio 미만. 미래 프레임을 쓰지 않는다."""
    t, ear, found = df["t"].to_numpy(), df["ear"].to_numpy(), df["face_found"].to_numpy(dtype=bool)
    out, seen, base, last = np.zeros(len(df), dtype=bool), [], np.nan, -np.inf
    for i in range(len(df)):
        if found[i] and not np.isnan(ear[i]):
            seen.append(ear[i])
        if t[i] - last >= BASELINE_EVERY_SEC and seen:
            base, last = float(np.quantile(seen, BASELINE_EAR_QUANTILE)), t[i]
        out[i] = t[i] >= RULE_MIN_SEC and found[i] and ear[i] < EAR_RATIO * base
    return out


def alarm_counts(score: dict, clip: dict, sec: float) -> dict:
    """경고 → 사건 수, 잡은 사건 수, 오경보 수."""
    alerts, events = score["alerts"][sec], clip["events"][sec]
    hit = sum(any(s <= a <= e + ALERT_TOLERANCE_SEC for a in alerts) for s, e in events)
    wrong = sum(not any(s <= a <= e + ALERT_TOLERANCE_SEC for s, e in events) for a in alerts)
    return {"events": len(events), "hit": hit, "wrong": wrong}


def yawn_counts(events: list[dict], found: list[tuple[float, float]], span: tuple[float, float]) -> dict:
    """하품 사건별 맨입·손 가림 수, 잡은 수, 오탐 수. 첫 3초 안의 사건은 뺀다."""
    keep = [e for e in events if e["end"] > span[0]]
    near = lambda a, b, e: a <= e["end"] + LAG_SEC and b >= e["start"]
    out = {"bare": 0, "bare_hit": 0, "hand": 0, "hand_hit": 0}
    for e in keep:
        key = "hand" if e["hand"] else "bare"
        out[key] += 1
        out[f"{key}_hit"] += any(near(a, b, e) for a, b in found)
    out["yawn_wrong"] = sum(not any(near(a, b, e) for e in keep) for a, b in found if b > span[0])
    return out


def rows_for(name: str, clip: dict, events: list[dict], times: np.ndarray, flags: dict, step: float, unknown, yawn_found: list, seed: int) -> dict:
    score = clip_score(clip, times, flags, step, unknown=unknown)
    row = {"방법": name, "시드": seed, "클립": clip["clip"], "시간": clip["hours"]}
    for target in TARGETS:
        row[f"{target}_tp"], row[f"{target}_fp"], row[f"{target}_fn"] = [int(v) for v in score[target]]
    for sec, key in ((3.0, "졸음"), (1.5, "주의")):
        row.update({f"{key}_{k}": v for k, v in alarm_counts(score, clip, sec).items()})
    row.update(yawn_counts(events, yawn_found, clip["span"]))
    return row


def score_clip(clip: dict, events: list[dict], fourclass: dict, yawn_runs: list) -> list[dict]:
    df = clip["df"]
    t = df["t"].to_numpy()
    step = 1 / clip["fps"]
    closed = live_closed(df)
    down = head_down(df)
    mar_yawn = yawning(t, df["mar"].to_numpy())
    rule_found = detections(t, mar_yawn)
    rule_flags = {"눈감김": closed, "하품": mar_yawn, "고개숙임": down}
    rows = [rows_for("규칙", clip, events, t, rule_flags, step, rule_timer_inputs(df)["unknown"], rule_found, 0)]

    mouths = grid_mouth_closed(df), grid_yawning(df)
    for name, predictors in fourclass.items():
        for seed, predictor in enumerate(predictors):
            times, probs, judged = model_probs(df, predictor)
            classes = decide(probs, judged, POST)
            flags = {label: classes == value for label, value in TARGETS.items()}
            flags["하품"] = shown_yawn(classes, *mouths, POST.yawn_mode)
            found = detections(times, flags["하품"], config.YAWN_MODEL_MIN_SEC)
            rows.append(rows_for(name, clip, events, times, flags, 1 / config.FPS, model_timer_unknown(classes), found, seed))

    for seed, predictor in enumerate(yawn_runs):
        times, _, _ = model_probs(df, predictor)
        on = yawn_flags(df, predictor)
        found = detections(times, on, 0.0)
        final = {"눈감김": closed, "고개숙임": down}
        score = rows_for("최종 구성", clip, events, t, {**final, "하품": np.zeros(len(t), dtype=bool)}, step, rule_timer_inputs(df)["unknown"], found, seed)
        yawn_score = clip_score(clip, times, {"하품": on}, 1 / config.FPS)
        score["하품_tp"], score["하품_fp"], score["하품_fn"] = [int(v) for v in yawn_score["하품"]]
        rows.append(score)
    return rows


def summarize(counts: pd.DataFrame) -> pd.DataFrame:
    """개수를 합쳐 방법별 지표를 낸다. 시드는 합친 뒤 평균한다."""
    rows = []
    for (name, seed), group in counts.groupby(["방법", "시드"], sort=False):
        total = group.drop(columns=["방법", "시드", "클립"]).sum()
        row = {"방법": name, "시드": seed, "시간": total["시간"]}
        for target in TARGETS:
            row[f"{target}_F1"] = prf(int(total[f"{target}_tp"]), int(total[f"{target}_fp"]), int(total[f"{target}_fn"]))["f1"]
        for key in ("졸음", "주의"):
            row[f"{key}_재현율"] = total[f"{key}_hit"] / total[f"{key}_events"] if total[f"{key}_events"] else np.nan
            row[f"{key}_오경보"] = total[f"{key}_wrong"] / total["시간"]
            row[f"{key}_사건"] = total[f"{key}_events"]
        row["맨입_재현율"] = total["bare_hit"] / total["bare"] if total["bare"] else np.nan
        row["손가림_재현율"] = total["hand_hit"] / total["hand"] if total["hand"] else np.nan
        row["하품_오탐"] = total["yawn_wrong"] / total["시간"]
        row["맨입_사건"], row["손가림_사건"] = total["bare"], total["hand"]
        rows.append(row)
    return pd.DataFrame(rows).groupby("방법", sort=False).mean(numeric_only=True).drop(columns="시드").round(3)


def predictors(tag: str, model: str, loader) -> list:
    table = pd.read_csv(config.RUNS / tag / "experiments.csv", encoding="utf-8-sig")
    return [loader(config.RUNS / tag / Path(p.replace("\\", "/")).name) for p in table.loc[table["모델"] == model, "weights"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="자체 영상으로 규칙과 모델을 비교한다")
    parser.add_argument("--name", help="사람 이름. 파일 이름이 이 이름으로 시작하는 라벨을 쓴다")
    parser.add_argument("--merge", nargs="+", type=Path, help="사람별 결과 폴더를 합친다")
    args = parser.parse_args()
    if args.merge:
        counts = pd.concat([pd.read_csv(folder / COUNTS, encoding="utf-8-sig") for folder in args.merge])
        out = config.RUNS / "self" / "전체"
    else:
        labels = sorted(config.LABEL_DIR.glob(f"{args.name}_*.json"))
        if not labels:
            raise SystemExit(f"{config.LABEL_DIR}에 {args.name}_로 시작하는 라벨이 없습니다")
        fourclass = {name: predictors(tag, model, Predictor) for name, (tag, model) in MODELS.items()}
        yawn_runs = predictors(*YAWN_MODEL, YawnPredictor)
        rows = []
        for path in labels:
            label = json.loads(path.read_text(encoding="utf-8"))
            if label["label_status"] != "reviewed":
                raise SystemExit(f"{path.name}은 검수되지 않았습니다")
            table = pd.read_parquet(config.FEATURE_DIR / f"{path.stem}.parquet", columns=["t"])
            if int(label["frames"]) != len(table):        # 라벨의 프레임 수는 녹화 시간 기준이라 변환한 영상에 맞춘다
                label["frames"] = len(table)
                path.write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
            rows += score_clip(load_clip(path.stem), label["yawn"], fourclass, yawn_runs)
            print(f"채점: {path.stem}", flush=True)
        counts = pd.DataFrame(rows)
        out = config.RUNS / "self" / args.name
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / COUNTS, index=False, encoding="utf-8-sig")
    table = summarize(counts)
    table.to_csv(out / "summary.csv", encoding="utf-8-sig")
    print(table.T.to_string())


if __name__ == "__main__":
    main()

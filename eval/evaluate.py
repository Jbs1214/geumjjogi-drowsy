# 사람 라벨(NITYMED)로 모델과 규칙을 채점한다. 기준은 README "판정 기준"에 있다.
#   val:  Val로 1D-CNN과 LSTM 중 하나를 고른다. 결과 runs/<tag>/eval_val
#   test: 고른 모델을 Test로 한 번 채점하고 눈 규칙 두 가지 중 나은 쪽과 비교한다. 결과가 있으면 다시 돌지 않는다
# 모델은 15 FPS 프레임마다 그 프레임에서 끝나는 3초 창을 판정한다. 눈감김 시간은 규칙과 같은 ClosureTimer로 잰다.
#
# 실행: python -m eval.evaluate val --tag v1
#       python -m eval.evaluate test --tag v1

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from sequence_model.build_windows import channel_masks, feature_matrix, frame_states, resample_index
from eval.metrics import (ALERT_TOLERANCE_SEC, alert_delay, classification_metrics, event_recall, false_alarms_per_hour, match_segments, prf,
                          segments_within)
from sequence_model.models import Predictor
from sequence_model.postprocess import CANDIDATES, NONE, Post, bridge, decide, drop_short, shown_yawn
from rule_based.alarm import DROWSY, WARN, stage_starts, timer_stages
from rule_based.common import runs
from rule_based.config import EAR_BASE_PERCENTILE
from rule_based.eyes import best_ear_ratio, is_closed
from rule_based.head import is_down
from rule_based.yawn import yawning

EAR_RATIO = best_ear_ratio()
BASELINE_EAR_QUANTILE = EAR_BASE_PERCENTILE / 100
MIN_CLOSE_SEC, MAX_GAP_SEC = 0.1, 0.2     # 평소 EAR 비율 규칙의 짧은 끊김 잇기와 짧은 구간 지우기 (이전 폴더 interim_label과 같다)
EYE_RULES = ("MVP 규칙 (EAR 0.2)", f"규칙 EAR {EAR_RATIO:g}")
YAWN_RULE, HEAD_RULE = "하품 규칙 (MAR)", "고개숙임 규칙 (pitch)"
TARGETS = {"눈감김": config.CLOSED, "하품": config.YAWN, "고개숙임": config.HEAD}
SIMPLER = "cnn1d"
BOOTSTRAPS, SEED = 2000, 42


def ear48_closed(df: pd.DataFrame, fps: float) -> np.ndarray:
    """EAR이 클립의 평소 값(best_params의 ear_ratio × EAR_BASE_PERCENTILE 분위) 미만인 프레임. 0.2초 미만 끊김은 잇고 0.1초 미만 구간은 지운다."""
    found = df["face_found"].to_numpy(dtype=bool)
    if not found.any():
        return np.zeros(len(df), dtype=bool)
    mask = (df["ear"].to_numpy() < EAR_RATIO * df.loc[found, "ear"].quantile(BASELINE_EAR_QUANTILE)) & found
    for start, end in zip(*runs(~mask)):
        if start > 0 and end < len(mask) - 1 and (end - start + 1) / fps < MAX_GAP_SEC:
            mask[start:end + 1] = True
    for start, end in zip(*runs(mask)):
        if (end - start + 1) / fps < MIN_CLOSE_SEC:
            mask[start:end + 1] = False
    return mask


def model_probs(df: pd.DataFrame, predictor: Predictor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """특징 표 → (15 FPS 판정 시각, 프레임별 출력별 확률 (N, 출력 수), 판정 여부).
    판정은 0.2초(격자 JUDGE_EVERY칸)마다 하고 사이 프레임은 직전 판정을 유지한다(수정 기획서 F 5-3).
    첫 창이 차기 전, 창의 미검출이 30%를 넘을 때, 창 끝 프레임에서 얼굴이 안 보일 때는 판정 불가다(5-1)."""
    grid, index = resample_index(df["t"].to_numpy())
    X = feature_matrix(df, getattr(predictor, "relative", False))[index]
    probs = np.zeros((len(grid), getattr(predictor, "outputs", config.N_CLASSES)))
    judged = np.zeros(len(grid), dtype=bool)
    first = config.WINDOW_FRAMES - 1
    if len(X) < config.WINDOW_FRAMES:
        return grid, probs, judged
    windows = np.lib.stride_tricks.sliding_window_view(X, config.WINDOW_FRAMES, axis=0).transpose(0, 2, 1)
    at = np.arange(0, len(windows), config.JUDGE_EVERY)          # 판정하는 창 (창 끝 = first + at)
    face = config.FEATURES.index(config.FACE_COLUMN)
    p = predictor.predict(np.ascontiguousarray(windows[at]))
    ok = (1 - windows[at, :, face].mean(axis=1) <= config.MAX_MISSING) & (windows[at, -1, face] > 0.5)
    hold = np.minimum((np.arange(len(windows)) // config.JUDGE_EVERY), len(at) - 1)  # 사이 프레임은 직전 판정
    probs[first:] = p[hold]
    judged[first:] = ok[hold]
    return grid, probs, judged


def head_down(df: pd.DataFrame) -> np.ndarray:
    return is_down(*(df[name].to_numpy() for name in ("t", "pitch", "ear_l", "ear_r")))


def rule_timer_inputs(df: pd.DataFrame) -> dict:
    """규칙의 타이머 입력 (최종 기획서 5-1·5-3): 판정 불가 = 그 프레임에서 얼굴이 안 보임, 고개 숙임 = pitch 규칙.
    둘 다 직전 눈 감김이 0.5초 이상이면 이어서 센다. 하품 중 눈 감김도 센다."""
    return {"unknown": ~df["face_found"].to_numpy(dtype=bool) | head_down(df)}


def model_timer_unknown(classes: np.ndarray) -> np.ndarray:
    """모델의 타이머 이어받기 대상: 판정 불가(-1)와 고개숙임 클래스 (최종 기획서 5-1·5-3)."""
    return (classes < 0) | (classes == config.HEAD)


def grid_mouth_closed(df: pd.DataFrame) -> np.ndarray:
    """원본 프레임의 입 다문 채 이어진 시간 → 15 FPS 격자 (화면 하품 표시용)."""
    from rule_based.alerts import mouth_closed_seconds

    _, index = resample_index(df["t"].to_numpy())
    return mouth_closed_seconds(df["t"].to_numpy(), df["mar"].to_numpy())[index]


def grid_yawning(df: pd.DataFrame) -> np.ndarray:
    """원본 프레임의 하품 중(MAR 규칙) → 15 FPS 격자. 격자마다 모델 입력과 같은 원본 프레임을 쓴다."""
    _, index = resample_index(df["t"].to_numpy())
    return yawning(df["t"].to_numpy(), df["mar"].to_numpy())[index]


def model_frames(df: pd.DataFrame, predictor: Predictor, post: Post = NONE) -> tuple[np.ndarray, np.ndarray]:
    """특징 표 → (15 FPS 판정 시각, 프레임별 클래스). 판정하지 않는 프레임은 -1이다. post는 후처리(기본은 argmax 그대로)."""
    grid, probs, judged = model_probs(df, predictor)
    return grid, decide(probs, judged, post)


def intervals(times: np.ndarray, flags: np.ndarray, step: float) -> list[tuple[float, float]]:
    return [(float(times[s]), float(times[e] + step)) for s, e in zip(*runs(flags))]


def wipe(times: np.ndarray, flags: np.ndarray, uncertain: list) -> np.ndarray:
    """불확실 구간의 판정을 지운다. 모든 방법에 똑같이 적용한다."""
    out = flags.copy()
    for start, end in uncertain:
        out[(times >= start) & (times < end)] = False
    return out


def load_clip(clip: str) -> dict:
    """클립 → 특징, 정답 구간(클래스별), 불확실 구간, 3초·1.5초 이상 눈 감김, 평가 시간."""
    label = json.loads((config.LABEL_DIR / f"{clip}.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(config.FEATURE_DIR / f"{clip}.parquet").iloc[:label["frames"]]
    fps = float(label["fps"])
    t = np.arange(len(df)) / fps
    states = frame_states(channel_masks(label, len(df), fps))
    truth = {name: intervals(t, states == value, 1 / fps) for name, value in TARGETS.items()}
    # 경고 채점의 정답: 하품 중 눈 감김도 넣는다 (최종 기획서는 하품 중 타이머 제외를 두지 않는다)
    closures = merge(truth["눈감김"])
    uncertain = [tuple(s) for s in label["uncertain"]]
    duration = len(df) / fps
    return {"clip": clip, "df": df, "fps": fps, "truth": truth, "uncertain": uncertain, "span": (config.WINDOW_SEC, duration),
            "events": {sec: [c for c in closures if round(c[1] - c[0], 3) >= sec] for sec in (3.0, 1.5)},
            "hours": (duration - sum(e - s for s, e in uncertain)) / 3600}


def merge(spans: list[tuple[float, float]], gap: float = 0.2) -> list[tuple[float, float]]:
    """사람이 나눠 표시한 이어지는 눈감김 구간을 합친다 (이전 폴더 merged_closures와 같다)."""
    out: list[list[float]] = []
    for start, end in sorted(spans):
        if out and start - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def clip_score(clip: dict, times: np.ndarray, flags: dict[str, np.ndarray], step: float, grace: float = 0.0,
               min_close: float = 0.0, unknown: np.ndarray | None = None, yawn: np.ndarray | None = None,
               carry_max: float | None = None) -> dict:
    """클래스별 판정 → 클래스별 구간 개수와 눈감김 경고 시각. grace는 눈감김의 끊김 허용(구간은 그만큼 잇고, 타이머도 이어 간다).
    min_close보다 짧은 눈감김 구간은 구간 채점에서 지운다(타이머는 그대로).
    타이머: unknown(판정 불가·고개 숙임)은 0.5초 이상 감은 뒤면 이어받고(최대 carry_max초), yawn을 주면 그 프레임의 감음은 넣지 않는다."""
    out = {}
    for name, value in flags.items():
        mask = wipe(times, value, clip["uncertain"])
        if name == "눈감김":
            mask = drop_short(bridge(times, mask, grace), step, min_close)
        out[name] = np.array(match_segments(segments_within(clip["truth"][name], *clip["span"]), segments_within(intervals(times, mask, step), *clip["span"])))
    if "눈감김" in flags:
        closed = flags["눈감김"] if yawn is None else flags["눈감김"] & ~yawn
        stages = timer_stages(times, wipe(times, closed, clip["uncertain"]), grace_sec=grace, unknown=unknown, carry_max_sec=carry_max)
        out["alerts"] = {3.0: stage_starts(times, stages, DROWSY), 1.5: stage_starts(times, stages, WARN)}
    return out


def alarm_scores(scores: dict, clips: list[dict], sec: float) -> dict:
    """경고 → 재현율, 지연, 시간당 오경보. 오경보는 기준 이상 눈 감김에 속하지 않는 경고다."""
    total = hits = wrong = 0
    delays = []
    for clip in clips:
        alerts, events = scores[clip["clip"]]["alerts"][sec], clip["events"][sec]
        total += len(events)
        hits += round(event_recall(events, alerts) * len(events)) if events else 0
        wrong += sum(not any(s <= a <= e + ALERT_TOLERANCE_SEC for s, e in events) for a in alerts)
        delay = alert_delay(events, alerts, sec)
        if not np.isnan(delay):
            delays.append(delay)
    hours = sum(c["hours"] for c in clips)
    return {"재현율": hits / total if total else float("nan"), "시간당_오경보": false_alarms_per_hour(np.arange(wrong), hours),
            "지연_초": float(np.mean(delays)) if delays else float("nan")}


def method_row(name: str, seed: int, scores: dict, clips: list[dict]) -> dict:
    row = {"방법": name, "시드": seed}
    for target in TARGETS:
        if target in next(iter(scores.values())):
            row[f"{target}_구간F1"] = prf(*np.sum([scores[c["clip"]][target] for c in clips], axis=0).tolist())["f1"]
    if "alerts" in next(iter(scores.values())):
        for sec, key in ((3.0, "3초"), (1.5, "주의")):
            for metric, value in alarm_scores(scores, clips, sec).items():
                row[f"{key}_{metric}"] = value
    return row


def bootstrap_f1(seeds: list[dict], clips: list[dict], target: str = "눈감김") -> tuple[float, float]:
    """클립을 다시 뽑아 시드 평균 F1의 95% 구간을 낸다."""
    rng = np.random.default_rng(SEED)
    names = [c["clip"] for c in clips]
    counts = np.array([[scores[n][target] for n in names] for scores in seeds])
    values = [np.mean([prf(*counts[s, picks].sum(axis=0).tolist())["f1"] for s in range(len(seeds))])
              for picks in (rng.integers(0, len(names), len(names)) for _ in range(BOOTSTRAPS))]
    return tuple(np.quantile(values, [0.025, 0.975]))


def window_macro_f1(predictor: Predictor, split: str) -> float:
    """build_windows.py 창으로 낸 창 단위 macro F1 (NITYMED만). 세션 기준 가중치는 세션 기준 창으로 잰다."""
    folder = config.DATA / "windows_session" if getattr(predictor, "relative", False) else config.WINDOW_DIR
    data = np.load(folder / "windows.npz")
    info = pd.read_csv(folder / "windows.csv", encoding="utf-8-sig")
    keep = ((info["split"] == split) & (info["source"] == "nitymed")).to_numpy()
    return classification_metrics(data["y"][keep], predictor.predict(data["X"][keep]).argmax(axis=1))["macro_f1"]


def selection_clips(with_webcam: bool = False) -> list[dict]:
    """선택에 쓰는 클립. NITYMED Val, with_webcam이면 Val2(웹캠 녹화, 사람 라벨을 9번 양식으로 바꾼 것)를 더한다."""
    table = pd.read_csv(config.SPLIT, encoding="utf-8-sig")
    clips = [load_clip(c) for c in table[(table["source"] == "nitymed") & (table["split"] == "val")]["clip"]]
    if with_webcam:
        webcam = [p.stem for p in sorted(config.WEBCAM_DIR.glob("*.mp4")) if (config.LABEL_DIR / f"{p.stem}.json").exists()]
        if not webcam:
            raise SystemExit("Val2 라벨이 없습니다. python -m eval.score_user --set webcam을 먼저 돌려 라벨을 변환하세요")
        clips += [load_clip(c) for c in webcam]
    return clips


def score_split(split: str, tag: str, with_webcam: bool = False) -> pd.DataFrame:
    if split == "val":
        return score_clips(selection_clips(with_webcam), tag, split)
    table = pd.read_csv(config.SPLIT, encoding="utf-8-sig")
    clips = [load_clip(c) for c in table[(table["source"] == "nitymed") & (table["split"] == split)]["clip"]]
    return score_clips(clips, tag, split)


def model_scores(clips: list[dict], predictor: Predictor, posts) -> dict:
    """모델 하나 → 후처리 설정마다 클립별 채점. 확률은 한 번만 계산한다."""
    probs = {c["clip"]: model_probs(c["df"], predictor) for c in clips}
    mouths = {c["clip"]: (grid_mouth_closed(c["df"]), grid_yawning(c["df"])) for c in clips}
    out = {}
    for post in posts:
        scores = {}
        for clip in clips:
            times, p, judged = probs[clip["clip"]]
            classes = decide(p, judged, post)
            flags = {name: classes == value for name, value in TARGETS.items()}
            flags["하품"] = shown_yawn(classes, *mouths[clip["clip"]], post.yawn_mode)     # 하품은 화면과 같은 판정으로 채점
            scores[clip["clip"]] = clip_score(clip, times, flags, 1 / config.FPS, post.grace, post.min_close,
                                              unknown=model_timer_unknown(classes), carry_max=post.carry_max)
        out[post] = scores
    return out


def score_clips(clips: list[dict], tag: str, window_split: str | None = None, post: Post = NONE,
                models: tuple[str, ...] | None = None) -> pd.DataFrame:
    """load_clip 결과 목록 → 모델(시드마다)과 규칙의 채점 행. window_split을 주면 그 분할의 창 macro F1도 낸다.
    post는 모델에 쓰는 후처리, models를 주면 그 모델 종류만 채점한다."""
    runs_table = pd.read_csv(config.RUNS / tag / "experiments.csv", encoding="utf-8-sig")
    rows = []
    for row in runs_table.itertuples(index=False):
        if models is not None and row.모델 not in models:
            continue
        predictor = Predictor(Path(row.weights))
        scores = model_scores(clips, predictor, [post])[post]
        window = {"창_macroF1": window_macro_f1(predictor, window_split)} if window_split else {}
        rows.append({**method_row(row.모델, row.seed, scores, clips), **window, "_scores": scores})
    rules = {EYE_RULES[0]: lambda c: {"눈감김": is_closed(c["df"]["ear"].to_numpy())},
             EYE_RULES[1]: lambda c: {"눈감김": ear48_closed(c["df"], c["fps"])},
             YAWN_RULE: lambda c: {"하품": yawning(c["df"]["t"].to_numpy(), c["df"]["mar"].to_numpy())},
             HEAD_RULE: lambda c: {"고개숙임": head_down(c["df"])}}
    for name, make in rules.items():
        scores = {c["clip"]: clip_score(c, c["df"]["t"].to_numpy(), make(c), 1 / c["fps"], **rule_timer_inputs(c["df"])) for c in clips}
        rows.append({**method_row(name, 0, scores, clips), "_scores": scores})
    detail = pd.DataFrame(rows)
    detail.attrs["clips"] = clips
    return detail


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    clips = detail.attrs["clips"]
    numeric = detail.drop(columns=["시드", "_scores"])
    summary = numeric.groupby("방법", sort=False).mean().reset_index()
    summary.insert(2, "시드_표준편차", numeric.groupby("방법", sort=False)["눈감김_구간F1"].std(ddof=1).reindex(summary["방법"]).to_numpy())
    eye = [m for m in summary["방법"] if m not in (YAWN_RULE, HEAD_RULE)]
    ci = {m: bootstrap_f1(list(detail[detail["방법"] == m]["_scores"]), clips) for m in eye}
    summary.insert(3, "ci_low", [ci.get(m, (np.nan, np.nan))[0] for m in summary["방법"]])
    summary.insert(4, "ci_high", [ci.get(m, (np.nan, np.nan))[1] for m in summary["방법"]])
    return summary


def pick(summary: pd.DataFrame) -> tuple[str, str]:
    """Val 요약 → (고른 모델, 이유). 차이가 시드 표준편차 이내면 1D-CNN."""
    models = summary[summary["방법"].isin(["cnn1d", "lstm"])].set_index("방법")
    first, second = models["눈감김_구간F1"].sort_values(ascending=False).index[:2]
    gap = models.loc[first, "눈감김_구간F1"] - models.loc[second, "눈감김_구간F1"]
    margin = max(models.loc[first, "시드_표준편차"], models.loc[second, "시드_표준편차"])
    if gap <= margin:
        return SIMPLER, f"{first}와 {second}의 차이 {gap:.4f}가 시드 표준편차 {margin:.4f} 이내라 {SIMPLER}를 고른다"
    return first, f"{first}가 {second}보다 {gap:.4f} 높고 시드 표준편차 {margin:.4f}를 넘는다"


def judge(model: pd.Series, rules: pd.DataFrame) -> tuple[bool, list[str]]:
    """고른 모델의 시드 평균 대 눈 규칙 두 가지 중 지표마다 나은 값. 두 조건을 모두 만족해야 향상이다."""
    best_f1, best_recall, best_alarm = rules["눈감김_구간F1"].max(), rules["3초_재현율"].max(), rules["3초_시간당_오경보"].min()
    f1_ok = model["눈감김_구간F1"] - best_f1 > model["시드_표준편차"]
    recall_ok = model["3초_재현율"] >= best_recall
    alarm_ok = model["3초_시간당_오경보"] < best_alarm
    notes = [f"1. 눈감김 구간 F1 {model['눈감김_구간F1']:.3f} 대 규칙 최고 {best_f1:.3f}, 차이 {model['눈감김_구간F1'] - best_f1:+.3f}, 시드 표준편차 {model['시드_표준편차']:.3f} → {'통과' if f1_ok else '실패'}",
             f"2. 3초 재현율 {model['3초_재현율']:.3f} 대 규칙 최고 {best_recall:.3f} → {'통과' if recall_ok else '실패'}",
             f"   3초 시간당 오경보 {model['3초_시간당_오경보']:.1f} 대 규칙 최저 {best_alarm:.1f} → {'통과' if alarm_ok else '실패'}"]
    return bool(f1_ok and recall_ok and alarm_ok), notes


def write(out: Path, detail: pd.DataFrame, summary: pd.DataFrame, head: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    detail.drop(columns="_scores").to_csv(out / "seeds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "report.md").write_text(f"{head}\n\n```\n{summary.round(3).to_string(index=False)}\n```\n", encoding="utf-8")
    print(head)
    print(summary.round(3).to_string(index=False))


SELECT_KEYS = ("주의_재현율", "주의_시간당_오경보", "눈감김_구간F1", "3초_재현율", "3초_시간당_오경보")
FINAL = config.RUNS / "final.json"
BASE_TAG = "v1"         # 오경보 기준을 정하는 tag (후처리 없음)
ALARM_FACTOR = 1.5      # 오경보는 기준의 이 배수까지 허용한다 (수정 기획서 F 반영, 사용자 결정 11.03)


def select(table: pd.DataFrame, limit: float) -> pd.Series:
    """README 선택 규칙: 1.5초 시간당 오경보가 limit 이하인 행 중 1.5초 재현율, 눈감김 구간 F1 순으로 높고, 같으면 _order가 작은 행."""
    ok = table[table["주의_시간당_오경보"] <= limit + 1e-9]
    return ok.sort_values(["주의_재현율", "눈감김_구간F1", "_order"], ascending=[False, False, True]).iloc[0]


def tune_post(tag: str, with_webcam: bool = False, base_tag: str = BASE_TAG) -> None:
    """Val로 후처리 후보 12개를 채점하고 고른 설정을 runs/<tag>/post.json에 쓴다. 모델은 pick.txt 그대로다."""
    from sequence_model.postprocess import FILE_NAME, save

    base = config.RUNS / tag
    chosen = (base / "pick.txt").read_text(encoding="utf-8").splitlines()[0]
    clips = selection_clips(with_webcam)
    runs_table = pd.read_csv(base / "experiments.csv", encoding="utf-8-sig")
    rows = []
    for row in runs_table[runs_table["모델"] == chosen].itertuples(index=False):
        for post, scores in model_scores(clips, Predictor(Path(row.weights)), CANDIDATES).items():
            rows.append({"_order": CANDIDATES.index(post), "설정": post.label(), **method_row(chosen, row.seed, scores, clips)})
    seeds = pd.DataFrame(rows)
    summary = seeds.drop(columns=["방법", "시드"]).groupby(["_order", "설정"]).mean().reset_index()
    summary.insert(2, "시드_표준편차", seeds.groupby("_order")["눈감김_구간F1"].std(ddof=1).to_numpy())
    none = summary[summary["_order"] == 0].iloc[0]
    # 오경보 기준 = base_tag 후처리 없음 × ALARM_FACTOR (README). base_tag를 먼저 돌려야 한다
    if tag == base_tag:
        base_alarm = float(none["주의_시간당_오경보"])
    else:
        base_alarm = json.loads((config.RUNS / base_tag / "post.json").read_text(encoding="utf-8"))["base_val"]["주의_시간당_오경보"]
    limit = base_alarm * ALARM_FACTOR
    best = select(summary, limit)
    post = CANDIDATES[int(best["_order"])]
    where = "Val+Val2" if with_webcam else "Val"
    reason = (f"{where} 1.5초 오경보 {best['주의_시간당_오경보']:.1f} ≤ 기준 {limit:.1f}({base_tag} 후처리 없음 {base_alarm:.1f} × {ALARM_FACTOR}), "
              f"1.5초 재현율 {none['주의_재현율']:.3f} → {best['주의_재현율']:.3f}, 눈감김 구간 F1 {none['눈감김_구간F1']:.3f} → {best['눈감김_구간F1']:.3f}")
    save(base / FILE_NAME, chosen, post, {"val": {k: float(best[k]) for k in SELECT_KEYS}, "base_val": {k: float(none[k]) for k in SELECT_KEYS},
                                         "reason": reason, "selection": where, "base_tag": base_tag})
    out = base / ("post_val2" if with_webcam else "post_val")
    out.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(out / "seeds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    head = f"{where} 후처리 선택 ({tag}, {chosen}). 고른 설정: {post.label()}. {reason}. 선택에 쓴 데이터라 성능 수치가 아니다."
    (out / "report.md").write_text(f"{head}\n\n```\n{summary.round(3).to_string(index=False)}\n```\n", encoding="utf-8")
    print(head)
    print(summary.round(3).to_string(index=False))


def tune_field(tag: str, field: str, with_webcam: bool = False) -> None:
    """post.json의 후처리 위에서 하품 판정(yawn_mode) 또는 이어받기 상한(carry_max) 하나만 바꿔 골라 post.json에 더한다.
    yawn_mode: 하품 구간 F1이 가장 높은 것. carry_max: 1.5초·3초 재현율이 상한 없음보다 낮지 않은 것 중 1.5초 오경보가 가장 적은 것."""
    from dataclasses import replace

    from sequence_model.postprocess import CARRY_CAPS, FILE_NAME, YAWN_MODES, from_dict

    base = config.RUNS / tag
    data = json.loads((base / FILE_NAME).read_text(encoding="utf-8"))
    post0, chosen = from_dict(data), data["model"]
    values = YAWN_MODES if field == "yawn_mode" else CARRY_CAPS
    posts = [replace(post0, **{field: v}) for v in values]
    clips = selection_clips(with_webcam)
    runs_table = pd.read_csv(base / "experiments.csv", encoding="utf-8-sig")
    rows = []
    for row in runs_table[runs_table["모델"] == chosen].itertuples(index=False):
        for i, (post, scores) in enumerate(model_scores(clips, Predictor(Path(row.weights)), posts).items()):
            rows.append({"_order": i, "설정": str(values[i]), **method_row(chosen, row.seed, scores, clips)})
    seeds = pd.DataFrame(rows)
    summary = seeds.drop(columns=["방법", "시드"]).groupby(["_order", "설정"]).mean().reset_index()
    if field == "yawn_mode":
        best = summary.sort_values(["하품_구간F1", "_order"], ascending=[False, True]).iloc[0]
    else:
        none = summary[summary["_order"] == 0].iloc[0]
        ok = summary[(summary["주의_재현율"] >= none["주의_재현율"] - 1e-9) & (summary["3초_재현율"] >= none["3초_재현율"] - 1e-9)]
        best = ok.sort_values(["주의_시간당_오경보", "_order"], ascending=[True, True]).iloc[0]
    value = values[int(best["_order"])]
    data[field] = value
    data["val"] = {k: float(best[k]) for k in SELECT_KEYS}
    data[f"{field}_reason"] = f"{'Val+Val2' if with_webcam else 'Val'}에서 {field}={value} 선택 (README \"하품 판정과 이어받기 상한\")"
    (base / FILE_NAME).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    out = base / f"post_{field}"
    out.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(out / "seeds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    cols = ["설정", "하품_구간F1", "눈감김_구간F1", "주의_재현율", "주의_시간당_오경보", "3초_재현율", "3초_시간당_오경보"]
    print(f"{tag} {chosen} {field} 선택: {value}")
    print(summary[cols].round(3).to_string(index=False))


def final(tags: list[str]) -> None:
    """각 tag의 post.json(Val 값) → README 규칙으로 하나를 고른다. 오경보 기준은 첫 tag의 후처리 없는 값이다."""
    from dataclasses import asdict
    from sequence_model.postprocess import from_dict

    entries, posts = [], {}
    for i, tag in enumerate(tags):
        data = json.loads((config.RUNS / tag / "post.json").read_text(encoding="utf-8"))
        posts[tag] = from_dict(data)
        entries.append({"_order": i, "tag": tag, "model": data["model"], "설정": posts[tag].label(), **data["val"]})
    base_alarm = json.loads((config.RUNS / tags[0] / "post.json").read_text(encoding="utf-8"))["base_val"]["주의_시간당_오경보"]
    table = pd.DataFrame(entries)
    best = select(table, base_alarm * ALARM_FACTOR)
    result = {"tag": best["tag"], "model": best["model"], **asdict(posts[best["tag"]]), "base_alarm": base_alarm,
              "table": table.drop(columns="_order").to_dict(orient="records")}
    FINAL.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(table.drop(columns="_order").round(3).to_string(index=False))
    print(f"고른 것: {result['tag']} {result['model']} {posts[best['tag']].label()} "
          f"(오경보 기준 {base_alarm * ALARM_FACTOR:.1f} = {base_alarm:.1f} × {ALARM_FACTOR}) → {FINAL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 라벨로 모델과 규칙을 채점한다")
    parser.add_argument("mode", choices=("val", "test", "post", "yawn", "carry", "final"))
    parser.add_argument("--tag")
    parser.add_argument("--tags", nargs="+", help="final: 비교할 tag. 첫 tag가 기준이다")
    parser.add_argument("--with-webcam", action="store_true", help="val·post: NITYMED Val에 Val2(웹캠 녹화)를 더해 채점한다")
    parser.add_argument("--base-tag", default=BASE_TAG, help="post: 오경보 기준을 정하는 tag (후처리 없음)")
    args = parser.parse_args()
    if args.mode == "final":
        final(args.tags or ["v1", "v2"])
        return
    if not args.tag:
        raise SystemExit("--tag가 필요합니다")
    base = config.RUNS / args.tag
    if args.mode == "post":
        tune_post(args.tag, args.with_webcam, args.base_tag)
        return
    if args.mode in ("yawn", "carry"):
        tune_field(args.tag, "yawn_mode" if args.mode == "yawn" else "carry_max", args.with_webcam)
        return

    if args.mode == "val":
        detail = score_split("val", args.tag, args.with_webcam)
        summary = summarize(detail)
        chosen, reason = pick(summary)
        where = "Val+Val2" if args.with_webcam else "Val"
        (base / "pick.txt").write_text(f"{chosen}\n{reason}\n", encoding="utf-8")
        write(base / ("eval_val2" if args.with_webcam else "eval_val"), detail, summary,
              f"{where}. 고른 모델: {chosen}. {reason}. 선택에 쓴 데이터라 성능 수치가 아니다.")
        return
    report = base / "eval_test" / "report.md"
    if report.exists():
        raise SystemExit(f"{report}가 있습니다. Test는 한 번만 채점합니다")
    chosen = (base / "pick.txt").read_text(encoding="utf-8").splitlines()[0]
    detail = score_split("test", args.tag)
    summary = summarize(detail)
    better, notes = judge(summary.set_index("방법").loc[chosen], summary[summary["방법"].isin(EYE_RULES)])
    verdict = "규칙 대비 향상이다." if better else "규칙 대비 향상이 아니다."
    write(base / "eval_test", detail, summary, f"Test. Val이 고른 모델: {chosen}.\n\n판정: {verdict}\n\n" + "\n".join(notes))


if __name__ == "__main__":
    main()

# 창별 모델 확률 → 켜짐·꺼짐 → 연속 눈 감김 타이머 → 눈 감김 지표 (기술서 3.1, 7절).
# 입력은 창별 확률 (N,4) .npy, 또는 clip·start_sec·y가 함께 든 .npz다. 정답과 시각은 windows.npz에서 읽는다.
#
# 실행: python -m eval.evaluate --windows <windows.npz> --probs <확률.npy> [--split val]
#       python -m eval.evaluate --windows <windows.npz> --probs val_probs.npz --probs-key probs_v01
#       python -m eval.evaluate --windows <windows.npz> --probs val_probs.npz --probs-key all --out-dir runs/eval_v8_check
#       python -m eval.evaluate ... --split test --final

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import (_first_alert, alert_delay, classification_metrics, event_recall, false_alarms_per_hour,
                          match_segments, prf, segments_from_labels)
from rule_based import config
from rule_based.alarm import CAUTION, DROWSY, alert_times
from rule_based.common import runs
from sequence_model.build_windows import CLASS_NAMES, EYE_CLOSED, STRIDE_SEC

LABEL_TOLERANCE_SEC = 0.5   # 구간 라벨 오차
ALERT_TOLERANCE_SEC = 0.5   # 창 간격 0.5초가 해상도 한계라서 0.5초로 둔다. Euro NCAP 허용 오차 0.2초는 참고로만 본다
MIN_CLOSE_SEC = 0.1         # 기술서 3.1. 감김 길이가 창 수 × 0.5초라 가장 짧아도 0.5초이므로 0.5초 간격에서는 효과가 없다

CLIP_GAP_SEC = 10.0         # 전체 타임라인에서 클립 사이 간격. 허용 오차보다 커서 다른 클립의 경고와 맞붙지 않는다
SUM_TOLERANCE = 1e-3        # 확률 행 합이 1에서 벗어나도 되는 범위. float32로 저장한 확률의 반올림 오차를 허용한다
ALL_KEYS = "all"            # --probs-key 값. probs_로 시작하는 키를 모두 평가한다
EVENT_COLUMNS = ["probs_key", "clip", "kind", "level", "start_sec", "end_sec", "length_sec", "alert_sec", "delay_sec", "result"]
LIMITS = ("0.5초 간격이라 경고 시각이 실제 감김 시작과 최대 0.5초 어긋나고, 지연 수치에 그게 포함된다.",)


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """머리글과 행 → 마크다운 표. 소수는 셋째 자리까지, None과 NaN은 '-'."""
    def cell(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return "\n".join(lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows])


def load_rows(windows_path: Path, split: str) -> dict:
    """windows.npz에서 해당 분할 행의 clip, start_sec, y만 읽는다. 다른 분할 행은 남기지 않는다."""
    with np.load(windows_path, allow_pickle=False) as data:
        keep = data["split"] == split
        rows = {"clip": data["clip"][keep], "start_sec": data["start_sec"][keep], "y": data["y"][keep]}
    if not len(rows["y"]):
        raise SystemExit(f"{windows_path}에 split={split} 행이 없습니다")
    return rows


def _time_order(clip: np.ndarray, start_sec: np.ndarray, source: str) -> np.ndarray:
    """클립·시작 시각 순서. 같은 클립에 같은 시각이 두 번 있으면 멈춘다."""
    order = np.lexsort((start_sec, clip))
    same = (clip[order][1:] == clip[order][:-1]) & np.isclose(start_sec[order][1:], start_sec[order][:-1])
    if same.any():
        i = order[np.flatnonzero(same)[0]]
        raise SystemExit(f"{source}: 같은 창이 두 번 있습니다 ({clip[i]}, {start_sec[i]}초)")
    return order


def _check_probs(probs: np.ndarray, n_rows: int, source: str) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES):
        raise SystemExit(f"{source}: 확률 모양은 (N, {len(CLASS_NAMES)})여야 합니다: {probs.shape}")
    if len(probs) != n_rows:
        raise SystemExit(f"{source}: 확률 {len(probs)}행, windows.npz {n_rows}행으로 행 수가 다릅니다")
    if not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
        raise SystemExit(f"{source}: 확률은 0~1 사이의 유한값이어야 합니다. 로짓을 저장한 것은 아닌지 확인하세요")
    if not np.isclose(probs.sum(axis=1), 1.0, atol=SUM_TOLERANCE).all():
        raise SystemExit(f"{source}: 확률의 행 합이 1이 아닙니다")
    return probs


def load_probs(path: Path, key: str | None, rows: dict) -> np.ndarray:
    """확률 파일 → rows와 같은 행 순서의 (N,4). .npy는 키가 없어 windows.npz 행 순서라고 가정한다."""
    if path.suffix == ".npy":
        return _check_probs(np.load(path, allow_pickle=False), len(rows["y"]), str(path))

    with np.load(path, allow_pickle=False) as data:
        names = [name for name in data.files if name.startswith("probs_")]
        if key not in data.files:
            raise SystemExit(f"{path}: --probs-key로 다음 중 하나를 고르세요: {names} (받은 값: {key})")
        missing = {"clip", "start_sec", "y", "classes"} - set(data.files)
        if missing:
            raise SystemExit(f"{path}: 키가 없습니다: {sorted(missing)}")
        clip, start_sec, y, classes, probs = (data[name] for name in ("clip", "start_sec", "y", "classes", key))
    # 팀원 파일은 "눈감김"처럼 띄어쓰기 없이 적을 수 있다. 띄어쓰기만 무시하고 순서는 그대로 검사한다
    if [str(c).replace(" ", "") for c in classes] != [name.replace(" ", "") for name in CLASS_NAMES]:
        raise SystemExit(f"{path}: 클래스 순서가 {list(CLASS_NAMES)}가 아닙니다: {[str(c) for c in classes]}")
    probs = _check_probs(probs, len(rows["y"]), f"{path}[{key}]")
    if not len(clip) == len(start_sec) == len(y) == len(probs):
        raise SystemExit(f"{path}: clip, start_sec, y, {key}의 행 수가 서로 다릅니다")

    # 파일 행 순서가 달라도 같은 창이면 맞춰 쓴다. 창이나 정답이 다르면 다른 데이터로 만든 확률이다
    file_order = _time_order(clip, start_sec, str(path))
    row_order = _time_order(rows["clip"], rows["start_sec"], "windows.npz")
    for name, same in (("clip", clip[file_order] == rows["clip"][row_order]),
                       ("start_sec", np.isclose(start_sec[file_order], rows["start_sec"][row_order])),
                       ("y", y[file_order] == rows["y"][row_order])):
        if not same.all():
            i = row_order[np.flatnonzero(~same)[0]]
            raise SystemExit(f"{path}: {name}이 windows.npz와 어긋납니다 (windows.npz 기준 {rows['clip'][i]}, {rows['start_sec'][i]}초)")
    aligned = np.empty_like(probs)
    aligned[row_order] = probs[file_order]
    return aligned


def closed_from_prob(prob: np.ndarray) -> np.ndarray:
    """눈 감김 확률 → 켜짐·꺼짐 적용 후 closed. 켜짐은 PROB_ON 이상, 꺼짐은 PROB_OFF 미만, 사이는 이전 상태를 유지한다.

    ProbToClosed는 PROB_OFF 이하에서 꺼져 기술서(0.4 미만)와 경계가 달라 쓰지 않는다.
    """
    closed = np.zeros(len(prob), dtype=bool)
    state = False
    for i, p in enumerate(prob):
        if p >= config.PROB_ON:
            state = True
        elif p < config.PROB_OFF:
            state = False
        closed[i] = state
    return closed


def drop_short_closures(closed: np.ndarray) -> np.ndarray:
    """MIN_CLOSE_SEC보다 짧은 감김을 지운다. 끝−시작으로 재면 창 1개가 0초가 되므로 창 수 × 간격으로 잰다."""
    out = closed.copy()
    for first, last in zip(*runs(closed)):
        if (last - first + 1) * STRIDE_SEC < MIN_CLOSE_SEC:
            out[first:last + 1] = False
    return out


def timer_alerts(t: np.ndarray, closed: np.ndarray) -> list[tuple[float, str]]:
    """연속 구간 하나 → 단계가 오른 (시각, 단계) 목록.

    타이머는 마지막 시각 − 첫 시각으로 재므로 창 시각만 넣으면 창 6개(3.0초)가 2.5초가 된다.
    정답 길이(창 수 × 간격)와 맞추려고 창 하나를 간격만큼의 칸으로 보고 칸 시작과 끝 시각을 모두 넣는다.
    """
    return alert_times(np.column_stack((t, t + STRIDE_SEC)).ravel(), np.repeat(closed, 2))


def continuous_parts(start_sec: np.ndarray) -> list[np.ndarray]:
    """시간순 시작 시각 → 연속 구간별 행 번호. 간격이 STRIDE_SEC가 아니면 제외된 창이 있으므로 끊는다."""
    breaks = np.flatnonzero(~np.isclose(np.diff(start_sec), STRIDE_SEC)) + 1
    return np.split(np.arange(len(start_sec)), breaks)


def _is_false_alarm(alert: float, long_events: list[tuple[float, float]]) -> bool:
    return not any(e[0] <= alert <= e[1] + ALERT_TOLERANCE_SEC for e in long_events)


def level_metrics(events: list[tuple[float, float]], alerts: np.ndarray, threshold: float) -> dict:
    """정답 이벤트와 한 단계의 경고 시각 → 재현율, 지연, 오경보 수. 시각은 모두 전체 타임라인 기준이다."""
    long_events = [e for e in events if e[1] - e[0] >= threshold]
    delays = [hit - (e[0] + threshold) for e in long_events
              if (hit := _first_alert(e, alerts, ALERT_TOLERANCE_SEC)) is not None]
    return {
        "threshold_sec": threshold,
        "n_events": len(long_events),
        "recall": event_recall(long_events, alerts, ALERT_TOLERANCE_SEC),
        "delay_mean": alert_delay(events, alerts, threshold, ALERT_TOLERANCE_SEC),
        "delay_median": float(np.median(delays)) if delays else float("nan"),
        "delay_max": float(np.max(delays)) if delays else float("nan"),
        "n_alerts": len(alerts),
        "false_alarms": int(sum(_is_false_alarm(a, long_events) for a in alerts)),
    }


def event_rows(true_spans: list[tuple[float, float]], alerts: dict, clips: list[tuple[str, float, float]]) -> list[dict]:
    """정답 이벤트와 오경보 → events.csv 행. 전체 타임라인 시각을 클립 안 시각으로 되돌린다. clips: (클립, 시각 차, 타임라인 끝)."""
    def locate(time: float) -> tuple[str, float]:
        return next((name, base) for name, base, end in clips if time <= end)

    levels = ((CAUTION, config.CAUTION_SEC), (DROWSY, config.DROWSY_SEC))
    rows = []
    for span in true_spans:
        name, base = locate(span[0])
        common = {"clip": name, "kind": "정답 이벤트", "start_sec": span[0] - base, "end_sec": span[1] - base,
                  "length_sec": span[1] - span[0]}
        reached = [(level, threshold) for level, threshold in levels if span[1] - span[0] >= threshold]
        if not reached:
            rows.append({**common, "result": "기준 미만"})
        for level, threshold in reached:
            hit = _first_alert(span, np.array(alerts[level]), ALERT_TOLERANCE_SEC)
            rows.append({**common, "level": level, "result": "놓침"} if hit is None else
                        {**common, "level": level, "alert_sec": hit - base, "delay_sec": hit - (span[0] + threshold), "result": "맞힘"})
    for level, threshold in levels:
        long_events = [e for e in true_spans if e[1] - e[0] >= threshold]
        for alert in alerts[level]:
            if _is_false_alarm(alert, long_events):
                name, base = locate(alert)
                inside = any(e[0] <= alert <= e[1] + ALERT_TOLERANCE_SEC for e in true_spans)
                rows.append({"clip": name, "kind": "오경보", "level": level, "alert_sec": alert - base,
                             "result": "기준보다 짧은 감김" if inside else "정답 감김 밖"})
    return sorted(rows, key=lambda r: (r["clip"], r.get("start_sec", r.get("alert_sec"))))


def evaluate(clip: np.ndarray, start_sec: np.ndarray, y: np.ndarray, probs: np.ndarray) -> dict:
    """창 정보와 확률 → 지표."""
    return evaluate_detail(clip, start_sec, y, probs)[0]


def evaluate_detail(clip: np.ndarray, start_sec: np.ndarray, y: np.ndarray, probs: np.ndarray) -> tuple[dict, list[dict]]:
    """창 정보와 확률 → (지표, events.csv 행). 행 순서는 상관없다. 클립마다 시각을 밀어 한 타임라인에 올린 뒤 eval.metrics를 그대로 쓴다."""
    order = _time_order(clip, start_sec, "windows.npz")
    clip, start_sec, y, probs = clip[order], start_sec[order], y[order], probs[order]
    truth = y == EYE_CLOSED

    true_spans, pred_spans, alerts = [], [], {CAUTION: [], DROWSY: []}
    predicted = np.zeros(len(y), dtype=bool)
    clips = []
    offset = 0.0
    for name in np.unique(clip):
        rows = np.flatnonzero(clip == name)
        t = start_sec[rows] - start_sec[rows[0]] + offset
        for part in continuous_parts(start_sec[rows]):
            closed = drop_short_closures(closed_from_prob(probs[rows[part], EYE_CLOSED]))
            predicted[rows[part]] = closed
            true_spans += segments_from_labels(t[part], truth[rows[part]], STRIDE_SEC, (True,))[True]
            pred_spans += segments_from_labels(t[part], closed, STRIDE_SEC, (True,))[True]
            for time, level in timer_alerts(t[part], closed):
                alerts[level].append(time)
        clips.append((str(name), t[0] - start_sec[rows[0]], t[-1] + STRIDE_SEC))
        offset = t[-1] + STRIDE_SEC + CLIP_GAP_SEC

    tp, fp, fn = match_segments(true_spans, pred_spans, LABEL_TOLERANCE_SEC)
    window_tp = int(np.count_nonzero(truth & predicted))
    window_fp, window_fn = int(np.count_nonzero(~truth & predicted)), int(np.count_nonzero(truth & ~predicted))
    caution = level_metrics(true_spans, np.array(alerts[CAUTION]), config.CAUTION_SEC)
    drowsy = level_metrics(true_spans, np.array(alerts[DROWSY]), config.DROWSY_SEC)

    open_sec = np.count_nonzero(~truth) * STRIDE_SEC    # 오경보 분모: 평가 전체 시간 − 정답 눈 감김 시간
    def per_hour(count: int) -> float:
        return false_alarms_per_hour(np.zeros(count), open_sec / 3600) if open_sec else float("nan")

    caution["false_alarms_per_hour"] = per_hour(caution["false_alarms"])
    drowsy["false_alarms_per_hour"] = per_hour(drowsy["false_alarms"])
    total = caution["false_alarms"] + drowsy["false_alarms"]
    argmax = classification_metrics(y, probs.argmax(axis=1))
    return {
        "n_windows": len(y),
        "n_clips": int(len(np.unique(clip))),
        "total_sec": len(y) * STRIDE_SEC,
        "open_sec": open_sec,
        "n_true_events": len(true_spans),
        "argmax": {"macro_f1": argmax["macro_f1"], "accuracy": argmax["accuracy"],
                   "f1_per_class": {CLASS_NAMES[c]: f1 for c, f1 in argmax["f1_per_class"].items()}},
        "span": {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)},
        "window": {"tp": window_tp, "fp": window_fp, "fn": window_fn, **prf(window_tp, window_fp, window_fn)},
        "caution": caution,
        "drowsy": drowsy,
        "false_alarms_total": {"count": total, "per_hour": per_hour(total)},
        "settings": {
            "LABEL_TOLERANCE_SEC": LABEL_TOLERANCE_SEC, "ALERT_TOLERANCE_SEC": ALERT_TOLERANCE_SEC,
            "MIN_CLOSE_SEC": MIN_CLOSE_SEC, "PROB_ON": config.PROB_ON, "PROB_OFF": config.PROB_OFF,
            "CAUTION_SEC": config.CAUTION_SEC, "DROWSY_SEC": config.DROWSY_SEC, "STRIDE_SEC": STRIDE_SEC,
        },
        "limits": list(LIMITS),
    }, event_rows(true_spans, alerts, clips)


def to_report(result: dict, title: str) -> str:
    """지표 → 마크다운 보고서."""
    levels = [("주의", result["caution"]), ("졸음", result["drowsy"])]
    settings = ", ".join(f"{name}={value:g}" for name, value in result["settings"].items())
    total = result["false_alarms_total"]
    return "\n\n".join([
        f"# {title}",
        f"창 {result['n_windows']}개, 클립 {result['n_clips']}개, 전체 {result['total_sec']:g}초, "
        f"오경보 분모 {result['open_sec']:g}초, 정답 이벤트 {result['n_true_events']}개",
        f"argmax macro F1 {result['argmax']['macro_f1']:.3f} (타이머를 거치지 않은 창 단위 4클래스 값)",
        markdown_table(["눈 감김 F1", "F1", "정밀도", "재현율", "맞힘", "잘못 냄", "놓침"],
                       [[name, m["f1"], m["precision"], m["recall"], m["tp"], m["fp"], m["fn"]]
                        for name, m in (("구간", result["span"]), ("창", result["window"]))]),
        markdown_table(["단계", "기준(초)", "이벤트 수", "재현율", "지연 평균", "지연 중앙값", "지연 최대", "경고 수", "오경보 수", "오경보/시간"],
                       [[name, m["threshold_sec"], m["n_events"], m["recall"], m["delay_mean"], m["delay_median"],
                         m["delay_max"], m["n_alerts"], m["false_alarms"], m["false_alarms_per_hour"]] for name, m in levels]
                       + [["합계", None, None, None, None, None, None, None, total["count"], total["per_hour"]]]),
        f"설정: {settings}",
        "한계: " + " ".join(result["limits"]),
    ]) + "\n"


SUMMARY_COLUMNS = {
    "argmax macro F1": lambda r: r["argmax"]["macro_f1"],
    "구간 F1": lambda r: r["span"]["f1"],
    "창 F1": lambda r: r["window"]["f1"],
    "3초 재현율": lambda r: r["drowsy"]["recall"],
    "1.5초 재현율": lambda r: r["caution"]["recall"],
    "졸음 지연 평균": lambda r: r["drowsy"]["delay_mean"],
    "주의 지연 평균": lambda r: r["caution"]["delay_mean"],
    "오경보 수": lambda r: r["false_alarms_total"]["count"],
    "오경보/시간": lambda r: r["false_alarms_total"]["per_hour"],
}


def summarize(results: dict) -> dict:
    """키별 지표 → 표 머리글별 평균·표준편차(모표준편차). 값이 없는 키는 그 열에서 뺀다."""
    table = {header: np.array([get(r) for r in results.values()], dtype=float) for header, get in SUMMARY_COLUMNS.items()}
    return {stat: {header: float(fn(values)) if not np.isnan(values).all() else float("nan") for header, values in table.items()}
            for stat, fn in (("mean", np.nanmean), ("std", np.nanstd))}


def to_all_report(title: str, results: dict, summary: dict) -> str:
    """키 여러 개 → 요약표와 키별 상세."""
    first = next(iter(results.values()))
    rows = [[key, *[get(r) for get in SUMMARY_COLUMNS.values()]] for key, r in results.items()]
    rows += [[name, *summary[stat].values()] for name, stat in (("평균", "mean"), ("표준편차", "std"))]
    return "\n\n".join([
        f"# {title}",
        f"3초 이벤트 {first['drowsy']['n_events']}개, 1.5초 이벤트 {first['caution']['n_events']}개, "
        f"오경보 분모 {first['open_sec']:g}초. 3초 이벤트가 적고 시간이 짧아 재현율과 시간당 오경보는 값이 크게 흔들린다.",
        markdown_table(["키", *SUMMARY_COLUMNS], rows),
        *[to_report(r, key).replace("# ", "## ", 1).strip() for key, r in results.items()],
    ]) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_all(args: argparse.Namespace, rows: dict) -> dict:
    """.npz의 probs_ 키를 모두 평가해 report.md, metrics.json, events.csv를 만든다. 모두 평가한 뒤에 저장하고 덮어쓰지 않는다."""
    if args.probs.suffix != ".npz":
        raise SystemExit(f"--probs-key {ALL_KEYS}는 .npz에만 쓸 수 있습니다: {args.probs}")
    paths = {name: args.out_dir / name for name in ("report.md", "metrics.json", "events.csv")}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise SystemExit(f"결과가 이미 있습니다. 다른 --out-dir를 쓰세요: {existing}")
    with np.load(args.probs, allow_pickle=False) as data:
        keys = sorted(name for name in data.files if name.startswith("probs_"))
    if not keys:
        raise SystemExit(f"{args.probs}에 probs_로 시작하는 키가 없습니다")

    results, events = {}, []
    for key in keys:
        probs = load_probs(args.probs, key, rows)
        results[key], key_events = evaluate_detail(rows["clip"], rows["start_sec"], rows["y"], probs)
        events += [{"probs_key": key, **row} for row in key_events]
    summary = summarize(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = to_all_report(f"{args.probs.name} 평가 ({args.split})", results, summary)
    paths["report.md"].write_text(report, encoding="utf-8")
    paths["metrics.json"].write_text(json.dumps(_json_safe({
        "probs": args.probs.name, "windows": args.windows.name, "split": args.split,
        "sha256": {"probs": _sha256(args.probs), "windows": _sha256(args.windows)},
        "summary": summary, "results": results}), ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(events, columns=EVENT_COLUMNS).to_csv(paths["events.csv"], index=False, encoding="utf-8-sig")
    print(report)
    print(f"저장: {', '.join(str(p) for p in paths.values())}")
    return results


def _json_safe(value):
    """NaN은 JSON 표준에 없어 null로 바꾼다."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return None if isinstance(value, float) and np.isnan(value) else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="창별 확률로 눈 감김 지표를 계산한다")
    parser.add_argument("--windows", type=Path, required=True, help="windows.npz")
    parser.add_argument("--probs", type=Path, required=True, help="창별 확률 (N,4) .npy, 또는 clip·start_sec·y·classes가 든 .npz")
    parser.add_argument("--probs-key", help=f".npz에서 쓸 확률 키 (예: probs_v01). {ALL_KEYS}면 probs_ 키를 모두 평가해 report.md, metrics.json, events.csv를 만든다")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final", action="store_true", help="--split test를 허용한다. 최종 판정 때만 쓴다")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final:
        parser.error("--split test는 --final이 있어야 합니다. 모델 선택은 Val로 합니다")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    rows = load_rows(args.windows, args.split)
    if args.probs_key == ALL_KEYS:
        return run_all(args, rows)
    probs = load_probs(args.probs, args.probs_key, rows)
    result = evaluate(rows["clip"], rows["start_sec"], rows["y"], probs)

    name = "_".join(part for part in ("eval", args.probs.stem, args.probs_key, args.split) if part)
    report = to_report(result, name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(
        json.dumps(_json_safe({"probs": str(args.probs), "probs_key": args.probs_key, "split": args.split, **result}),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / f"{name}.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"저장: {args.out_dir / name}.json, .md")
    return result


if __name__ == "__main__":
    main()

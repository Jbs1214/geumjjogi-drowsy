# 사람 라벨과 AI Hub 자동 라벨 → 클립당 라벨 JSON 1개 (기획서 9번 양식) + 분할 파일 data/split.csv.
#   NITYMED (label_status reviewed): 재검수 segments, eye_check_segments.csv, 사람이 표시한 고개숙임(data/head_down)
#   AI Hub 089 (label_status auto): 2차 자동 라벨의 정상·하품만. 합쳐진 눈감김·고개숙임은 불확실이다
#   DMD는 dataset/dmd_convert.py가 만든다.
# 라벨을 새로 판단하지 않고 사람이 정한 규칙대로 옮긴다. 프레임별 y와 창 정답은 build_windows.py가 계산한다.
#
# 실행: python -m dataset.convert_labels [--skip-aihub]

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from sequence_model.build_windows import frame_states
from rule_based.common import runs

CHECK_ORDER = ("눈감음", "고개숙임", "하품", "정상")      # 작업표 값이 여럿이면 앞의 것을 쓴다
HEAD_STATES = {"고개숙임": "head", "고개숙임(눈 안 보임)": "hidden"}
NITYMED_RULES = (
    "eye_closed: 재검수 '눈감김·고개숙임'. eye_check_segments.csv 판정이 우선이다(눈감음 > 고개숙임 > 하품 > 정상). 사람이 '고개숙임(눈 안 보임)'으로 표시한 구간은 뺀다",
    "head_down: 사람이 표시한 '고개숙임'과 '고개숙임(눈 안 보임)'",
    "yawn: 재검수 '하품'. 감음만 끼어 끊긴 하품은 잇고, 경계는 0.5초 단위로 바깥쪽에 맞춘다. '손·팔 동작'과 닿으면 hand",
    "uncertain: 재검수 '불확실', '손·팔 동작'",
    "segments: 채널에서 만든 보기용. 겹치면 불확실 > 눈감김 > 고개숙임 > 하품 > 정상. 학습에 쓰지 않는다",
)
AIHUB_RULES = (
    "AI Hub 089 2차 자동 라벨. 사람이 프레임을 검수한 라벨이 아니다",
    "정상(0)과 하품(2)만 쓴다. 눈감김·고개숙임(1)이 합쳐져 있어 불확실로 넣는다. 원래 불확실(3)도 불확실이다",
    "하드 네거티브 클립은 전부 정상이다",
)


def check_value(text: str) -> str:
    """작업표 칸 → 값 하나. '눈감음,고개숙임'처럼 여럿이면 CHECK_ORDER에서 앞의 것을 고른다."""
    parts = {part.strip() for part in str(text).replace(".", ",").split(",") if part.strip()}
    for value in CHECK_ORDER:
        if value in parts:
            return value
    raise ValueError(f"작업표 값을 알 수 없습니다: {text}")


def inside(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t >= start) & (t < end)


def spans(mask: np.ndarray, fps: float) -> list[list[float]]:
    """프레임별 표시 → [시작 초, 끝 초] 목록. 끝은 마지막 프레임 다음 시각이다."""
    return [[round(s / fps, 3), round((e + 1) / fps, 3)] for s, e in zip(*runs(mask))]


def join_yawns(yawn: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """하품 사이가 모두 감음이면 하나로 잇는다. 하품 중 감음을 눈감김으로 표시해 끊긴 곳이다."""
    out = yawn.copy()
    starts, ends = runs(yawn)
    for end, start in zip(ends[:-1], starts[1:]):
        if eye[end + 1:start].all():
            out[end + 1:start] = True
    return out


def yawn_events(yawn: np.ndarray, hand: np.ndarray, fps: float) -> list[dict]:
    """프레임별 하품 → {start, end, hand}. 경계는 0.5초 단위로 바깥쪽에 맞추고 겹치면 합친다."""
    duration = len(yawn) / fps
    merged: list[list] = []
    for s, e in zip(*runs(yawn)):
        start = math.floor(s / fps / config.YAWN_STEP + 1e-9) * config.YAWN_STEP
        end = min(math.ceil((e + 1) / fps / config.YAWN_STEP - 1e-9) * config.YAWN_STEP, round(duration, 3))
        touches = bool(hand[max(s - 1, 0):min(e + 2, len(hand))].any())
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = merged[-1][2] or touches
        else:
            merged.append([start, end, touches])
    return [{"start": round(s, 3), "end": round(e, 3), "hand": h} for s, e, h in merged]


def view_segments(channels: dict[str, np.ndarray], fps: float) -> list[dict]:
    """채널 → 보기용 segments. 겹치면 불확실 > 눈감김 > 고개숙임 > 하품 > 정상."""
    names = {config.NORMAL: "정상", config.CLOSED: "눈감김", config.YAWN: "하품", config.HEAD: "고개숙임", config.UNCERTAIN: "불확실"}
    state = frame_states(channels)
    edges = np.r_[0, np.flatnonzero(np.diff(state)) + 1, len(state)]
    return [{"start": round(a / fps, 3), "end": round(b / fps, 3), "state": names[int(state[a])]} for a, b in zip(edges[:-1], edges[1:])]


def nitymed_masks(segments: list[dict], n: int, fps: float, checks: list[tuple[float, float, str]],
                  heads: list[dict]) -> dict[str, np.ndarray]:
    """재검수 segments, 작업표, 고개숙임 표시 → 프레임별 채널. hand는 손·팔 동작 표시다."""
    t = np.arange(n) / fps
    masks = {name: np.zeros(n, dtype=bool) for name in ("eye_closed", "yawn", "uncertain", "head_down", "hand")}
    for segment in segments:
        span = inside(t, segment["start"], segment["end"])
        state = segment["state"]
        if state == "눈감김·고개숙임":
            masks["eye_closed"] |= span
        elif state == "하품":
            masks["yawn"] |= span
        elif state == "불확실":
            masks["uncertain"] |= span
        elif state == "손·팔 동작":
            masks["uncertain"] |= span
            masks["hand"] |= span
    for start, end, value in checks:
        span = inside(t, start, end)
        if value == "눈감음":
            masks["eye_closed"] |= span
        else:
            masks["eye_closed"] &= ~span
        if value == "고개숙임":
            masks["head_down"] |= span      # 이전 폴더에서는 불확실로 뺐지만 이번에는 고개숙임 클래스가 있다
        if value == "하품":
            masks["yawn"] |= span
    for segment in heads:
        kind = HEAD_STATES.get(segment["state"])
        if kind is None:
            continue
        span = inside(t, segment["start"], segment["end"])
        masks["head_down"] |= span
        if kind == "hidden":
            masks["eye_closed"] &= ~span
    masks["yawn"] = join_yawns(masks["yawn"], masks["eye_closed"])
    return masks


def label_json(clip: str, source: str, status: str, driver: str, fps: float, masks: dict[str, np.ndarray], rules: tuple[str, ...]) -> dict:
    """프레임별 채널 → 기획서 9번 양식."""
    channels = {name: masks[name] for name in ("eye_closed", "yawn", "uncertain", "head_down")}
    return {"clip": clip, "source": source, "driver_id": driver, "fps": fps, "frames": int(len(masks["eye_closed"])),
            "conditions": {}, "rules": list(rules), "label_status": status,
            "eye_closed": spans(masks["eye_closed"], fps), "yawn": yawn_events(masks["yawn"], masks["hand"], fps),
            "uncertain": spans(masks["uncertain"], fps), "head_down": spans(masks["head_down"], fps),
            "segments": view_segments(channels, fps)}


def read_checks() -> dict[str, list[tuple[float, float, str]]]:
    table = pd.read_csv(config.EYE_CHECK, encoding="utf-8-sig")
    out: dict[str, list] = {}
    for row in table.itertuples(index=False):
        out.setdefault(row.clip, []).append((float(row.start), float(row.end), check_value(row.확인)))
    return out


def check_driver_split(table: pd.DataFrame) -> None:
    """운전자 번호가 비었거나 운전자가 분할을 넘으면 멈춘다."""
    if table["driver_id"].fillna("").eq("").any():
        raise SystemExit("운전자 번호가 없는 클립이 있습니다")
    mixed = table.groupby("driver_id")["split"].nunique()
    if (mixed > 1).any():
        raise SystemExit(f"분할을 넘는 운전자: {mixed[mixed > 1].index.tolist()}")


def convert_nitymed(require_heads: bool = True) -> pd.DataFrame:
    """NITYMED 41개 → 라벨 JSON. 분할 행 목록을 돌려준다."""
    split = pd.read_csv(config.OLD_SPLIT, encoding="utf-8-sig")[["clip", "kind", "split", "driver_id"]]
    check_driver_split(split)
    checks = read_checks()
    rows = []
    for folder in config.REVIEWED_DIRS:
        for path in sorted(folder.glob("*.json")):
            clip = path.stem
            features = config.FEATURE_DIR / f"{clip}.parquet"
            head_path = config.HEAD_DIR / f"{clip}.json"
            if require_heads and not head_path.exists():
                raise SystemExit(f"고개숙임 표시가 끝나지 않았습니다: {clip}")
            df = pd.read_parquet(features, columns=["t"])
            fps = round(1 / float(np.median(np.diff(df["t"]))), 3)
            heads = json.loads(head_path.read_text(encoding="utf-8"))["segments"] if head_path.exists() else []
            masks = nitymed_masks(json.loads(path.read_text(encoding="utf-8"))["segments"], len(df), fps, checks.get(clip, []), heads)
            info = split[split["clip"] == clip].iloc[0]
            label = label_json(clip, "nitymed", "reviewed", info["driver_id"], fps, masks, NITYMED_RULES)
            (config.LABEL_DIR / f"{clip}.json").write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
            rows.append({"clip": clip, "source": "nitymed", "driver_id": info["driver_id"], "split": info["split"], "kind": info["kind"]})
    return pd.DataFrame(rows)


def aihub_group(clip: str) -> str:
    match = re.match(r"SGA\d+(?=S\d+$)", clip)
    return match[0] if match else clip


def convert_aihub() -> pd.DataFrame:
    """AI Hub 089 → 라벨 JSON. 탑승자 B 그룹은 쓰지 않는다. 운전자 번호가 없어 SGA 그룹을 driver_id로 쓴다."""
    holdout = set(config.AIHUB_HOLDOUT.read_text(encoding="utf-8-sig").split())
    negatives = set(config.AIHUB_NEGATIVES.read_text(encoding="utf-8").split())
    rows = []
    for path in sorted(config.AIHUB_LABELS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        clip = path.stem.split("_")[0]
        group = aihub_group(clip)
        if group in holdout or not (config.AIHUB_FEATURES / f"{clip}.parquet").exists():
            continue
        y = np.asarray(data["y"], dtype=int)
        if clip in negatives:
            y = np.where(y == 3, 3, 0)
        masks = {"eye_closed": np.zeros(len(y), bool), "head_down": np.zeros(len(y), bool), "yawn": y == 2,
                 "uncertain": (y == 1) | (y == 3), "hand": np.zeros(len(y), bool)}
        label = label_json(clip, "aihub089", "auto", f"aihub_{group}", float(data.get("fps", 30)), masks, AIHUB_RULES)
        (config.LABEL_DIR / f"{clip}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        rows.append({"clip": clip, "source": "aihub089", "driver_id": f"aihub_{group}", "split": "train", "kind": "aihub"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨 JSON과 분할 파일을 만든다")
    parser.add_argument("--skip-aihub", action="store_true", help="AI Hub 변환을 건너뛴다 (이미 만들었을 때)")
    parser.add_argument("--aihub-only", action="store_true", help="AI Hub만 변환한다. 고개숙임 표시가 끝나기 전에 미리 만들 때 쓴다")
    args = parser.parse_args()

    config.LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.aihub_only:
        aihub = convert_aihub()
        aihub.to_csv(config.DATA / "split_aihub.csv", index=False, encoding="utf-8-sig")
        print(f"AI Hub 라벨 {len(aihub)}개, 그룹 {aihub['driver_id'].nunique()}개")
        return
    nitymed = convert_nitymed()
    if args.skip_aihub and (config.DATA / "split_aihub.csv").exists():
        aihub = pd.read_csv(config.DATA / "split_aihub.csv", encoding="utf-8-sig")
    else:
        aihub = convert_aihub()
    parts = [nitymed, aihub]
    split = pd.concat(parts, ignore_index=True)
    check_driver_split(split)
    split.to_csv(config.SPLIT, index=False, encoding="utf-8-sig")
    print(split.groupby(["source", "split"]).agg(클립=("clip", "size"), 운전자=("driver_id", "nunique")).to_string())


if __name__ == "__main__":
    main()

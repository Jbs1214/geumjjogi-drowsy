# DMD 졸음 세트 주석(OpenLABEL) → 클립당 라벨 JSON. 규칙은 기술서 5.4이고 라벨의 rules에 함께 적는다.
#   이미 있는 라벨은 덮어쓰지 않는다(사람이 고친 값이 들어 있다). 새 클립만 만들고 data/split.csv에 행을 더한다.
#   --check는 기존 라벨과 같은 규칙으로 다시 만들어 비교만 하고 아무것도 쓰지 않는다.
#
# 실행: python -m dataset.dmd_convert --split train
#       python -m dataset.dmd_convert --check
#       python -m dataset.dmd_convert --basis-b <저장 폴더>

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from rule_based.common import runs

ANN_SUFFIX = "_rgb_ann_drowsiness.json"
CLOSED = ("eyes_state/close", "eyes_state/closing")
OPENING = ("eyes_state/opening",)
UNCERTAIN = "eyes_state/undefined"
BASIS_B_RULE = "basis: B 기준. A 기준 eye_closed에 eyes_state/opening 프레임을 더했다. 평가 보고용이고 학습에는 A 기준을 쓴다"
YAWN, HAND_YAWN = "yawning/Yawning without hand", "yawning/Yawning with hand"
YAWN_STEP_SEC = 0.5
RULES = (
    "eye_closed: DMD eyes_state/close·closing 프레임 구간. 맞닿은 구간은 1개로 잇는다. opening은 뜸, blinks는 쓰지 않는다 (기술서 5.4 A 기준)",
    "yawn: DMD yawning. 프레임이 맞닿은 구간은 1회로 합치고 일부라도 손이 있으면 hand=true. 시작·끝은 가까운 0.5초로 반올림",
    "head_down: DMD에 라벨이 없어 비운다",
    "uncertain: DMD eyes_state/undefined 구간. MediaPipe 얼굴 미검출은 넣지 않는다(DMD가 눈 상태를 표시했으므로)",
    "conditions: lighting은 DMD에 조명 항목이 없어 weather 값을 대신 넣었다",
    "label_status: 변환 결과를 사람이 아직 보지 않았다",
)


def frame_mask(openlabel: dict, kinds: tuple[str, ...], n: int) -> np.ndarray:
    """주석 종류들 → 프레임별 표시."""
    mask = np.zeros(n, dtype=bool)
    for action in openlabel["actions"].values():
        if action["type"] in kinds:
            for interval in action.get("frame_intervals", []):
                mask[max(int(interval["frame_start"]), 0):min(int(interval["frame_end"]) + 1, n)] = True
    return mask


def spans(mask: np.ndarray, fps: float) -> list[list[float]]:
    """프레임별 표시 → [시작 초, 끝 초] 목록. 끝은 마지막 프레임 다음 시각이다."""
    return [[round(s / fps, 3), round((e + 1) / fps, 3)] for s, e in zip(*runs(mask))]


def nearest_half(seconds: float) -> float:
    return math.floor(seconds / YAWN_STEP_SEC + 0.5) * YAWN_STEP_SEC


def yawn_events(openlabel: dict, n: int, fps: float) -> list[dict]:
    """맞닿은 하품 프레임 → {start, end, hand} 목록. 끝은 마지막 프레임 다음 시각을 반올림한다."""
    hand = frame_mask(openlabel, (HAND_YAWN,), n)
    return [{"start": nearest_half(s / fps), "end": nearest_half((e + 1) / fps), "hand": bool(hand[s:e + 1].any())}
            for s, e in zip(*runs(frame_mask(openlabel, (YAWN, HAND_YAWN), n)))]


def driver_id(clip: str) -> str:
    """파일 이름의 운전자 번호. 이미 만든 라벨과 같은 형식이다."""
    match = re.match(r"g[A-Z]_(\d+)_s\d+_", clip)
    if not match:
        raise SystemExit(f"DMD 운전자를 알 수 없습니다: {clip}")
    return match[1]


def conditions(openlabel: dict) -> dict:
    driver = next(o for o in openlabel["objects"].values() if o["type"] == "driver")
    flags = {item["name"]: item["val"] for item in driver["object_data"].get("boolean", [])}
    texts = {item["name"]: item["val"] for c in openlabel["contexts"].values() for item in c["context_data"].get("text", [])}
    return {"glasses": bool(flags.get("glasses", False)), "lighting": texts.get("weather", "")}


def convert(clip: str, openlabel: dict, frames: int, fps: float) -> dict:
    """주석 → 라벨 JSON. frames는 특징의 프레임 수다."""
    return {"clip": clip, "source": "dmd", "driver_id": driver_id(clip), "fps": fps, "frames": frames,
            "conditions": conditions(openlabel), "rules": list(RULES), "label_status": "auto",
            "eye_closed": spans(frame_mask(openlabel, CLOSED, frames), fps), "yawn": yawn_events(openlabel, frames, fps),
            "uncertain": spans(frame_mask(openlabel, (UNCERTAIN,), frames), fps), "head_down": []}


def basis_b(label: dict, openlabel: dict) -> dict:
    """A 기준 라벨 → B 기준 라벨. eye_closed에 opening만 더하고 나머지(사람이 고친 값 포함)는 그대로 둔다."""
    n, fps = int(label["frames"]), float(label["fps"])
    t = np.arange(n) / fps
    closed = frame_mask(openlabel, OPENING, n)
    for start, end in label["eye_closed"]:
        closed |= (t >= start - 1e-9) & (t < end - 1e-9)
    return {**label, "eye_closed": spans(closed, fps), "rules": [*label["rules"], BASIS_B_RULE]}


def load(path: Path) -> tuple[str, dict, int, float]:
    """주석 파일 → (클립, OpenLABEL, 특징 프레임 수, FPS). 주석과 특징의 프레임 수가 다르면 멈춘다."""
    clip = path.name.removesuffix(ANN_SUFFIX)
    openlabel = json.loads(path.read_text(encoding="utf-8"))["openlabel"]
    t = pd.read_parquet(config.FEATURE_DIR / f"{clip}.parquet", columns=["t"])["t"].to_numpy()
    total = int(openlabel["streams"]["face_camera"]["stream_properties"]["total_frames"])
    if total != len(t):
        raise SystemExit(f"{clip}: 주석 {total}프레임, 특징 {len(t)}프레임")
    return clip, openlabel, len(t), round(1 / float(np.median(np.diff(t))), 3)


def compare(new: dict, old: dict) -> list[str]:
    """다시 만든 라벨과 기존 라벨의 다른 항목. rules는 클립별 수동 수정 기록이 있어 비교하지 않는다."""
    return [key for key in old if key != "rules" and new.get(key) != old[key]]


def main() -> None:
    parser = argparse.ArgumentParser(description="DMD 주석을 라벨 JSON으로 바꾼다")
    parser.add_argument("--ann", type=Path, default=config.DMD_ANN_DIR, help="*_rgb_ann_drowsiness.json을 찾을 폴더")
    parser.add_argument("--split", choices=("train", "val", "test"), help="새 클립의 분할")
    parser.add_argument("--check", action="store_true", help="기존 라벨과 비교만 한다")
    parser.add_argument("--basis-b", type=Path, help="기존 라벨에 opening을 더한 B 기준 라벨을 이 폴더에 쓴다")
    args = parser.parse_args()
    if not (args.check or args.split or args.basis_b):
        parser.error("--split, --check, --basis-b 중 하나가 필요합니다")
    if args.basis_b:
        args.basis_b.mkdir(parents=True, exist_ok=True)
        for path in sorted(args.ann.rglob(f"*{ANN_SUFFIX}")):
            clip = path.name.removesuffix(ANN_SUFFIX)
            label = json.loads((config.LABEL_DIR / f"{clip}.json").read_text(encoding="utf-8"))
            out = basis_b(label, json.loads(path.read_text(encoding="utf-8"))["openlabel"])
            (args.basis_b / f"{clip}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{clip}: 눈 감김 {len(label['eye_closed'])} → {len(out['eye_closed'])}개", flush=True)
        return

    rows = []
    for path in sorted(args.ann.rglob(f"*{ANN_SUFFIX}")):
        clip = path.name.removesuffix(ANN_SUFFIX)
        out = config.LABEL_DIR / f"{clip}.json"
        if args.check:
            if out.exists():
                old = json.loads(out.read_text(encoding="utf-8"))
                different = compare(convert(*load(path)), old)
                print(f"{clip}: {'같음' if not different else f'다름 {different}'}")
            continue
        if out.exists():
            print(f"{clip}: 라벨이 있어 건너뜀")
            continue
        label = convert(*load(path))
        out.write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append({"clip": clip, "source": "dmd", "driver_id": label["driver_id"], "split": args.split, "kind": "dmd"})
        print(f"{clip}: 눈 감김 {len(label['eye_closed'])}개, 하품 {len(label['yawn'])}개, 불확실 {len(label['uncertain'])}개", flush=True)
    if rows:
        split = pd.read_csv(config.SPLIT, encoding="utf-8-sig", dtype={"driver_id": str})
        added = pd.DataFrame(rows)
        if set(added["driver_id"]) & set(split["driver_id"]):
            raise SystemExit("이미 분할에 있는 운전자입니다. 운전자가 분할을 넘지 않게 확인하세요")
        pd.concat([split, added], ignore_index=True).to_csv(config.SPLIT, index=False, encoding="utf-8-sig")
        print(f"분할에 {len(rows)}개를 {args.split}로 더했습니다")


if __name__ == "__main__":
    main()

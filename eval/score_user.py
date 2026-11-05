# 사람 라벨(auto_label/user_review) → 라벨 JSON(기획서 9번 양식, data/labels) → 모델과 규칙 채점.
#   --set user:  사용자 영상 2개. 참고용이다(runs/<tag>/eval_user). 판정 기준에 넣지 않는다
#   --set test2: 새 평가 데이터. runs/final.json이 고른 모델로 한 번만 채점해 판정한다(runs/eval_test2)
# 영상은 split.csv에 없으므로 학습·Val·Test에 섞이지 않는다. 결과로 모델이나 임계값을 바꾸지 않는다.
#
# 실행: python -m eval.score_user --tag v1
#       python -m eval.score_user --set test2

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from sequence_model.postprocess import from_dict
from dataset.convert_labels import inside, join_yawns, label_json
from eval.evaluate import EYE_RULES, FINAL, feature_path, judge, load_clip, score_clips, summarize, write

DRIVER = "user01"
MIN_EVENTS = 20         # Test2의 3초 이상 눈감김이 이보다 적으면 예비 결과다 (README)
USER_RULES = (
    "사용자가 찍은 영상에 사람이 auto_label/user_review로 상태 하나씩 표시했다. 겹치면 불확실 > 눈감김 > 고개숙임 > 하품 > 정상",
    "eye_closed: '눈감김'. head_down: '고개숙임'과 '고개숙임(눈 안 보임)'. yawn: '하품'(눈감김만 끼어 끊긴 하품은 잇는다). uncertain: '불확실'",
    "참고용 채점에만 쓴다. 학습·Val·Test에 쓰지 않는다",
)
CHANNEL_OF = {"눈감김": "eye_closed", "하품": "yawn", "고개숙임": "head_down", "고개숙임(눈 안 보임)": "head_down", "불확실": "uncertain"}


def user_masks(segments: list[dict], n: int, fps: float) -> dict[str, np.ndarray]:
    """사람이 표시한 상태 구간 → 프레임별 채널."""
    t = np.arange(n) / fps
    masks = {name: np.zeros(n, dtype=bool) for name in ("eye_closed", "yawn", "uncertain", "head_down", "hand")}
    for segment in segments:
        channel = CHANNEL_OF.get(segment["state"])
        if channel:
            masks[channel] |= inside(t, segment["start"], segment["end"])
    masks["yawn"] = join_yawns(masks["yawn"], masks["eye_closed"])
    return masks


def convert(clip: str, label_dir: Path = config.USER_LABEL_DIR, source: str = "user", driver: str = DRIVER) -> dict:
    segments = json.loads((label_dir / f"{clip}.json").read_text(encoding="utf-8"))["segments"]
    t = pd.read_parquet(feature_path(clip)[0], columns=["t"])["t"].to_numpy()
    fps = round(1 / float(np.median(np.diff(t))), 4)
    return label_json(clip, source, "reviewed", driver, fps, user_masks(segments, len(t), fps), USER_RULES)


def prepare(set_name: str) -> list[dict]:
    """세트의 사람 라벨을 9번 양식으로 바꿔 data/labels에 쓰고 load_clip 결과를 돌려준다."""
    from auto_label.user_review.server import video_set

    if set_name == "webcam":        # Val2는 영상이 없는 PC에서도 채점한다: 완료 저장된 라벨이 기준이다
        label_dir = config.WEBCAM_LABEL_DIR
        clips = [p.stem for p in sorted(label_dir.glob("*.json"))]
    else:
        videos, label_dir, _ = video_set(set_name)
        clips = [Path(p).stem for p in videos]
    missing = [c for c in clips if not (label_dir / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"완료 저장된 라벨이 없습니다: {missing}. python -m auto_label.user_review.server --set {set_name}으로 표시하세요")
    no_features = [c for c in clips if not feature_path(c)[0].exists()]
    if no_features:
        raise SystemExit(f"특징이 없습니다: {no_features}. python -m auto_label.extract --videos <영상 폴더>로 뽑으세요")
    source, driver = {"user": ("user", DRIVER), "webcam": ("webcam", DRIVER)}.get(set_name, ("test2", "test2"))
    for clip in clips:
        label = convert(clip, label_dir, source=source, driver=driver)
        (config.LABEL_DIR / f"{clip}.json").write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
    return [load_clip(c) for c in clips]


def describe(loaded: list[dict]) -> tuple[float, dict, dict]:
    seconds = sum(len(c["df"]) / c["fps"] for c in loaded)
    events = {sec: sum(len(c["events"][sec]) for c in loaded) for sec in (3.0, 1.5)}
    counts = {name: sum(len(c["truth"][name]) for c in loaded) for name in ("눈감김", "하품", "고개숙임")}
    return seconds, events, counts


def score_test2() -> None:
    """Test2 최종 판정. runs/final.json이 고른 tag·모델·후처리로 한 번만 채점한다."""
    out = config.RUNS / "eval_test2"
    if (out / "report.md").exists():
        raise SystemExit(f"{out / 'report.md'}가 있습니다. Test2는 한 번만 채점합니다")
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    post = from_dict(final)
    loaded = prepare("test2")
    seconds, events, counts = describe(loaded)
    detail = score_clips(loaded, final["tag"], post=post, models=(final["model"],))
    summary = summarize(detail)
    better, notes = judge(summary.set_index("방법").loc[final["model"]], summary[summary["방법"].isin(EYE_RULES)])
    verdict = "규칙 대비 향상이다." if better else "규칙 대비 향상이 아니다."
    if events[3.0] < MIN_EVENTS:
        verdict += f" 3초 이상 눈감김이 {events[3.0]}개로 {MIN_EVENTS}개보다 적어 예비 결과다."
    head = (f"Test2 ({len(loaded)}개, {seconds:.1f}초). 모델: {final['tag']} {final['model']}, 후처리 {post.label()}.\n\n판정: {verdict}\n\n"
            + "\n".join(notes) + f"\n\n정답 구간 수: 눈감김 {counts['눈감김']}, 하품 {counts['하품']}, 고개숙임 {counts['고개숙임']}. "
            f"3초 이상 눈감김 {events[3.0]}개, 1.5초 이상 {events[1.5]}개.")
    write(out, detail, summary, head)


def yawn_lag(loaded: list[dict], tag: str, model: str, post) -> float:
    """모델 하품 판정이 겹친 정답 하품의 끝보다 얼마나 늦게 끝나는지(초, 중앙값). 대표 시드(_v01)로 잰다."""
    from rule_based.common import runs
    from eval.evaluate import load_predictor, model_frames

    predictor, lags = load_predictor(config.RUNS / tag / f"{model}_v01.pt"), []
    for clip in loaded:
        t, k = model_frames(clip["df"], predictor, post, clip["team"])
        for s, e in zip(*runs(k == config.YAWN)):
            end = t[e] + 1 / config.FPS
            ends = [te for ts, te in clip["truth"]["하품"] if ts < end and te > t[s]]
            if ends:
                lags.append(end - max(ends))
    return float(np.median(lags)) if lags else float("nan")


def score_webcam(tags: list[str]) -> None:
    """Val2(웹캠 녹화) 채점. tag마다 자기 post.json의 모델·후처리로 채점하고 규칙 4개와 함께 낸다. 여러 번 다시 채점해도 된다."""
    loaded = prepare("webcam")
    seconds, events, counts = describe(loaded)
    parts, lags = [], {}
    for i, tag in enumerate(tags):
        data = json.loads((config.RUNS / tag / "post.json").read_text(encoding="utf-8"))
        post, model = from_dict(data), data["model"]
        part = score_clips(loaded, tag, post=post, models=(model,))
        if i:
            part = part[part["방법"] == model]          # 규칙은 한 번만
        part = part.copy()
        part.loc[part["방법"] == model, "방법"] = f"{tag} {model}"
        parts.append(part)
        lags[f"{tag} {model}"] = yawn_lag(loaded, tag, model, post)
    names = set(lags)
    for part in parts:
        part.attrs = {}         # 클립 데이터(attrs)는 합친 뒤 한 번만 넣는다. pandas가 합칠 때 attrs를 비교하다 실패한다
    models = [p[p["방법"].isin(names)] for p in parts]
    rules = parts[0][~parts[0]["방법"].isin(names)]
    detail = pd.concat(models + [rules], ignore_index=True)
    detail.attrs["clips"] = loaded
    summary = summarize(detail)
    summary["하품_끝_지연_초"] = summary["방법"].map(lags)
    head = (f"Val2 웹캠 녹화 {len(loaded)}개 ({seconds / 60:.1f}분, 사람 1명). 결과를 본 데이터라 비교·선택에 쓰고 최종 판정에는 쓰지 않는다. "
            f"모델마다 자기 후처리(post.json)를 쓴다.\n\n정답 구간 수: 눈감김 {counts['눈감김']}, 하품 {counts['하품']}, 고개숙임 {counts['고개숙임']}. "
            f"3초 이상 눈감김 {events[3.0]}개, 1.5초 이상 {events[1.5]}개. 하품_끝_지연_초는 모델 하품이 정답 하품보다 늦게 끝난 시간(중앙값)이다.")
    write(config.RUNS / "eval_webcam", detail, summary, head)


def main() -> None:
    parser = argparse.ArgumentParser(description="사용자 영상(참고용), Val2 웹캠 녹화(비교용), Test2(최종 판정)를 사람 라벨로 채점한다")
    parser.add_argument("--set", default="user", choices=("user", "webcam", "test2"))
    parser.add_argument("--tag", help="user 세트에서 채점할 tag")
    parser.add_argument("--tags", nargs="+", default=["v1", "v4", "v6"], help="webcam 세트에서 비교할 tag")
    args = parser.parse_args()
    if args.set == "test2":
        score_test2()
        return
    if args.set == "webcam":
        score_webcam(args.tags)
        return
    if not args.tag:
        raise SystemExit("--tag가 필요합니다")
    loaded = prepare("user")
    clips = [c["clip"] for c in loaded]
    seconds, events, counts = describe(loaded)
    base = config.RUNS / args.tag
    chosen = (base / "pick.txt").read_text(encoding="utf-8").splitlines()[0]
    detail = score_clips(loaded, args.tag)
    summary = summarize(detail)
    head = (f"사용자 영상 {len(clips)}개 (사람 1명, {seconds:.1f}초). 참고용이며 판정 기준이 아니다. Val이 고른 모델: {chosen}.\n\n"
            f"정답 구간 수: 눈감김 {counts['눈감김']}, 하품 {counts['하품']}, 고개숙임 {counts['고개숙임']}. "
            f"3초 이상 눈감김 {events[3.0]}개, 1.5초 이상 {events[1.5]}개. 구간 수가 적어 하나가 수치를 크게 바꾼다.")
    write(base / "eval_user", detail, summary, head)


if __name__ == "__main__":
    main()

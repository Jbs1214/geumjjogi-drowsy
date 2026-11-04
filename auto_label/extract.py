# 영상 → 프레임별 특징 parquet (원본 FPS). 라벨이 있는 NITYMED 41개와 사람 촬영 영상을 처리한다. CPU로 돌린다.
# 프로세스를 나눠 병렬로 처리하고, 있는 파일은 건너뛰어 중단 후 다시 실행할 수 있다.
#
# 실행: python -m auto_label.extract [--workers 6]

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2

import config


def init_worker() -> None:
    cv2.setNumThreads(1)


def process(job: tuple[Path, Path]) -> tuple[str, int, float, float, str]:
    """영상 하나 → (이름, 프레임 수, 원본 FPS, 얼굴 검출률, 에러)."""
    from landmarks.extract import extract_video
    from landmarks.features import add_features

    path, out_dir = job
    try:
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        df = add_features(extract_video(path))
        if df.empty:
            return path.stem, 0, fps, 0.0, "프레임 없음"
        out = out_dir / f"{path.stem}.parquet"
        temp = out.with_suffix(".tmp")
        df.to_parquet(temp, index=False)
        temp.replace(out)
        return path.stem, len(df), fps, float(df["face_found"].mean()), ""
    except Exception as error:
        return path.stem, 0, 0.0, 0.0, f"{type(error).__name__}: {error}"


def targets() -> list[Path]:
    """처리할 영상. 재검수 라벨이 있는 NITYMED 영상과 사람 촬영 영상이다."""
    labeled = {p.stem for folder in config.REVIEWED_DIRS for p in folder.glob("*.json")}
    videos = [p for p in sorted(config.NITYMED_DIR.rglob("*.mp4")) if p.stem in labeled]
    if len(videos) != len(labeled):
        raise SystemExit(f"라벨 {len(labeled)}개 중 영상 {len(videos)}개만 찾았습니다")
    return videos + ([config.USER_VIDEO] if config.USER_VIDEO.exists() else [])


def main() -> None:
    parser = argparse.ArgumentParser(description="영상에서 프레임별 특징을 뽑는다")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--videos", type=Path, help="이 폴더의 mp4만 처리한다 (예: Test2 폴더)")
    args = parser.parse_args()

    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in config.FEATURE_DIR.glob("*.parquet")}
    videos = sorted(args.videos.rglob("*.mp4")) if args.videos else targets()
    todo = [p for p in videos if p.stem not in done]
    print(f"처리할 영상 {len(todo)}개, 이미 있음 {len(done)}개, 프로세스 {args.workers}개", flush=True)
    began = time.perf_counter()
    log = config.FEATURE_DIR / "extract_log.jsonl"
    with mp.Pool(args.workers, initializer=init_worker) as pool, log.open("a", encoding="utf-8") as out:
        for i, (name, frames, fps, found, error) in enumerate(pool.imap_unordered(process, [(p, config.FEATURE_DIR) for p in todo]), 1):
            out.write(json.dumps({"clip": name, "frames": frames, "fps": fps, "face_rate": round(found, 4), "error": error}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {name}: {frames}프레임, {fps:.2f} FPS, 검출률 {found:.3f} {error}", flush=True)
    print(f"끝: {time.perf_counter() - began:.0f}초", flush=True)


if __name__ == "__main__":
    main()

# 사용자가 찍은 영상에 정답 상태를 사람이 표시하는 로컬 도구. 고개숙임 표시 도구(auto_label/head_review)의 서버를 그대로 쓰고
# 상태 목록, 화면, 저장 위치만 바꾼다. 이 PC 안에서만 열리고 얼굴 영상을 외부로 보내지 않는다. 모델·규칙 판정은 화면에 주지 않는다.
# 완료한 영상은 data/user_labels/<영상>.json, 초안은 data/user_labels_drafts/에 저장한다. 채점은 eval/score_user.py가 한다.
#
# 실행: python -m auto_label.user_review.server [--set user|test2] 후 브라우저에서 http://127.0.0.1:8765

import argparse
from pathlib import Path

import cv2

import config
from auto_label.head_review.server import HOST, PORT, ReviewStore, make_server

# 한 시점에 상태 하나. 겹치면 우선순위(불확실 > 눈감김 > 고개숙임 > 하품 > 정상)가 높은 쪽을 고른다
STATE_BY_NAME = {"정상": 0, "눈감김": 1, "하품": 2, "고개숙임": 3, "고개숙임(눈 안 보임)": 4, "불확실": 5}
PAGE = Path(__file__).with_name("review.html")


def video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return round(float(fps), 4) if fps and fps > 0 else 30.0


def video_set(name: str) -> tuple[list[Path], Path, Path]:
    """세트 이름 → (영상 목록, 완료 폴더, 초안 폴더). user는 이미 본 영상 2개, test2는 새 평가 데이터 폴더다."""
    if name == "user":
        return list(config.USER_VIDEOS), config.USER_LABEL_DIR, config.USER_LABEL_DRAFT_DIR
    if name == "webcam":
        videos = sorted(config.WEBCAM_DIR.glob("*.mp4"))
        if not videos:
            raise SystemExit(f"{config.WEBCAM_DIR}에 mp4가 없습니다. python -m demo.webcam --record webcam으로 녹화하세요")
        return videos, config.WEBCAM_LABEL_DIR, config.WEBCAM_LABEL_DRAFT_DIR
    if name == "test2":
        seen = {Path(p).stem for p in config.USER_VIDEOS}
        videos = [p for p in sorted(config.TEST2_DIR.glob("*.mp4")) if p.stem not in seen]
        if not videos:
            raise SystemExit(f"{config.TEST2_DIR}에 mp4가 없습니다")
        return videos, config.TEST2_LABEL_DIR, config.TEST2_LABEL_DRAFT_DIR
    raise SystemExit(f"없는 세트: {name}")


def user_store(videos=config.USER_VIDEOS, done_dir: Path = config.USER_LABEL_DIR, draft_dir: Path = config.USER_LABEL_DRAFT_DIR) -> ReviewStore:
    paths = {Path(p).stem: Path(p) for p in videos if Path(p).exists()}
    missing = [str(p) for p in videos if not Path(p).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    return ReviewStore(list(paths), done_dir=done_dir, draft_dir=draft_dir, paths=paths, states=STATE_BY_NAME,
                       fps={clip: video_fps(path) for clip, path in paths.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="사용자 영상 정답 표시 도구")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--set", default="user", choices=("user", "webcam", "test2"), help="표시할 영상 묶음")
    args = parser.parse_args()
    store = user_store(*video_set(args.set))
    server = make_server(store, args.port, PAGE)
    print(f"사용자 영상 정답 표시 도구: http://{HOST}:{args.port}  영상 {len(store.order)}개 (끝내려면 Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

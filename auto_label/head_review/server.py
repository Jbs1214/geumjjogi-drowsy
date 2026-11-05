# 고개숙임 구간을 사람이 표시하는 로컬 도구. drowsy_all_in_one/tools/review_server.py를 복사해 상태와 저장 위치만 바꿨다.
# 이 PC 안에서만 열리고 얼굴 영상을 외부로 보내지 않는다. 기존 라벨이나 모델 값은 화면에 주지 않는다.
# 완료한 클립은 data/head_down/<클립>.json, 작업 중인 초안은 data/head_down_drafts/에 저장한다. 기준은 README "고개숙임 표시 기준"에 있다.
#
# 실행: python -m auto_label.head_review.server 후 브라우저에서 http://127.0.0.1:8765

import argparse
import json
import random
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import config

STATE_BY_NAME = {"숙이지 않음": 0, "고개숙임": 1, "고개숙임(눈 안 보임)": 2}
HOST = "127.0.0.1"
PORT = 8765
PAGE = Path(__file__).with_name("review.html")
DONE_DIR = config.HEAD_DIR
DRAFT_DIR = config.HEAD_DRAFT_DIR
ORDER_SEED = 42
CHUNK = 1 << 20
MAX_BODY = 1 << 20
ROUTE = re.compile(r"^/(api/label|video)/([^/\\]+?)(\.mp4)?$")     # 이름은 검수 목록에 있는 것만 통과시킨다


def review_order(clips: list[str], seed: int = ORDER_SEED) -> list[str]:
    """클립 목록 → 섞은 순서. 뽑힌 이유가 순서로 드러나지 않게 하고, 다시 열어도 순서가 같다."""
    order = sorted(clips)
    random.Random(seed).shuffle(order)
    return order


def clean_segments(segments: list[dict], states: dict[str, int] = STATE_BY_NAME) -> list[dict]:
    """화면에서 받은 구간을 검사한다. 0초부터 빈틈 없이 이어져야 하고 상태는 states에 있어야 한다."""
    if not isinstance(segments, list) or not segments:
        raise ValueError("구간이 없습니다")
    out = []
    previous = 0.0
    for segment in segments:
        start, end, state = float(segment["start"]), float(segment["end"]), segment["state"]
        if state not in states:
            raise ValueError(f"없는 상태: {state}")
        if end <= start or abs(start - previous) > 1e-3:
            raise ValueError(f"구간이 이어지지 않습니다: {start}~{end}")
        out.append({"start": round(start, 3), "end": round(end, 3), "state": state})
        previous = end
    return out


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Range 헤더 → (시작, 끝) 바이트. 없거나 형식이 다르면 None. 끝을 넘으면 파일 끝으로 자른다."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header or "")
    if not match or not (match[1] or match[2]):
        return None
    if not match[1]:
        start, end = max(size - int(match[2]), 0), size - 1
    else:
        start, end = int(match[1]), min(int(match[2]) if match[2] else size - 1, size - 1)
    return (start, end) if start <= end else None


class ReviewStore:
    """검수 대상과 저장 파일을 다룬다."""

    def __init__(self, clips: list[str], raw_dir: Path = config.NITYMED_DIR, done_dir: Path = DONE_DIR, draft_dir: Path = DRAFT_DIR,
                 paths: dict[str, Path] | None = None, states: dict[str, int] = STATE_BY_NAME,
                 fps: dict[str, float] | None = None) -> None:
        """paths는 클립 → 영상 경로다. 하위 폴더에 나뉜 공개 데이터에 쓰고, 없으면 raw_dir에서 찾는다.
        states는 저장할 수 있는 상태, fps는 클립 → 원본 FPS다(화면의 1프레임 이동 폭)."""
        self.order = review_order(clips)
        self.allowed = set(clips)
        self.raw_dir, self.done_dir, self.draft_dir = raw_dir, done_dir, draft_dir
        self.paths = paths or {}
        self.states, self.fps = states, fps or {}
        done_dir.mkdir(parents=True, exist_ok=True)
        draft_dir.mkdir(parents=True, exist_ok=True)

    def video_path(self, clip: str) -> Path:
        return self.paths.get(clip, self.raw_dir / f"{clip}.mp4")

    def status(self, clip: str) -> str:
        if (self.draft_dir / f"{clip}.json").exists():
            return "draft"
        return "done" if (self.done_dir / f"{clip}.json").exists() else "todo"

    def listing(self) -> list[dict]:
        """화면에 줄 목록. 맹검을 위해 번호와 진행 상태만 준다."""
        return [{"clip": clip, "status": self.status(clip), **({"fps": self.fps[clip]} if clip in self.fps else {})} for clip in self.order]

    def load(self, clip: str) -> list[dict]:
        for folder in (self.draft_dir, self.done_dir):
            path = folder / f"{clip}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))["segments"]
        return []

    def save(self, clip: str, segments: list[dict], done: bool) -> None:
        """구간 저장. 완료하면 초안을 지운다. 쓰다 끊겨도 깨진 파일이 남지 않게 임시 이름으로 쓴 뒤 바꾼다."""
        cleaned = clean_segments(segments, self.states)
        path = (self.done_dir if done else self.draft_dir) / f"{clip}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"clip": clip, "source": "reviewed", "segments": cleaned}, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        if done:
            (self.draft_dir / f"{clip}.json").unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    store: ReviewStore
    page: Path = PAGE

    def log_message(self, *_) -> None:
        pass

    def allowed_host(self) -> bool:
        """주소가 이 PC인 요청만 받는다. 외부 웹페이지가 이름을 바꿔 들어오는 경우를 막는다."""
        return self.headers.get("Host", "").split(":")[0] in ("127.0.0.1", "localhost")

    def send_json(self, value, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def route(self) -> tuple[str, str] | None:
        """요청 경로 → (종류, 클립). 검수 목록에 없는 클립은 None이다."""
        match = ROUTE.match(unquote(self.path.split("?")[0]))
        if not match or match[2] not in self.store.allowed or (match[1] == "video") != bool(match[3]):
            return None
        return match[1], match[2]

    def do_GET(self) -> None:
        if not self.allowed_host():
            self.send_error(HTTPStatus.FORBIDDEN)
        elif self.path == "/":
            body = self.page.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/clips":
            self.send_json(self.store.listing())
        else:
            target = self.route()
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            elif target[0] == "video":
                self.send_video(self.store.video_path(target[1]))
            else:
                self.send_json({"segments": self.store.load(target[1])})

    def send_video(self, path: Path) -> None:
        """영상 전송. 탐색이 되도록 Range 요청을 처리한다."""
        size = path.stat().st_size
        wanted = parse_range(self.headers.get("Range"), size)
        if self.headers.get("Range") and wanted is None:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = wanted or (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if wanted else HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if wanted:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = file.read(min(CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass        # 브라우저는 탐색할 때 연결을 끊는다

    def do_POST(self) -> None:
        target = self.route()
        length = int(self.headers.get("Content-Length") or 0)
        if not self.allowed_host() or self.headers.get("Content-Type") != "application/json":
            self.send_error(HTTPStatus.FORBIDDEN)
        elif target is None or target[0] != "api/label" or not 0 < length <= MAX_BODY:
            self.send_error(HTTPStatus.BAD_REQUEST)
        else:
            try:
                body = json.loads(self.rfile.read(length))
                self.store.save(target[1], body["segments"], bool(body.get("done")))
                self.send_json({"status": self.store.status(target[1])})
            except (KeyError, TypeError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def make_server(store: ReviewStore, port: int = PORT, page: Path = PAGE) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"store": store, "page": page})
    return ThreadingHTTPServer((HOST, port), handler)


def folder_store(folder: Path, done_dir: Path, draft_dir: Path, sample: int | None = None, seed: int = ORDER_SEED,
                 only: set[str] | None = None) -> ReviewStore:
    """폴더의 mp4를 검수 대상으로 만든다. only를 주면 그 이름만 쓰고, sample을 주면 시드 고정으로 그 수만큼 무작위로 고른다."""
    paths = {path.stem: path for path in sorted(folder.rglob("*.mp4")) if only is None or path.stem in only}
    clips = review_order(list(paths), seed)[:sample]
    return ReviewStore(clips, folder, done_dir, draft_dir, {clip: paths[clip] for clip in clips})


def main() -> None:
    parser = argparse.ArgumentParser(description="고개숙임 표시 도구")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    labeled = {p.stem for folder in config.REVIEWED_DIRS for p in folder.glob("*.json")}
    store = folder_store(config.NITYMED_DIR, DONE_DIR, DRAFT_DIR, only=labeled)
    server = make_server(store, args.port)
    print(f"고개숙임 표시 도구: http://{HOST}:{args.port}  클립 {len(store.order)}개 (끝내려면 Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

# 브라우저 웹캠으로 실시간 판정과 경고를 보여 주는 Streamlit 데모 (최종 기획서 4 유저 시나리오, 산출물). 로컬 시연용이다.
# 판정은 demo/webcam.py와 같은 엔진(LiveJudge·LiveRule·알림)을 쓴다. 모델은 runs/final.json이 고른 것이고, 하품은 최종 구성의 하품 전용 모델이다.
# 화면은 세 페이지다: 운전하기(웹캠 + 계기판, 끝나면 운전 후 리포트), 지난 운전(runs/sessions/ 판정 로그를 같은 리포트로 보기),
# 관람(발표자 PC의 시연을 다른 PC 브라우저에서 실시간으로 보기, 카메라 없음. 같은 네트워크에서 열려면 서버를 --server.address 0.0.0.0으로 켠다.
# 관람은 카메라를 쓰지 않아 http로 충분하고, 발표자는 http://localhost로 열면 웹캠이 켜진다).
# 프레임은 이 PC의 파이썬으로만 오간다. STUN 서버를 쓰지 않고 서버는 127.0.0.1에만 연다 (.streamlit/config.toml). 영상은 저장하지 않는다.
#
# 실행: drowsy_v3 폴더에서 python -m streamlit run demo/streamlit_app.py

import os
import sys
import threading
import time
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # streamlit은 스크립트 폴더를 경로로 잡는다

import numpy as np
import pandas as pd

import config
from rule_based.alarm import DROWSY, NORMAL, WARN
from rule_based.alerts import Event, MouthClosedWatch, YawnTracker, summary_markdown

REFRESH_SEC = 0.1            # 계기판을 다시 그리는 간격(초). 길면 계기판이 영상보다 늦어 보인다
CHART_SEC = 2.0             # 실시간 그래프를 다시 그리는 간격(초). 깜빡임을 줄이려고 계기판보다 드물게 그린다
LIVE_CHART_SEC = 300.0      # 실시간 그래프에 보여 줄 최근 구간(초). 긴 운전에서 2초마다 전체 기록을 다시 그리지 않게 한다
VIEW_EVERY_SEC = 0.1        # 관람 페이지로 보내는 프레임 간격(초). 관람 화면은 초당 약 10장으로 보인다
VIEW_WIDTH = 640            # 관람 페이지로 보내는 프레임 너비(픽셀)
VIEW_JPEG_QUALITY = 70      # 관람 프레임 JPEG 품질
VIEW_STALE_SEC = 3.0        # 마지막 프레임이 이보다 오래되면 관람 화면에 "영상이 멈춤"을 알린다
LOCAL_ONLY = {"iceServers": []}     # 같은 PC 안에서만 연결한다
CONSTRAINTS = {"video": {"width": 640, "height": 480, "frameRate": {"ideal": 30}}, "audio": False}
STAGE_NAMES = ("정상", "주의", "졸음")


class LiveSession:
    """프레임 하나를 추출 → 판정까지 처리한다(영상에는 그리지 않는다). 콜백 스레드와 화면 스레드가 같이 쓰므로 잠금을 건다.
    운전 한 번 = reset()부터 finish()까지. finish() 뒤에 들어오는 프레임은 버리고, 다음 운전은 reset()으로 새로 시작한다.
    모델·얼굴 인식기·경고음은 운전이 바뀌어도 그대로 쓴다(얼굴 인식기의 시각은 계속 늘어나야 한다)."""

    def __init__(self, weights: Path, sound: bool = True) -> None:
        from landmarks.extract import LandmarkExtractor
        from sequence_model.models import Predictor
        from sequence_model.postprocess import for_weights
        from sequence_model.train_yawn import load_yawn

        self.predictor = Predictor(weights)
        self.post = for_weights(weights, self.predictor.name)
        self.yawn_predictor = load_yawn(config.YAWN_WEIGHTS)
        self.weights = weights
        self.extractor = LandmarkExtractor()
        self.beeper = None
        if sound:
            from demo.webcam import Beeper
            self.beeper = Beeper()
        self.lock = threading.Lock()
        self.began: float | None = None
        self.mask = False
        self.broadcast: "Broadcast | None" = None     # 관람 페이지로 보낼 곳(drive_page가 넣는다)
        self._shared_at = 0.0
        self.reset()

    def reset(self) -> None:
        """새 운전: 판정 기록·타이머·처리 속도를 비운다."""
        from demo.webcam import LiveJudge, LiveRule

        with self.lock:
            self.judge = LiveJudge(self.predictor, self.post, self.yawn_predictor)
            self.judge.mask = self.mask
            self.rule = LiveRule(self.judge.events if self.judge.final else None)
            self.mouth, self.yawn = MouthClosedWatch(), YawnTracker()
            self.stamps: list[float] = []
            self.ear = self.mar = np.nan
            self.log: list[dict] = []               # 판정 로그(기술서 6.2), 프레임마다 한 행
            self._logged, self._was_found, self._carrying = 0, True, False
            self.finished: dict | None = None       # finish()가 저장한 마지막 기록. 있으면 운전이 끝난 상태다
        if self.beeper:
            self.beeper.set(False)

    def set_mask(self, on: bool) -> None:
        """마스크 착용: 하품 감지를 끈다(기술서 한계 항목, 화면 스위치)."""
        with self.lock:
            self.mask = self.judge.mask = on

    def process(self, frame: np.ndarray) -> None:
        from demo.webcam import alarm_stage, check_active, frame_vector

        now = time.perf_counter()
        self.began = self.began or now
        t = now - self.began
        raw = self.extractor.process(frame, t)
        vector = frame_vector(raw)
        found = bool(vector[config.FEATURES.index(config.FACE_COLUMN)] > 0.5)
        value = {name: float(vector[config.FEATURES.index(name)]) if found else np.nan for name in ("ear", "ear_l", "ear_r", "mar", "pitch")}
        with self.lock:
            if self.finished is not None:       # 끝난 운전에 늦게 도착한 프레임
                return
            self.judge.update(t, vector, self.mouth.update(t, value["mar"]), self.yawn.update(t, value["mar"]))
            self.rule.update(t, found, value["ear"], value["pitch"], value["ear_l"], value["ear_r"])
            self.ear, self.mar = value["ear"], value["mar"]
            self.stamps = (self.stamps + [time.perf_counter()])[-60:]
            self._log_frame(t, found)
            # 판정 값은 오른쪽 계기판에만 보인다. 영상 위에는 그리지 않아 운전자 얼굴을 가리지 않는다
            if self.beeper:
                self.beeper.set(alarm_stage(self.judge, self.rule) == DROWSY or check_active(self.judge, self.rule))
        # 관람 페이지(다른 PC 브라우저)로 최근 프레임 한 장을 보낸다. 메모리에만 두고 저장하지 않는다
        if self.broadcast is not None and now - self._shared_at >= VIEW_EVERY_SEC:
            self._shared_at = now
            jpeg = encode_view(frame)
            if jpeg is not None:
                self.broadcast.publish(self, jpeg)

    def _log_frame(self, t: float, found: bool) -> None:
        """잠금을 잡은 상태에서 부른다. 판정 로그 한 행(기술서 6.2): 실제 시각, 영상 시각, 단계, 연속 눈 감김, 현재 상태, 4클래스 확률, 이벤트.
        이벤트는 새 경고(주의·졸음·하품 주의·확인 필요)와 판정 불가 시작, 이어받기 시작이다."""
        from demo.webcam import alarm_stage, current_state

        j, r = self.judge, self.rule
        kinds = [e.kind for e in j.events[self._logged:]]
        self._logged = len(j.events)
        if self._was_found and not found:
            kinds.append("판정 불가")
        carrying = ((not found) or r.head_down) and r.closed_sec > 0
        if carrying and not self._carrying:
            kinds.append("이어받기")
        self._was_found, self._carrying = found, carrying
        probs = j.probs if j.probs is not None else np.full(len(config.CLASSES), np.nan)
        self.log.append({"wall_time": datetime.now().isoformat(timespec="milliseconds"), "video_t": round(t, 3),
                         "stage": STAGE_NAMES[alarm_stage(j, r)], "closed_sec": round(r.closed_sec if j.final else j.closed_sec, 3),
                         "state": current_state(j, r), **{f"p_{name}": round(float(p), 4) for name, p in zip(config.CLASSES, probs)},
                         "event": ";".join(kinds)})

    def _snapshot(self, history: bool = True) -> dict:
        """잠금을 잡은 상태에서 부른다. history=False면 판정 기록·이벤트 목록을 복사하지 않는다(계기판만 그릴 때).
        최종 구성이면 경고 단계와 연속 눈 감김은 규칙 쪽 값이다(webcam.alarm_stage와 같다)."""
        from demo.webcam import alarm_stage, check_active, current_state

        j, r = self.judge, self.rule
        rate = (len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0]) if len(self.stamps) > 1 else 0.0
        now = j.history[-1][0] if j.history else 0.0
        stage = alarm_stage(j, r)
        records, t0 = [], 0.0
        if history:
            # 실시간 그래프는 최근 LIVE_CHART_SEC만 복사한다. 운전 후 리포트는 판정 로그(self.log) 전체를 쓴다
            source = r.history if j.final else j.history
            t0 = source[0][0] if source else 0.0
            recent = source[bisect_left(source, source[-1][0] - LIVE_CHART_SEC, key=lambda h: h[0]):] if source else []
            records = [(t, -1, s, -1, closed) for t, s, closed in recent] if j.final else list(recent)
        model_state = config.CLASSES[j.shown] if j.shown >= 0 else ("판정 불가" if j.history else "준비 중")
        return {"state": current_state(j, r), "model_state": model_state, "stage": STAGE_NAMES[stage],
                "stage_index": stage, "yawn_alert": j.yawn_repeat.active(now), "check_alert": check_active(j, r),
                "ear": self.ear, "mar": self.mar, "final": j.final,
                "closed_sec": r.closed_sec if j.final else j.closed_sec, "rate": rate,
                "events": list(j.events) if history else [], "history": records, "t0": t0,
                "seconds": j.history[-1][0] - j.history[0][0] if j.history else 0.0}

    def snapshot(self, history: bool = True) -> dict:
        with self.lock:
            return self._snapshot(history)

    def finish(self) -> dict:
        """운전 종료: 이벤트 요약(.md)과 판정 로그(기술서 6.2, _log.csv)를 runs/sessions/에 한 번만 저장하고, 저장한 그 기록을 돌려준다.
        이미 끝난 운전이면 저장하지 않고 같은 기록을 돌려준다(화면이 다시 그려질 때). 영상은 저장하지 않는다."""
        with self.lock:
            if self.finished is not None:
                return self.finished
            snap = self._snapshot()
            self.finished = snap
            log = pd.DataFrame(self.log)
        if self.beeper:
            self.beeper.set(False)
        if snap["seconds"] > 0:
            # 얼굴 인식기 시각은 운전이 바뀌어도 이어지므로, 요약의 이벤트 시각은 이번 운전 시작(로그 첫 행)부터 센다
            t0 = float(log["video_t"].iloc[0]) if len(log) else 0.0
            events = [Event(e.t - t0, e.kind) for e in snap["events"]]
            text = summary_markdown(events, snap["seconds"], f"운전 후 요약 ({time.strftime('%Y-%m-%d %H:%M')}, Streamlit 데모)")
            stem = f"{time.strftime('%Y%m%d_%H%M%S')}_streamlit"
            out = config.RUNS / "sessions" / f"{stem}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            log.to_csv(out.with_name(f"{stem}_log.csv"), index=False, encoding="utf-8-sig")
        return snap


class Broadcast:
    """관람 페이지로 보낼 발표자 시연. 서버에 하나만 두고(broadcast()), 카메라를 켠 마지막 세션이 최근 프레임 한 장을 올린다.
    관람 페이지는 이 세션의 계기판 값과 프레임을 읽기만 한다. 프레임은 메모리에 한 장만 두고 파일로 저장하지 않는다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: LiveSession | None = None
        self._jpeg: bytes | None = None
        self._at = 0.0

    def publish(self, session: "LiveSession", jpeg: bytes) -> None:
        with self._lock:
            self._session, self._jpeg, self._at = session, jpeg, time.time()

    def latest(self) -> tuple["LiveSession | None", bytes | None, float]:
        """(발표 중인 세션, 최근 프레임 JPEG, 올린 시각 time.time()). 아직 없으면 (None, None, 0.0)."""
        with self._lock:
            return self._session, self._jpeg, self._at


def _new_broadcast() -> Broadcast:
    return Broadcast()


def broadcast() -> Broadcast:
    """서버에 하나인 방송 저장소. 모든 브라우저 탭(운전하기·관람)이 같이 쓴다."""
    import streamlit as st

    return st.cache_resource(show_spinner=False)(_new_broadcast)()


def encode_view(frame: np.ndarray) -> bytes | None:
    """BGR 프레임 → 관람용 JPEG(너비 VIEW_WIDTH 이하). 실패하면 None."""
    import cv2

    height, width = frame.shape[:2]
    if width > VIEW_WIDTH:
        frame = cv2.resize(frame, (VIEW_WIDTH, round(height * VIEW_WIDTH / width)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, VIEW_JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def fmt(value: float, digits: int = 2) -> str:
    return "-" if np.isnan(value) else f"{value:.{digits}f}"


def set_alarm(area, on: bool) -> None:
    """졸음 테두리(화면 가장자리 깜빡임)를 켜거나 끈다. 한 줄 조건식으로 쓰면 Streamlit 매직이 반환값을 화면에 찍으므로 if 문으로 쓴다."""
    from demo.cluster import ALARM_HTML

    if on:
        area.markdown(ALARM_HTML, unsafe_allow_html=True)
    else:
        area.empty()


def draw_cluster(area, snap: dict | None) -> None:
    """계기판 한 장. snap이 None이면 대기 화면이다(운전하기·관람이 같이 쓴다)."""
    from demo.cluster import cluster_html

    names = ("현재 상태", "모델 판정(비교)", "EAR", "MAR", "처리 속도", "운전 시간")
    if snap is None:
        area.markdown(cluster_html(0.0, NORMAL, "대기", {}, [(name, "-") for name in names]), unsafe_allow_html=True)
        return
    stage, seconds = snap["stage_index"], snap["seconds"]
    lit = {"주의": stage == WARN, "졸음": stage == DROWSY, "하품 주의": snap["yawn_alert"], "확인 필요": snap["check_alert"]}
    values = (snap["state"], snap["model_state"], fmt(snap["ear"]), fmt(snap["mar"]), f"{snap['rate']:.1f} FPS",
              f"{int(seconds // 60):02d}:{int(seconds % 60):02d}")
    area.markdown(cluster_html(snap["closed_sec"], stage, snap["stage"], lit, list(zip(names, values))), unsafe_allow_html=True)


def drive_page() -> None:
    """운전하기: 왼쪽 웹캠, 오른쪽 계기판. 카메라를 끄면 같은 자리에 운전 후 리포트를 보여 준다."""
    import av
    import streamlit as st
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from demo.charts import closure_chart
    from demo.report import live_frames, show_report
    from demo.webcam import default_weights

    # 브라우저 탭마다 자기 판정 상태를 쓴다. 서버 전체에서 하나를 나눠 쓰면 탭 두 개의 프레임과 로그가 섞인다
    if "drowsy_session" not in st.session_state:
        with st.spinner("모델과 얼굴 인식기를 불러오는 중입니다"):
            st.session_state["drowsy_session"] = LiveSession(default_weights())
    session = st.session_state["drowsy_session"]
    session.broadcast = broadcast()     # 관람 페이지가 이 탭의 시연을 볼 수 있게 한다
    if st.sidebar.button("새 운전 시작", help="지금까지의 판정 기록을 지우고 처음부터 잽니다", use_container_width=True):
        session.reset()
    session.set_mask(st.sidebar.checkbox("마스크 착용 (하품 감지 끄기)", value=session.mask))
    st.sidebar.caption(f"모델 {session.weights.parent.name}/{session.weights.name}  \n후처리 {session.post.label()}  \n"
                       f"하품 {config.YAWN_WEIGHTS.parent.name}/{config.YAWN_WEIGHTS.name}")

    st.title("졸음 계기판")
    st.caption("연속 눈 감김 1.5초면 주의, 3초면 졸음 경고. 하품 5분 3회면 하품 주의, 판정 불가·고개 숙임(눈 감김 이어받기 아님) 3초면 확인 필요. "
               "현재 상태는 규칙과 하품 전용 모델로 정하고, 4클래스 모델 판정은 비교용입니다. 연구용 데모라 실제 운전 보조로 쓰면 안 됩니다.")

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        session.process(frame.to_ndarray(format="bgr24"))
        return frame        # 영상은 받은 그대로 돌려준다

    left, right = st.columns([3, 2], gap="large")
    with left:
        context = webrtc_streamer(key="drowsy", mode=WebRtcMode.SENDRECV, rtc_configuration=LOCAL_ONLY,
                                  media_stream_constraints=CONSTRAINTS, video_frame_callback=callback, async_processing=True,
                                  translations={"start": "카메라 켜기", "stop": "운전 끝내기", "select_device": "카메라 고르기"})
    area, alarm = right.empty(), st.empty()
    chart = st.empty()

    if context.state.playing and session.finished is not None:     # 운전을 끝낸 뒤 카메라를 다시 켰다 = 새 운전
        session.reset()
    alarm_on, chart_at = False, 0.0
    while context.state.playing:
        now = time.perf_counter()
        draw_chart = now - chart_at >= CHART_SEC
        snap = session.snapshot(history=draw_chart)
        draw_cluster(area, snap)
        if (snap["stage_index"] == DROWSY) != alarm_on:        # 졸음 테두리는 바뀔 때만 다시 그린다(깜빡임 효과가 끊기지 않게)
            alarm_on = not alarm_on
            set_alarm(alarm, alarm_on)
        if draw_chart and snap["history"]:
            chart_at = now
            timeline = live_frames(snap)
            chart.altair_chart(closure_chart(timeline.iloc[::config.JUDGE_EVERY], timeline["시각"].iloc[-1], timeline["시각"].iloc[0]),
                               use_container_width=True)
        time.sleep(REFRESH_SEC)
    alarm.empty()
    draw_cluster(area, None)
    if session.snapshot(history=False)["seconds"] > 0 or session.finished is not None:
        snap = session.finish()
        chart.empty()
        st.divider()
        show_report(st, pd.DataFrame(session.log), note="방금 끝난 운전입니다. 기록은 지난 운전에서 다시 볼 수 있습니다.")
    else:
        chart.info("카메라 켜기를 누르면 브라우저가 카메라 권한을 묻습니다. 영상은 저장하지 않고, 관람 페이지를 연 브라우저에만 보입니다.")


def watch_page() -> None:
    """관람: 발표자 PC의 운전하기 탭에서 켠 시연(최근 프레임·계기판·실시간 그래프)을 다른 PC 브라우저에서 본다.
    이 페이지는 카메라를 쓰지 않고 판정도 하지 않는다. 발표자가 운전을 끝내면 같은 운전 후 리포트를 보여 준다."""
    import streamlit as st

    from demo.charts import closure_chart
    from demo.report import live_frames, show_report

    st.title("시연 관람")
    st.caption("발표자 PC에서 운전하기의 카메라를 켜면 그 영상과 계기판이 여기에 실시간으로 나옵니다. "
               "이 페이지는 카메라를 쓰지 않고, 영상은 저장하지 않습니다.")
    hub = broadcast()
    left, right = st.columns([3, 2], gap="large")
    picture, area = left.empty(), right.empty()
    alarm, chart, report = st.empty(), st.empty(), st.empty()
    alarm_on, chart_at, reported = False, 0.0, None
    while True:
        session, jpeg, at = hub.latest()
        if session is None or jpeg is None:
            picture.info("발표자 PC에서 카메라를 켜면 여기에 나옵니다.")
            draw_cluster(area, None)
            time.sleep(VIEW_EVERY_SEC)
            continue
        stale = time.time() - at > VIEW_STALE_SEC
        picture.image(jpeg, caption="발표자 영상이 멈췄습니다(카메라가 꺼졌거나 운전이 끝남)" if stale else None,
                      use_container_width=True)
        now = time.perf_counter()
        draw_chart = now - chart_at >= CHART_SEC
        snap = session.snapshot(history=draw_chart)
        draw_cluster(area, snap)
        drowsy = snap["stage_index"] == DROWSY and not stale
        if drowsy != alarm_on:              # 졸음 테두리는 바뀔 때만 다시 그린다
            alarm_on = drowsy
            set_alarm(alarm, alarm_on)
        finished = session.finished
        if finished is not None and reported is not finished:       # 발표자가 운전을 끝냈다: 같은 리포트를 한 번 그린다
            reported = finished
            chart.empty()
            with report.container():
                show_report(st, pd.DataFrame(list(session.log)), note="발표자가 방금 끝낸 운전입니다.")
        elif finished is None:
            if reported is not None:        # 새 운전이 시작됐다
                reported = None
                report.empty()
            if draw_chart and snap["history"]:
                chart_at = now
                timeline = live_frames(snap)
                chart.altair_chart(closure_chart(timeline.iloc[::config.JUDGE_EVERY], timeline["시각"].iloc[-1], timeline["시각"].iloc[0]),
                                   use_container_width=True)
        time.sleep(VIEW_EVERY_SEC)


def history_page() -> None:
    """지난 운전: runs/sessions/의 판정 로그를 골라 같은 리포트로 본다."""
    import streamlit as st

    from demo.report import load_log, log_label, log_title, session_logs, show_report

    st.title("지난 운전")
    logs = session_logs()
    if not logs:
        st.info("저장된 운전 기록이 없습니다. 운전하기에서 운전을 한 번 끝내면 여기에 쌓입니다.")
        return
    path = st.selectbox("운전 기록", logs, format_func=log_label, help="판정 로그(_log.csv)가 있는 기록만 보입니다")
    show_report(st, load_log(path), note=log_title(path))


def main() -> None:
    import streamlit as st

    from demo.cluster import CSS

    st.set_page_config(page_title="졸음 계기판", page_icon=":material/visibility:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    page = st.navigation([st.Page(drive_page, title="운전하기", icon=":material/directions_car:", default=True),
                          st.Page(history_page, title="지난 운전", icon=":material/history:", url_path="history"),
                          st.Page(watch_page, title="관람", icon=":material/live_tv:", url_path="watch")])
    page.run()


if __name__ == "__main__":
    main()

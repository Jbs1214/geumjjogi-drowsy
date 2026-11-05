# 브라우저 웹캠으로 실시간 판정과 경고를 보여 주는 Streamlit 데모 (최종 기획서 4 유저 시나리오, 산출물). 로컬 시연용이다.
# 판정은 demo/webcam.py와 같은 엔진(LiveJudge·LiveRule·알림)을 쓴다. 모델은 runs/final.json이 고른 것이다.
# 프레임은 이 PC의 파이썬으로만 오간다. STUN 서버를 쓰지 않고 서버는 127.0.0.1에만 연다 (.streamlit/config.toml). 영상은 저장하지 않는다.
#
# 실행: drowsy_v3 폴더에서 python -m streamlit run demo/streamlit_app.py

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # streamlit은 스크립트 폴더를 경로로 잡는다

import numpy as np

import config
from rule_based.alarm import DROWSY
from rule_based.alerts import MouthClosedWatch, YawnTracker, summary_markdown

REFRESH_SEC = 0.5
CHART_EVERY = 4             # REFRESH_SEC × 4 = 2초마다 그래프를 다시 그린다
LOCAL_ONLY = {"iceServers": []}     # 같은 PC 안에서만 연결한다
CONSTRAINTS = {"video": {"width": 640, "height": 480, "frameRate": {"ideal": 30}}, "audio": False}
STAGE_NAMES = ("정상", "주의", "졸음")


class LiveSession:
    """프레임 하나를 추출 → 판정 → 그리기까지 처리한다. 콜백 스레드와 화면 스레드가 같이 쓰므로 잠금을 건다."""

    def __init__(self, weights: Path, sound: bool = True) -> None:
        from landmarks.extract import LandmarkExtractor
        from sequence_model.models import Predictor
        from sequence_model.postprocess import for_weights
        from demo.webcam import Beeper, LiveJudge, LiveRule, load_yawn

        predictor = Predictor(weights)
        self.post = for_weights(weights, predictor.name)
        self.weights = weights
        self.judge = LiveJudge(predictor, self.post, load_yawn(config.YAWN_WEIGHTS))
        self.rule = LiveRule(self.judge.events if self.judge.final else None)
        self.mouth, self.yawn = MouthClosedWatch(), YawnTracker()
        self.extractor = LandmarkExtractor()
        self.beeper = Beeper() if sound else None
        self.lock = threading.Lock()
        self.stamps: list[float] = []
        self.began: float | None = None

    def process(self, frame: np.ndarray) -> None:
        from demo.webcam import alarm_stage, draw, frame_vector

        now = time.perf_counter()
        self.began = self.began or now
        t = now - self.began
        raw = self.extractor.process(frame, t)
        vector = frame_vector(raw)
        found = bool(vector[config.FEATURES.index(config.FACE_COLUMN)] > 0.5)
        value = {name: float(vector[config.FEATURES.index(name)]) if found else np.nan for name in ("ear", "ear_l", "ear_r", "mar", "pitch")}
        with self.lock:
            self.judge.update(t, vector, self.mouth.update(t, value["mar"]), self.yawn.update(t, value["mar"]))
            self.rule.update(t, found, value["ear"], value["pitch"], value["ear_l"], value["ear_r"])
            self.stamps = (self.stamps + [time.perf_counter()])[-60:]
            rate = (len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0]) if len(self.stamps) > 1 else 0.0
            draw(frame, self.judge, self.rule, value["ear"], value["mar"], rate)
            if self.beeper:
                self.beeper.set(alarm_stage(self.judge, self.rule) == DROWSY or self.judge.unknown_watch.active)

    def snapshot(self) -> dict:
        from demo.webcam import alarm_stage

        with self.lock:
            j = self.judge
            rate = (len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0]) if len(self.stamps) > 1 else 0.0
            return {"state": config.CLASSES[j.shown] if j.shown >= 0 else "준비 중", "stage": STAGE_NAMES[alarm_stage(j, self.rule)], "rule_stage": STAGE_NAMES[self.rule.stage],
                    "closed_sec": self.rule.closed_sec if j.final else j.closed_sec, "rate": rate, "events": list(j.events), "history": list(j.history),
                    "seconds": j.history[-1][0] - j.history[0][0] if j.history else 0.0}

    def finish(self) -> str:
        """운전 종료: 이벤트를 시간순으로 요약해 runs/sessions/에 저장하고 내용을 돌려준다(영상 없음)."""
        snap = self.snapshot()
        text = summary_markdown(snap["events"], snap["seconds"], f"운전 후 요약 ({time.strftime('%Y-%m-%d %H:%M')}, Streamlit 데모)")
        if snap["events"] or snap["seconds"] > 0:
            out = config.RUNS / "sessions" / f"{time.strftime('%Y%m%d_%H%M%S')}_streamlit.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        if self.beeper:
            self.beeper.set(False)
        return text


def main() -> None:
    import av
    import pandas as pd
    import streamlit as st
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from demo.webcam import default_weights

    st.set_page_config(page_title="졸음운전 감지", layout="wide")
    st.title("졸음운전 감지")
    st.caption("연속 눈 감김 1.5초면 주의, 3초면 졸음 경고. 하품 5분 3회면 하품 주의, 판정 불가 3초면 확인 필요. "
               "연구용 데모이며 실제 운전 보조에 쓰면 안 된다.")

    @st.cache_resource
    def load(path: str) -> LiveSession:
        return LiveSession(Path(path))

    if st.sidebar.button("새 운전 시작 (기록 지우기)"):
        load.clear()
    session = load(str(default_weights()))
    st.caption(f"모델 {session.weights.parent.name}/{session.weights.name}, 후처리 {session.post.label()}")

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        session.process(image)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    from demo.charts import closure_chart, counts, events_chart, events_frame, timeline_frame

    left, right = st.columns([3, 2])
    with left:
        context = webrtc_streamer(key="drowsy", mode=WebRtcMode.SENDRECV, rtc_configuration=LOCAL_ONLY,
                                  media_stream_constraints=CONSTRAINTS, video_frame_callback=callback, async_processing=True)
    area = right.empty()
    charts = st.empty()

    def show(snap: dict, draw_charts: bool) -> None:
        with area.container():
            items = (("모델 판정", snap["state"]), ("모델 경고", snap["stage"]), ("연속 눈 감김", f"{snap['closed_sec']:.1f}초"),
                     ("규칙 경고 (EAR 0.48)", snap["rule_stage"]), ("처리 속도", f"{snap['rate']:.1f} FPS"), ("운전 시간", f"{snap['seconds'] / 60:.1f}분"))
            for i in range(0, len(items), 2):
                for column, (name, value) in zip(st.columns(2), items[i:i + 2]):
                    column.metric(name, value)
        if not draw_charts:
            return
        events = events_frame(snap["events"])
        with charts.container():
            st.subheader("판정 결과")
            for column, (kind, n) in zip(st.columns(4), counts(events).items()):
                column.metric(kind, f"{n}회")
            if snap["history"]:
                st.altair_chart(closure_chart(timeline_frame(snap["history"]), snap["seconds"]), use_container_width=True)
                st.altair_chart(events_chart(events, snap["seconds"]), use_container_width=True)
            with st.expander("이벤트 표로 보기"):
                st.dataframe(events.rename(columns={"시각": "시작 후 초"}), hide_index=True, use_container_width=True)

    show(session.snapshot(), True)
    ticks = 0
    while context.state.playing:
        time.sleep(REFRESH_SEC)
        ticks += 1
        show(session.snapshot(), ticks % CHART_EVERY == 0)      # 그래프는 2초마다 다시 그린다(깜빡임을 줄임)
    show(session.snapshot(), True)
    if session.snapshot()["seconds"] > 0:
        st.subheader("운전 후 요약")
        st.markdown(session.finish())


if __name__ == "__main__":
    main()

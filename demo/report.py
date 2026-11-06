# Streamlit 사이트의 운전 후 리포트. 방금 끝난 운전(LiveJudge 기록)과 저장된 판정 로그(runs/sessions/*_log.csv)를 같은 화면으로 보여 준다.
# 기술서 6.3: 새 점수를 만들지 않고 화면에 뜬 단계와 연속 눈 감김 시간만 쓴다.

import re
from html import escape
from pathlib import Path

import pandas as pd

import config
from demo.charts import (bucket_chart, closure_chart, closure_episodes, counts, episode_chart, events_chart, events_frame,
                         timeline_frame)
from demo.cluster import lamps_html

SESSIONS = config.RUNS / "sessions"


def live_frames(snap: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LiveSession.snapshot() → (판정 기록 표, 이벤트 표). 시각은 운전 시작 후 초다."""
    timeline = timeline_frame(snap["history"], step=1)
    events = events_frame(snap["events"])
    if len(timeline):
        t0 = timeline["시각"].iloc[0]
        timeline["시각"] = (timeline["시각"] - t0).round(2)
        events["시각"] = (events["시각"] - t0).round(1)
    return timeline, events


def session_logs() -> list[Path]:
    """저장된 판정 로그, 최근 것부터."""
    return sorted(SESSIONS.glob("*_log.csv"), reverse=True)


def log_label(path: Path) -> str:
    """20261105_145132_log.csv → 2026-11-05 14:51 (Streamlit이면 표시를 붙인다). 이름 규칙과 다르면 파일 이름을 그대로 쓴다."""
    stem = path.name.removesuffix("_log.csv")
    found = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\d{2}", stem)
    if not found:
        return path.name
    y, m, d, hh, mm = found.groups()
    return f"{y}-{m}-{d} {hh}:{mm}" + (" Streamlit" if stem.endswith("_streamlit") else "")


def log_title(path: Path) -> str:
    """같은 이름의 요약 .md 첫 줄(모델·후처리 설정). 없으면 빈 글자."""
    md = path.with_name(path.name.removesuffix("_log.csv") + ".md")
    if not md.exists():
        return ""
    return md.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()


def log_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """판정 로그 CSV(기술서 6.2) → (판정 기록 표, 이벤트 표). event 열에 한 격자의 이벤트가 ;로 이어져 있다."""
    log = pd.read_csv(path, encoding="utf-8-sig")
    t0 = float(log["video_t"].iloc[0]) if len(log) else 0.0
    t = (log["video_t"] - t0).round(2)
    timeline = pd.DataFrame({"시각": t, "연속 눈 감김": log["closed_sec"].round(2), "단계": log["stage"]})
    rows = [(round(at, 1), kind) for at, cell in zip(t, log["event"].fillna("")) for kind in str(cell).split(";") if kind]
    return timeline, pd.DataFrame(rows, columns=["시각", "이벤트"])


def clock(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def show_report(st, timeline: pd.DataFrame, events: pd.DataFrame, note: str = "", empty: str = "판정 기록이 없습니다.") -> None:
    """리포트 한 장: 제목 줄 → 경고등(횟수) → 가장 긴 눈 감김 → 그래프 4개 → 이벤트 표."""
    seconds = float(timeline["시각"].iloc[-1]) if len(timeline) else 0.0
    n = counts(events)
    if n["졸음"]:
        head = f"{seconds / 60:.1f}분 운전, 졸음 경고 {n['졸음']}번"
    elif n["주의"]:
        head = f"{seconds / 60:.1f}분 운전, 주의 {n['주의']}번"
    else:
        head = f"{seconds / 60:.1f}분 운전, 눈 감김 경고는 없었습니다"
    st.markdown(f'<div class="gj-head">{head}</div>' + (f'<p class="gj-sub">{escape(note)}</p>' if note else ""), unsafe_allow_html=True)
    st.markdown(lamps_html({k: v > 0 for k, v in n.items()}, {k: f"{v}회" for k, v in n.items()}), unsafe_allow_html=True)

    if len(timeline) and timeline["연속 눈 감김"].max() > 0:
        i = timeline["연속 눈 감김"].idxmax()
        top, end = float(timeline.at[i, "연속 눈 감김"]), float(timeline.at[i, "시각"])
        st.markdown(f'<div class="gj-fact"><span class="gj-display">{top:.1f}초</span>'
                    f'<span>가장 길게 이어진 눈 감김, {clock(max(end - top, 0.0))}부터</span></div>', unsafe_allow_html=True)
    if not len(timeline):
        st.info(empty)
        return

    chart = timeline.iloc[::config.JUDGE_EVERY]         # 판정 간격으로 줄인다 (그래프만)
    left, right = st.columns(2)
    left.altair_chart(closure_chart(chart, seconds), use_container_width=True)
    right.altair_chart(events_chart(events, seconds).properties(height=220), use_container_width=True)     # 옆 그래프와 높이를 맞춘다
    left, right = st.columns(2)
    left.altair_chart(episode_chart(closure_episodes(timeline), seconds), use_container_width=True)
    right.altair_chart(bucket_chart(events, seconds), use_container_width=True)
    with st.expander(f"이벤트 {len(events)}개를 표로 보기"):
        table = events.assign(시각=events["시각"].map(clock)).rename(columns={"시각": "시작 후"})
        st.dataframe(table, hide_index=True, use_container_width=True)

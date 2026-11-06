# Streamlit 사이트의 운전 후 리포트(기술서 6.3, 그림 5 구성). 방금 끝난 운전과 저장된 판정 로그(runs/sessions/*_log.csv)를 같은 화면으로 보여 준다.
#   머리말(시각·운전 길이) → 핵심 숫자 4칸(주의, 졸음, 가장 긴 눈 감김, 하품) → 눈 감김 길이 막대·5분 구간별 주의·졸음 막대
#   → 고개 숙임·확인 필요·이어받기와 경고가 몰린 시간대 안내 → 이벤트 표.
# 새 점수를 만들지 않고 로그에 남은 단계, 연속 눈 감김, 현재 상태, 이벤트만 쓴다.

import re
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd

import config
from demo.charts import bucket_chart, closure_chart, closure_episodes, episode_chart, report_style, timeline_frame
from rule_based.config import ALERT_SEC, UNKNOWN_ALERT_SEC, WARN_SEC

SESSIONS = config.RUNS / "sessions"
STATE_NAMES = {"하품": "하품 감지", "고개숙임": "고개 숙임", "눈감김": "눈 감김"}     # 11.05 이전 로그의 class 열 이름


def live_frames(snap: dict) -> pd.DataFrame:
    """LiveSession.snapshot() → 실시간 그래프용 판정 기록 표. 기록은 최근 구간만 오고, 시각은 운전 시작(snap["t0"]) 후 초다."""
    timeline = timeline_frame(snap["history"], step=1)
    if len(timeline):
        timeline["시각"] = (timeline["시각"] - snap.get("t0", timeline["시각"].iloc[0])).round(2)
    return timeline


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
    """같은 이름의 요약 .md 첫 줄. 없으면 빈 글자."""
    md = path.with_name(path.name.removesuffix("_log.csv") + ".md")
    if not md.exists():
        return ""
    return md.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()


def load_log(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare(log: pd.DataFrame) -> tuple[pd.DataFrame, datetime | None]:
    """판정 로그 → (리포트용 표, 시작 실제 시각). 표 열: t(시작 후 초), stage, closed_sec, state, event.
    실제 시각(wall_time)이 없는 예전 로그는 시작 후 시각으로 보여 준다."""
    out = pd.DataFrame({"t": (log["video_t"] - log["video_t"].iloc[0]).round(3), "stage": log["stage"], "closed_sec": log["closed_sec"]})
    state = log["state"] if "state" in log else log.get("class", pd.Series("", index=log.index))
    out["state"] = state.fillna("").astype(str).map(lambda s: STATE_NAMES.get(s, s))
    out["event"] = log["event"].fillna("").astype(str)
    start = pd.to_datetime(log["wall_time"].iloc[0]).to_pydatetime() if "wall_time" in log else None
    return out, start


def make_clock(start: datetime | None):
    """시작 후 초 → 화면 시각 글자. 실제 시각이 있으면 HH:MM:SS, 없으면 시작 후 MM:SS."""
    if start is not None:
        return lambda sec: (start + timedelta(seconds=float(sec))).strftime("%H:%M:%S")
    return lambda sec: f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def episodes_of(frame: pd.DataFrame, state: str) -> list[tuple[float, float]]:
    """현재 상태가 state인 구간 목록 [(시작, 끝)] (시작 후 초)."""
    on = (frame["state"] == state).to_numpy()
    t = frame["t"].to_numpy()
    spans, start = [], None
    for i, value in enumerate(on):
        if value and start is None:
            start = t[i]
        if not value and start is not None:
            spans.append((start, t[i]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def summarize(log: pd.DataFrame) -> dict:
    """판정 로그 → 리포트에 쓰는 값."""
    frame, start = prepare(log)
    clock = make_clock(start)
    seconds = float(frame["t"].iloc[-1]) if len(frame) else 0.0
    events = pd.DataFrame([(t, kind) for t, cell in zip(frame["t"], frame["event"]) for kind in cell.split(";") if kind],
                          columns=["t", "kind"])
    count = lambda kind: int((events["kind"] == kind).sum())

    timeline = pd.DataFrame({"시각": frame["t"], "연속 눈 감김": frame["closed_sec"], "단계": frame["stage"]})
    episodes = closure_episodes(timeline)
    episodes["시각"] = episodes["시작"].map(clock)
    longest = float(frame["closed_sec"].max()) if len(frame) else 0.0
    longest_at = float(frame["t"].iloc[frame["closed_sec"].idxmax()]) - longest if longest > 0 else None

    # 5분 구간별 주의·졸음 (기술서 6.3). 가장 많이 몰린 구간이 팀장 요청의 '경고가 많았던 시간대'다
    alerts = events[events["kind"].isin(["주의", "졸음"])]
    n = max(int(seconds // config.SUMMARY_BUCKET_SEC) + 1, 1)
    which = (alerts["t"] // config.SUMMARY_BUCKET_SEC).astype(int).clip(upper=n - 1)
    names = [f"{clock(i * config.SUMMARY_BUCKET_SEC)}~" for i in range(n)]
    buckets = pd.DataFrame([(names[b], b, kind, int(((which == b) & (alerts["kind"] == kind)).sum()))
                            for b in range(n) for kind in ("주의", "졸음")], columns=["구간", "순서", "단계", "횟수"])
    per = buckets.groupby("순서")["횟수"].sum()
    peak = int(per.idxmax()) if len(alerts) else None

    head = episodes_of(frame, "고개 숙임")
    return {"seconds": seconds, "clock": clock, "start": clock(0.0), "end": clock(seconds),
            "warn": count("주의"), "drowsy": count("졸음"), "yawn_alert": count("하품 주의"), "check": count("확인 필요"),
            "carry": count("이어받기"), "unknown": count("판정 불가"),
            "longest": longest, "longest_at": longest_at, "episodes": episodes,
            "yawns": len(episodes_of(frame, "하품 감지")), "has_state": bool((frame["state"] != "").any()),
            "head": len(head), "head_sec": sum(b - a for a, b in head),
            "alerts": len(alerts), "buckets": buckets, "peak": peak, "n_buckets": n,
            "peak_count": int(per.iloc[peak]) if peak is not None else 0,
            "peak_range": (clock(peak * config.SUMMARY_BUCKET_SEC), clock(min((peak + 1) * config.SUMMARY_BUCKET_SEC, seconds))) if peak is not None else None,
            "events": events, "timeline": timeline}


def card(label: str, value: str, unit: str, note: str, dot: str | None = None) -> str:
    mark = f'<i style="background:{dot}"></i>' if dot else ""
    return (f'<div class="gj-card"><div class="gj-card-label">{mark}{escape(label)}</div>'
            f'<div class="gj-card-value"><span class="gj-display">{escape(value)}</span><small>{escape(unit)}</small></div>'
            f'<div class="gj-card-note">{escape(note)}</div></div>')


def peak_html(s: dict) -> str:
    """경고가 가장 많았던 시간대 안내 칸 (팀장 피드백 11.06). 제목 → 시각(큰 글씨) → 설명, 세 줄로 나눠 보여 준다."""
    title = '<div class="gj-callout-title">경고가 가장 많았던 시간대</div>'
    if not s["alerts"]:
        return title + "<div>주의·졸음 경고가 한 번도 없었습니다.</div>"
    if s["n_buckets"] == 1:
        return title + (f"<div>운전이 {config.SUMMARY_BUCKET_SEC / 60:g}분보다 짧아 시간대를 나누지 않았습니다. "
                        f"주의 {s['warn']}회, 졸음 {s['drowsy']}회가 있었습니다.</div>")
    a, b = s["peak_range"]
    time = f'<div class="gj-callout-time">{a} ~ {b}</div>'
    if s["alerts"] == 1:
        return title + time + "<div>주의·졸음 경고가 이 시간대에 한 번 있었습니다.</div>"
    share = s["peak_count"] / s["alerts"]
    return title + time + f"<div>주의·졸음 {s['alerts']}건 중 {s['peak_count']}건({share:.0%}). 이 시간대 전에 쉬어 가세요.</div>"


def show_report(st, log: pd.DataFrame, note: str = "", empty: str = "판정 기록이 없습니다.") -> None:
    """운전 후 리포트 한 장."""
    from demo.charts import KIND_COLORS

    if not len(log):
        st.info(empty)
        return
    s = summarize(log)
    st.markdown(f'<div class="gj-head">운전 종료 후 요약</div>'
                f'<p class="gj-sub">{s["start"]} ~ {s["end"]} · {s["seconds"] / 60:.1f}분 운전'
                + (f" · {escape(note)}" if note else "") + "</p>", unsafe_allow_html=True)

    longest = (f"{s['longest']:.1f}", "초", f"{s['clock'](s['longest_at'])} 시작") if s["longest_at"] is not None else ("-", "", "눈 감김 없음")
    yawn = (str(s["yawns"]), "회", f"하품 주의 {s['yawn_alert']}회") if s["has_state"] else (str(s["yawn_alert"]), "회", "하품 주의(예전 로그)")
    st.markdown('<div class="gj-cards">'
                + card("주의", str(s["warn"]), "회", f"눈 감김 {WARN_SEC:g}초 이상", KIND_COLORS["주의"])
                + card("졸음", str(s["drowsy"]), "회", f"눈 감김 {ALERT_SEC:g}초 이상", KIND_COLORS["졸음"])
                + card("가장 긴 눈 감김", *longest)
                + card("하품", *yawn, dot=KIND_COLORS["하품 주의"]) + "</div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        st.markdown('<div class="gj-panel-title">눈 감김 길이</div><div class="gj-panel-sub">주의·졸음이 뜬 눈 감김 한 번마다 연속 눈 감김 시간</div>',
                    unsafe_allow_html=True)
        if len(s["episodes"]):
            st.altair_chart(report_style(episode_chart(s["episodes"])), use_container_width=True)
        else:
            st.caption(f"{WARN_SEC:g}초 이상 이어진 눈 감김이 없었습니다.")
    with right.container(border=True):
        st.markdown('<div class="gj-panel-title">5분 구간별 주의·졸음</div><div class="gj-panel-sub">시간이 갈수록 경고가 늘어나는지 확인합니다</div>',
                    unsafe_allow_html=True)
        st.altair_chart(report_style(bucket_chart(s["buckets"])), use_container_width=True)

    head = f"총 {s['head_sec']:.1f}초" if s["has_state"] else "예전 로그에는 없음"
    st.markdown('<div class="gj-cards gj-cards-bottom">'
                + card("고개 숙임", str(s["head"]), "회", head)
                + card("확인 필요", str(s["check"]), "회", f"판정 불가·고개 숙임 {UNKNOWN_ALERT_SEC:g}초", KIND_COLORS["확인 필요"])
                + card("이어받기", str(s["carry"]), "회", "눈 감은 채 고개 숙임·미검출")
                + f'<div class="gj-callout">{peak_html(s)}</div></div>', unsafe_allow_html=True)

    with st.expander(f"이벤트 {len(s['events'])}개를 표로 보기"):
        table = s["events"].assign(시각=s["events"]["t"].map(s["clock"])).rename(columns={"kind": "이벤트"})[["시각", "이벤트"]]
        st.dataframe(table, hide_index=True, use_container_width=True)
    with st.expander("연속 눈 감김 시간 전체 보기"):
        chart = s["timeline"].iloc[::config.JUDGE_EVERY]
        st.altair_chart(report_style(closure_chart(chart, s["seconds"]).properties(title="")), use_container_width=True)

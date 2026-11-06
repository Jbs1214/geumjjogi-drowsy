# Streamlit 데모의 결과 그래프. 판정 기록(LiveJudge.history)과 이벤트(LiveJudge.events)로 Altair 그래프를 만든다.
#   연속 눈 감김 시간: 시간에 따른 선 하나 + 주의 1.5초·졸음 3초 기준선. 마우스를 올리면 그 시각의 값이 보인다
#   이벤트 시간표: 종류별 줄에 이벤트를 점으로 찍는다. 줄 이름이 종류를 말하므로 색만으로 구분하지 않는다
#   운전 후 요약(기술서 6.3): 눈 감김 한 번마다 길이 막대, 5분 구간별 이벤트 수 막대
# 색: 상태 팔레트(주의 warning, 졸음 critical, 확인 필요 serious)는 상태에만, 하품 주의는 범주 색(보라), 선은 범주 1번(파랑).

import altair as alt
import pandas as pd

from config import GAUGE_MAX_SEC, SUMMARY_BUCKET_SEC
from rule_based.config import ALERT_SEC, WARN_SEC

LINE = "#3987e5"            # 범주 1번 (어두운 화면용)
KINDS = ("졸음", "주의", "하품 주의", "확인 필요")
KIND_COLORS = {"졸음": "#d03b3b", "주의": "#fab219", "하품 주의": "#9085e9", "확인 필요": "#ec835a"}
STAGES = ("정상", "주의", "졸음")


def timeline_frame(history: list, step: int = 3) -> pd.DataFrame:
    """판정 기록 → 그래프용 표 (0.2초 간격으로 줄임). 열: 시각(초), 연속 눈 감김(초), 단계."""
    rows = history[::step]
    return pd.DataFrame({"시각": [round(h[0], 2) for h in rows], "연속 눈 감김": [round(h[4], 2) for h in rows],
                         "단계": [STAGES[h[2]] for h in rows]})


def events_frame(events: list) -> pd.DataFrame:
    return pd.DataFrame({"시각": [round(e.t, 1) for e in events], "이벤트": [e.kind for e in events]})


def closure_chart(frame: pd.DataFrame, duration: float) -> alt.LayerChart:
    """연속 눈 감김 시간 선 + 기준선 2개 + 가까운 시각에 세로선과 툴팁."""
    x = alt.X("시각:Q", title="시작 후 초", scale=alt.Scale(domain=[0, max(duration, 10.0)]))
    top = max(4.0, float(frame["연속 눈 감김"].max()) + 0.5 if len(frame) else 4.0)
    y = alt.Y("연속 눈 감김:Q", title="연속 눈 감김 (초)", scale=alt.Scale(domain=[0, top]))
    line = alt.Chart(frame).mark_line(color=LINE, strokeWidth=2, interpolate="step-after").encode(x=x, y=y)
    rules = pd.DataFrame({"기준": [WARN_SEC, ALERT_SEC], "이름": [f"주의 {WARN_SEC:g}초", f"졸음 {ALERT_SEC:g}초"]})
    rule = alt.Chart(rules).mark_rule(strokeDash=[4, 4], strokeWidth=1).encode(
        y="기준:Q", color=alt.Color("이름:N", scale=alt.Scale(domain=list(rules["이름"]), range=[KIND_COLORS["주의"], KIND_COLORS["졸음"]]),
                                    legend=None))
    label = alt.Chart(rules).mark_text(align="left", dx=4, dy=-6).encode(x=alt.value(0), y="기준:Q", text="이름:N")
    near = alt.selection_point(nearest=True, on="pointerover", fields=["시각"], empty=False)
    hover = alt.Chart(frame).mark_rule(strokeWidth=1, opacity=0.6).encode(
        x="시각:Q", opacity=alt.condition(near, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("시각:Q", title="시작 후 초"), alt.Tooltip("연속 눈 감김:Q", title="연속 눈 감김(초)"), alt.Tooltip("단계:N")]
    ).add_params(near)
    dot = line.mark_point(color=LINE, size=70, filled=True).encode(opacity=alt.condition(near, alt.value(1), alt.value(0)))
    return alt.layer(line, rule, label, hover, dot).properties(height=220, title="연속 눈 감김 시간")


def events_chart(frame: pd.DataFrame, duration: float) -> alt.Chart:
    """이벤트 시간표: 종류별 줄, 점 하나 = 이벤트 하나. 점에 마우스를 올리면 시각과 종류가 보인다."""
    x = alt.X("시각:Q", title="시작 후 초", scale=alt.Scale(domain=[0, max(duration, 10.0)]))
    y = alt.Y("이벤트:N", title=None, sort=list(KINDS), scale=alt.Scale(domain=list(KINDS)))
    color = alt.Color("이벤트:N", scale=alt.Scale(domain=list(KINDS), range=[KIND_COLORS[k] for k in KINDS]), legend=None)
    return (alt.Chart(frame).mark_point(filled=True, size=110, stroke="#1a1a19", strokeWidth=2)
            .encode(x=x, y=y, color=color, tooltip=[alt.Tooltip("시각:Q", title="시작 후 초"), alt.Tooltip("이벤트:N")])
            .properties(height=170, title="이벤트 시간표"))


def closure_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    """판정 기록 → 주의나 졸음이 화면에 뜬 눈 감김 한 번마다 한 줄. 열: 시작(초), 길이(초), 단계(그동안 뜬 가장 높은 단계).
    눈 감김 한 번은 연속 눈 감김이 0보다 커졌다가 0으로 돌아오기까지다(기술서 6.3 이벤트별 눈 감김 길이, 화면에 뜬 단계만 쓴다)."""
    rows, start, longest, top = [], None, 0.0, 0
    for t, sec, stage in zip(frame["시각"], frame["연속 눈 감김"], frame["단계"]):
        if sec > 0 and start is None:
            start, longest, top = t, 0.0, 0
        if start is not None:
            longest, top = max(longest, sec), max(top, STAGES.index(stage))
            if sec <= 0:
                rows.append((start, longest, top))
                start = None
    if start is not None:
        rows.append((start, longest, top))
    rows = [(round(s, 1), round(n, 2), STAGES[k]) for s, n, k in rows if k > 0]
    return pd.DataFrame(rows, columns=["시작", "길이", "단계"])


def episode_chart(episodes: pd.DataFrame, duration: float) -> alt.LayerChart:
    """이벤트별 눈 감김 길이 막대 + 주의·졸음 기준선."""
    x = alt.X("시작:Q", title="시작 후 초", scale=alt.Scale(domain=[0, max(duration, 10.0)]))
    top = max(GAUGE_MAX_SEC, float(episodes["길이"].max()) + 0.5 if len(episodes) else GAUGE_MAX_SEC)
    color = alt.Color("단계:N", scale=alt.Scale(domain=["주의", "졸음"], range=[KIND_COLORS["주의"], KIND_COLORS["졸음"]]), legend=None)
    bars = alt.Chart(episodes).mark_bar(size=7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=x, y=alt.Y("길이:Q", title="눈 감김 길이 (초)", scale=alt.Scale(domain=[0, top])), color=color,
        tooltip=[alt.Tooltip("시작:Q", title="시작 후 초"), alt.Tooltip("길이:Q", title="길이(초)"), alt.Tooltip("단계:N")])
    rules = pd.DataFrame({"기준": [WARN_SEC, ALERT_SEC]})
    rule = alt.Chart(rules).mark_rule(strokeDash=[4, 4], strokeWidth=1, color="#99A2AD").encode(y="기준:Q")
    return alt.layer(bars, rule).properties(height=200, title="눈 감김 한 번마다 길이")


def bucket_chart(events: pd.DataFrame, duration: float) -> alt.Chart:
    """SUMMARY_BUCKET_SEC(5분) 구간별 이벤트 수 (종류별로 쌓은 막대)."""
    n = max(int(duration // SUMMARY_BUCKET_SEC) + 1, 1)
    names = [f"{i * SUMMARY_BUCKET_SEC / 60:g}~{(i + 1) * SUMMARY_BUCKET_SEC / 60:g}분" for i in range(n)]
    which = (events["시각"] // SUMMARY_BUCKET_SEC).astype(int).clip(upper=n - 1) if len(events) else pd.Series(dtype=int)
    frame = pd.DataFrame([(names[b], kind, int(((which == b) & (events["이벤트"] == kind)).sum()) if len(events) else 0)
                          for b in range(n) for kind in KINDS], columns=["구간", "이벤트", "횟수"])
    color = alt.Color("이벤트:N", scale=alt.Scale(domain=list(KINDS), range=[KIND_COLORS[k] for k in KINDS]),
                      legend=alt.Legend(orient="top", title=None))
    return (alt.Chart(frame).mark_bar(width=alt.RelativeBandSize(0.5)).encode(x=alt.X("구간:N", title=None, sort=names, axis=alt.Axis(labelAngle=0)),
                                               y=alt.Y("횟수:Q", title="이벤트 수", axis=alt.Axis(tickMinStep=1)), color=color,
                                               order=alt.Order("이벤트:N"), tooltip=["구간:N", "이벤트:N", "횟수:Q"])
            .properties(height=200, title="5분 구간별 이벤트 수"))


def counts(frame: pd.DataFrame) -> dict[str, int]:
    """종류별 이벤트 횟수 (항상 네 종류 모두)."""
    return {kind: int((frame["이벤트"] == kind).sum()) if len(frame) else 0 for kind in KINDS}

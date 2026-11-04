# Streamlit 데모의 결과 그래프. 판정 기록(LiveJudge.history)과 이벤트(LiveJudge.events)로 Altair 그래프를 만든다.
#   연속 눈 감김 시간: 시간에 따른 선 하나 + 주의 1.5초·졸음 3초 기준선. 마우스를 올리면 그 시각의 값이 보인다
#   이벤트 시간표: 종류별 줄에 이벤트를 점으로 찍는다. 줄 이름이 종류를 말하므로 색만으로 구분하지 않는다
# 색: 상태 팔레트(주의 warning, 졸음 critical, 확인 필요 serious)는 상태에만, 하품 주의는 범주 색(보라), 선은 범주 1번(파랑).

import altair as alt
import pandas as pd

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


def counts(frame: pd.DataFrame) -> dict[str, int]:
    """종류별 이벤트 횟수 (항상 네 종류 모두)."""
    return {kind: int((frame["이벤트"] == kind).sum()) if len(frame) else 0 for kind in KINDS}

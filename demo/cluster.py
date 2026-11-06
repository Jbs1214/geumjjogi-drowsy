# Streamlit 사이트의 계기판 화면 조각. 연속 눈 감김 게이지, 경고등, 계측값 표, 졸음 테두리를 HTML 문자열로 만든다.
# 색은 경고등의 의미에만 쓴다: 정상 청록, 주의 호박, 졸음 적색, 확인 필요 주황, 하품 주의 보라 (demo/charts.py와 같은 색).
# 글꼴은 Google Fonts에서 불러온다. 인터넷이 없으면 맑은 고딕으로 보인다.

import math
from html import escape

from config import GAUGE_MAX_SEC
from demo.charts import KIND_COLORS
from rule_based.config import ALERT_SEC, UNKNOWN_ALERT_SEC, WARN_SEC, YAWN_REPEAT, YAWN_REPEAT_WINDOW_SEC

STAGE_COLORS = ("#36BFAE", KIND_COLORS["주의"], KIND_COLORS["졸음"])     # 정상은 계기판에만 쓰는 청록
LAMP_COLORS = {kind: KIND_COLORS[kind] for kind in ("주의", "졸음", "하품 주의", "확인 필요")}
LAMP_NOTES = {"주의": f"눈 감김 {WARN_SEC:g}초", "졸음": f"눈 감김 {ALERT_SEC:g}초",
              "하품 주의": f"{YAWN_REPEAT_WINDOW_SEC / 60:g}분에 {YAWN_REPEAT}회", "확인 필요": f"판정 불가·고개 숙임 {UNKNOWN_ALERT_SEC:g}초"}
ALARM_HTML = '<div class="gj-alarm"></div>'     # 졸음 단계 동안 화면 테두리를 깜빡인다. 바뀔 때만 따로 그려야 효과가 끊기지 않는다

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap');
:root { --gj-bg:#1F2329; --gj-panel:#2A2F37; --gj-line:#3B424C; --gj-text:#E6E8EB; --gj-muted:#99A2AD; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp td, .stApp th {
  font-family: 'IBM Plex Sans KR', 'Malgun Gothic', sans-serif; }
.stApp h1, .stApp h2 { font-family: 'Black Han Sans', 'Malgun Gothic', sans-serif; font-weight: 400; letter-spacing: 0.01em; }
.gj-display { font-family: 'Black Han Sans', 'Malgun Gothic', sans-serif; }
.gj-cluster { background: var(--gj-panel); border: 1px solid var(--gj-line); border-radius: 22px; padding: 18px 20px 16px; }
.gj-gauge svg { display: block; width: 100%; max-width: 360px; margin: 0 auto; }
.gj-stage { text-align: center; font-family: 'Black Han Sans', 'Malgun Gothic', sans-serif; font-size: 3.2rem; line-height: 1;
  margin: 2px 0 14px; transition: color .25s; }
.gj-lamps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 0 0 14px; }
.gj-cluster .gj-lamps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.gj-lamp { border: 1px solid var(--gj-line); border-radius: 10px; padding: 8px 6px 7px; text-align: center; color: #5E6670;
  background: #23272E; transition: background .25s, color .25s, box-shadow .25s; }
.gj-lamp b { display: block; font-size: .95rem; font-weight: 700; }
.gj-lamp span { display: block; font-size: .72rem; margin-top: 2px; }
.gj-lamp.on { color: #15181C; background: var(--lamp); border-color: var(--lamp); box-shadow: 0 0 18px -2px var(--lamp); }
.gj-read { display: grid; grid-template-columns: 1fr 1fr; gap: 0 18px; font-size: .9rem; }
.gj-read div { display: flex; justify-content: space-between; padding: 5px 0; border-top: 1px solid var(--gj-line); }
.gj-read dt { color: var(--gj-muted); }
.gj-read dd { margin: 0 0 0 8px; font-variant-numeric: tabular-nums; font-weight: 500; white-space: nowrap; }
.gj-alarm { position: fixed; inset: 0; pointer-events: none; z-index: 999990; border: 12px solid #D03B3B;
  animation: gj-pulse 0.9s ease-in-out infinite; }
@keyframes gj-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
.gj-head { font-family: 'Black Han Sans', 'Malgun Gothic', sans-serif; font-size: 2.6rem; line-height: 1.15; margin: 4px 0 6px; }
.gj-sub { color: var(--gj-muted); margin: 0 0 18px; }
.gj-fact { display: flex; align-items: baseline; gap: 12px; margin: 4px 0 18px; }
.gj-fact .gj-display { font-size: 2.2rem; }
.gj-cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 16px; }
.gj-cards-bottom { grid-template-columns: repeat(3, minmax(0, 1fr)) minmax(0, 1.6fr); margin: 16px 0; }
.gj-card { background: var(--gj-panel); border: 1px solid var(--gj-line); border-radius: 14px; padding: 14px 16px 12px; min-width: 0; }
.gj-card-label { color: var(--gj-muted); font-size: .88rem; display: flex; align-items: center; gap: 7px; }
.gj-card-label i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
.gj-card-value { display: flex; align-items: baseline; gap: 4px; margin: 6px 0 2px; }
.gj-card-value .gj-display { font-size: 2.3rem; line-height: 1.1; color: var(--gj-text); }
.gj-card-value small { color: var(--gj-muted); font-size: .95rem; }
.gj-card-note { color: var(--gj-muted); font-size: .78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.gj-callout { background: rgba(242, 201, 76, .12); border: 1px solid rgba(242, 201, 76, .45); border-radius: 14px; padding: 14px 18px;
  color: var(--gj-text); font-size: .92rem; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; gap: 2px; }
.gj-callout-title { color: var(--gj-muted); font-size: .82rem; }
.gj-callout-time { font-family: 'Black Han Sans', 'Malgun Gothic', sans-serif; font-size: 1.55rem; line-height: 1.25; color: #F2C94C;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.gj-head, .gj-sub, .gj-card, .gj-callout, .gj-panel-title, .gj-panel-sub { word-break: keep-all; overflow-wrap: break-word; }
.gj-panel-title { font-weight: 700; font-size: 1.05rem; color: var(--gj-text); }
.gj-panel-sub { color: var(--gj-muted); font-size: .82rem; margin: 2px 0 6px; }
@media (max-width: 900px) { .gj-cards, .gj-cards-bottom { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .gj-lamps { grid-template-columns: repeat(2, minmax(0, 1fr)); } .gj-read { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .gj-alarm { animation: none; } .gj-lamp, .gj-stage { transition: none; } }
</style>
"""


def _point(sec: float, r: float, cx: float = 130, cy: float = 142) -> tuple[float, float]:
    """게이지 눈금 sec초의 좌표. 0초는 왼쪽 끝, GAUGE_MAX_SEC는 오른쪽 끝이다."""
    theta = math.pi * (1 - min(max(sec, 0.0), GAUGE_MAX_SEC) / GAUGE_MAX_SEC)
    return cx + r * math.cos(theta), cy - r * math.sin(theta)


def _arc(start: float, end: float, r: float) -> str:
    (x0, y0), (x1, y1) = _point(start, r), _point(end, r)
    return f"M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}"


def gauge_html(closed_sec: float, stage: int) -> str:
    """연속 눈 감김 반원 게이지. 바깥 띠는 정상·주의·졸음 구간, 안쪽 호는 지금 값이다."""
    color = STAGE_COLORS[stage]
    parts = [f'<path d="{_arc(0, GAUGE_MAX_SEC, 100)}" stroke="#3B424C" stroke-width="16" fill="none"/>']
    for (a, b), zone in zip(((0, WARN_SEC), (WARN_SEC, ALERT_SEC), (ALERT_SEC, GAUGE_MAX_SEC)), STAGE_COLORS):
        parts.append(f'<path d="{_arc(a, b, 114)}" stroke="{zone}" stroke-width="4" fill="none" opacity=".75"/>')
    if closed_sec > 0:
        parts.append(f'<path d="{_arc(0, closed_sec, 100)}" stroke="{color}" stroke-width="16" fill="none"/>')
        x, y = _point(closed_sec, 100)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#1F2329" stroke-width="3"/>')
    for sec in (WARN_SEC, ALERT_SEC):
        (x0, y0), (x1, y1), (tx, ty) = _point(sec, 90), _point(sec, 120), _point(sec, 132)
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#E6E8EB" stroke-width="2"/>')
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" fill="#99A2AD" font-size="12" text-anchor="middle">{sec:g}초</text>')
    parts.append(f'<text x="130" y="122" fill="#E6E8EB" font-size="50" text-anchor="middle" class="gj-display">{closed_sec:.1f}</text>')
    parts.append('<text x="130" y="142" fill="#99A2AD" font-size="13" text-anchor="middle">연속 눈 감김 (초)</text>')
    return f'<div class="gj-gauge"><svg viewBox="0 0 260 152" role="img" aria-label="연속 눈 감김 {closed_sec:.1f}초">{"".join(parts)}</svg></div>'


def lamps_html(lit: dict[str, bool], notes: dict[str, str] = LAMP_NOTES) -> str:
    """경고등 4개. lit[종류]가 참이면 켜진다. notes는 경고등 아래 작은 글씨다(실시간은 기준, 리포트는 횟수)."""
    cells = [f'<div class="gj-lamp{" on" if lit.get(kind) else ""}" style="--lamp:{color}"><b>{kind}</b><span>{escape(notes[kind])}</span></div>'
             for kind, color in LAMP_COLORS.items()]
    return f'<div class="gj-lamps">{"".join(cells)}</div>'


def readout_html(items: list[tuple[str, str]]) -> str:
    rows = "".join(f"<div><dt>{escape(name)}</dt><dd>{escape(value)}</dd></div>" for name, value in items)
    return f'<dl class="gj-read">{rows}</dl>'


def cluster_html(closed_sec: float, stage: int, stage_text: str, lit: dict[str, bool], items: list[tuple[str, str]]) -> str:
    """실시간 계기판 전체. 졸음 테두리(ALARM_HTML)는 따로 그린다."""
    return (f'<div class="gj-cluster">{gauge_html(closed_sec, stage)}'
            f'<div class="gj-stage" style="color:{STAGE_COLORS[stage]}">{escape(stage_text)}</div>'
            f"{lamps_html(lit)}{readout_html(items)}</div>")

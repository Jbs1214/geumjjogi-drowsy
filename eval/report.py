# 규칙 대 학습 비교표 (최종 기획서 산출물). 채점 결과(summary.csv)를 모아 runs/compare.md 하나로 만든다.
#   Val+Val2: 모델·후처리 선택에 쓴 데이터 (runs/<final tag>/eval_val2)
#   Val2:     웹캠 녹화만 (runs/eval_webcam)
#   Test2:    최종 판정 (runs/eval_test2, 있을 때만)
# 선택에 쓴 데이터의 숫자는 성능 수치가 아니라는 점을 표 위에 적는다.
#
# 실행: python -m eval.report

import json

import pandas as pd

import config

COLUMNS = {"방법": "방법", "3초_재현율": "3초 재현율", "3초_시간당_오경보": "3초 오경보/시간", "주의_재현율": "1.5초 재현율",
           "주의_시간당_오경보": "1.5초 오경보/시간", "눈감김_구간F1": "눈감김 F1", "하품_구간F1": "하품 F1", "고개숙임_구간F1": "고개숙임 F1"}


def table(path, note: str) -> str:
    if not path.exists():
        return f"_{note}: 아직 없음_\n"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame[[c for c in COLUMNS if c in frame.columns]].rename(columns=COLUMNS)
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in row) + " |")
    return f"{note}\n\n" + "\n".join(lines) + "\n"


def main() -> None:
    final = json.loads((config.RUNS / "final.json").read_text(encoding="utf-8"))
    tag = final["tag"]
    parts = [
        "# 규칙 대 학습 비교표",
        "",
        f"고른 모델: {tag} {final['model']}. 목표(최종 기획서 7-3): 3초 재현율 90% 이상, 눈 감김 F1 규칙 대비 향상, 웹캠 15 FPS 이상.",
        "",
        "## Test2 (최종 판정, 한 번만)",
        "",
        table(config.RUNS / "eval_test2" / "summary.csv", "판정 화면 없이 녹화한 영상"),
        "## Val + Val2 (선택에 쓴 데이터, 성능 수치가 아님)",
        "",
        table(config.RUNS / tag / "eval_val2" / "summary.csv", "NITYMED Val 6개 + 웹캠 녹화. 모델은 후처리 없이 채점(모델 종류 선택용)"),
        "## Val2 웹캠 녹화만 (선택에 쓴 데이터)",
        "",
        table(config.RUNS / "eval_webcam" / "summary.csv", "사용자 1명. 모델은 고른 후처리(post.json)를 적용해 채점"),
    ]
    out = config.RUNS / "compare.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

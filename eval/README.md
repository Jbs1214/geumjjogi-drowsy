# eval — 공통 평가와 비교표

담당: 육주영. 개정 기획서(2026-10-30) 9절 기준으로 구현했다.

**지금 상태:** 합성 데이터로만 검증했다. **실제 비교 수치는 아직 없다.**

## 파일
| 파일 | 하는 일 |
| --- | --- |
| `metrics.py` | 9절 지표 계산 (규칙·모델 공통) |
| `compare.py` | 규칙 대 학습 모델 비교표 → `runs/compare.md`, `runs/compare.csv` |
| `tests/` | 손으로 계산한 값과 지표 대조 |

## 지표 (기획서 9절)
| 지표 | 목표 | 평가 데이터 | 코드 |
| --- | --- | --- | --- |
| 눈 감김 3초(졸음) 이벤트 재현율 | 90% 이상 | 자체 연기 영상(3~7초 감기) | `event_recall` |
| 마이크로슬립(1.5초) 주의 재현율 | 결과 보고 | 자체 연기 영상(1~2초 감기) | `event_recall` |
| 눈 감김 구간 검출 F1 | 규칙 대비 향상 | AI Hub 검수 라벨, 보정에 안 쓴 클립 | `compare` |
| 하품 구간 검출 | 결과 보고 | 하품 구간이 적어 참고용 | `compare` |
| 졸음 점수 ROC-AUC | 규칙 대비 향상 | 눈 감김·하품 대 정상 구간 | `compare` |
| 오경보율 | 값 보고 | 정상 운전 영상 | `false_alarms_per_hour` |
| 경고 지연 | 값 보고 | 기준 시간 도달 → 경고 | `alert_delay` |
| 처리 속도 | 15 FPS 이상 | 웹캠 실측 | `fps` |

자체 영상 지표와 웹캠 처리 속도는 영상·데모가 나와야 잴 수 있어 `compare.py`에 넣지 않았다. 함수만 준비되어 있다.

## 비교 방식
- **정답:** 검수 라벨. 자동 라벨은 규칙의 결과라 정답으로 쓰지 않는다.
- **평가 클립:** test 분할 중 검수 라벨이 있는 클립. 보정에 쓴 클립의 그룹은 test에 들어가지 않는다.
- **규칙 행:** 자동 라벨 `y`를 그대로 쓴다. `y`가 규칙의 프레임별 결과이기 때문이다.
- **창 지표(정확도, macro F1, AUC):** 모든 방법이 같은 0.5초 간격 창으로 잰다.
- **구간 지표:** 정답 구간은 프레임 라벨에서 구한다. 예측 구간은 규칙은 프레임, 모델은 0.5초 칸(창 가운데)에서 구한다. 라벨 오차가 있어 ±0.5초 이내로 닿으면 찾은 것으로 본다.
- **창보다 짧은 클립:** 모든 방법에서 평가에서 뺀다.

창 라벨을 구간 정답으로 쓰면 창 비율 기준에 못 미치는 짧은 눈 감김이 정답에서 지워져 재현율이 부풀려진다. 그래서 프레임 라벨을 쓴다.

## 실행
```bash
python -m eval.compare --reviewed-dir <검수 라벨 폴더> --calibration-clips <보정 클립 목록.txt> \
    --weights checkpoints/best_model_v01.pt checkpoints/best_model_v02.pt
python -m eval.compare ... --fps 15                     # 입력 15 FPS 흉내. 규칙 행은 빠진다
python -m eval.compare ... --conditions <조건.csv>      # clip_id, 조건 열(조명·안경 등). 조건별 행 추가
python -m eval.compare ... --no-baseline                # scikit-learn 기준선 제외
python -m unittest discover -t . -s eval/tests          # 테스트
```

| 결과 열 | 뜻 |
| --- | --- |
| `accuracy`, `macro_f1` | 창 단위 |
| `eye_segment_f1`, `eye_segment_recall` | 눈감김·고개숙임 구간 단위 |
| `yawn_segment_f1`, `yawn_segment_recall` | 하품 구간 단위 |
| `auc` | 정상 대 이상 창의 졸음 점수 구분력 |
| `ms_per_window` | 창 하나 추론 시간(ms). 얼굴 랜드마크 추론 시간은 포함하지 않는다 |

결과표는 `runs/`에 저장하고 저장소에는 올리지 않는다 (`.gitignore`).

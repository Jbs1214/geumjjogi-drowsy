# rule_based — 규칙 기반 졸음 판정과 경고

담당: 김민영. 지금 있는 것: **눈 감김 판정, 연속 눈 감김 타이머(주의 1.5초·졸음 3초), PERCLOS, 졸음 점수(0~100)와 점수 단계**

## 맡은 일 (기획서 7절)
- 규칙 기반 기준선 (PERCLOS·졸음 점수)
- 경고 로직 (연속 눈 감김 주의 1.5초·졸음 3초)

## 파일
| 파일 | 하는 일 |
| --- | --- |
| `config.py` | 임계값·설정. 코드에 값을 직접 쓰지 않고 여기서 가져다 쓴다 |
| `common.py` | 지표들이 같이 쓰는 계산: 임계값 판정(NaN은 False), 연속 구간, 최근 `WINDOW_SEC`초 창 |
| `eyes.py` | 프레임별 눈 감음 판정 (`ear < EAR_CLOSED`, 얼굴 미검출은 False)과 연속 감음 구간. 타이머 입력용 임계값은 `ear_threshold()` |
| `alarm.py` | 규칙·모델 공용 타이머. `EyeClosureAlarm.update(closed, t, eyes_visible=True)` → (단계, 연속 눈 감김 초). 모델 확률은 `ProbToClosed`로 `closed`를 만든다 |
| `perclos.py` | 최근 `WINDOW_SEC`초 중 눈 감은 프레임 비율 |
| `closure.py` | 최근 `WINDOW_SEC`초 안 가장 긴 연속 눈 감음(초) |
| `yawn.py` | 최근 `WINDOW_SEC`초 하품 횟수. `mar > MAR_YAWN`이 `YAWN_MIN_SEC` 이상이면 1회 |
| `head.py` | 최근 `WINDOW_SEC`초 중 고개 숙인(`HeadDown`) 프레임 비율 |
| `score.py` | 지표 4개를 0~1로 맞춰 가중합한 졸음 점수(0~100)와 단계 |

지표는 모두 창이 `WINDOW_SEC`만큼 차기 전에는 NaN이다.

| 설정 | 값 | 근거 |
| --- | --- | --- |
| `EAR_CLOSED` | 0.2 | 웹캠 검증: 감음 0.04, 뜬 상태 최저 0.25 |
| `CAUTION_SEC` / `DROWSY_SEC` | 1.5 / 3.0 | 기획서 3-4. 연속 눈 감김 주의 / 졸음 |
| `BEST_PARAMS_PATH` / `EAR_BASE_PERCENTILE` | `auto_label/best_params.json` / 75 | 타이머 입력의 임계값 = `ear_ratio` × 클립 EAR P75. 파일이 없으면 `EAR_CLOSED` |
| `PROB_ON` / `PROB_OFF` | 0.6 / 0.4 | 기획서 3-3. 모델 확률 히스테리시스 |
| `BEST_PARAMS_PATH` / `EAR_BASE_PERCENTILE` | `auto_label/best_params.json` / 75 | 눈 감음 임계값 = `ear_ratio`(0.6) × 평소 EAR |
| `MAR_YAWN` | 0.3 | 하품 영상: 하품 최고 0.47~0.50, 입 다묾 최고 0.06 |
| `YAWN_MIN_SEC` | 1.0 | 짧은 입 벌림과 구분. 하품 길이 1.47~2.17초 |
| `PITCH_DOWN_RELATIVE` / `PITCH_DOWN_ABSOLUTE` | 17.5 / 25.0 | 기술서 3.3. 정면 pitch 대비 기준, 정면 pitch가 아직 없을 때의 기준 |
| `FRONT_PITCH_SEC` | 60 | 정면 pitch를 구하는 세션 시작 구간(초) |
| `WINDOW_SEC` | 7 | 팀 합의 (15 FPS 기준 105행) |
| `PERCLOS_FULL` / `CLOSURE_FULL_SEC` / `YAWN_FULL` | 0.5 / 3.0초 / 1회 | 지표가 1.0이 되는 값 (그 위는 1.0으로 자름) |
| `WEIGHTS` | 0.25 × 4 | 근거 데이터가 없어 일단 모두 같게 |
| `STAGE_WARN` / `STAGE_DROWSY` | 40 / 70 | 주의 / 졸음 점수 경계 |

값을 바꾸면 이유와 결과를 README 실험 기록 표에 남긴다.

## 졸음 점수
점수 = 100 × (0.25 × PERCLOS/0.5 + 0.25 × 가장 긴 감김/3초 + 0.25 × 하품/1회 + 0.25 × 고개 숙임 비율), 각 지표는 0~1로 자름.

| 영상 (웹캠, 1명) | 최고 점수 | 단계 |
| --- | --- | --- |
| 눈 4초 감기 | 48.3 | 주의 |
| 눈 2초 감기 + 고개 숙이기·들기 | 39.2 | 정상 |
| 하품 3번 | 31.0 | 정상 |

가중치가 같으면 눈 지표 둘만으로는 최대 50점이라 하품이나 고개 숙임이 함께 있어야 졸음(70)에 닿는다.

## 입력 / 출력
- 입력: `landmarks.extract` CSV (`add_features()`로 `ear`·`mar` 계산, `pitch`는 CSV에 있음)
- 출력: 주의·졸음이 된 시각(초), 프레임별 지표 4개·졸음 점수·점수 단계
- 훅 미구현 상태의 `eyes_visible=False`: 값을 쓰지 않고 `closed`만 본다. `closed=False`면 타이머를 지우고 (정상, 0.0)을 낸다. 이어받기와 '판정 불가' 표시는 없다
- 하품 제외와 이어받기는 `alarm.py`에 훅 자리만 있다 (기획서 7절 육주영 담당)

## 실행
```bash
python -m rule_based.eyes outputs/landmarks/<이름>.csv      # 눈 감음 구간
python -m rule_based.alarm outputs/landmarks/<이름>.csv     # 주의·졸음 시각 (--best-params로 임계값 파일 지정)
python -m rule_based.perclos outputs/landmarks/<이름>.csv   # 1초마다 PERCLOS
python -m rule_based.closure outputs/landmarks/<이름>.csv   # 1초마다 가장 긴 눈 감김
python -m rule_based.yawn outputs/landmarks/<이름>.csv      # 하품 구간과 1초마다 하품 횟수
python -m rule_based.head outputs/landmarks/<이름>.csv      # 1초마다 고개 숙임 비율
python -m rule_based.score outputs/landmarks/<이름>.csv     # 1초마다 지표·점수·단계
```

## 정할 것
- 얼굴 미검출 프레임 처리: 지금은 NaN이면 감음 구간이 끊긴다. 짧은 끊김을 채울 `MAX_GAP_SEC`를 정한다 (육주영 님 `sequence_model/config.py`의 `FILL_MAX_SEC`와 맞출 것)
- 가중치·단계 경계: 지금은 같은 가중치. 졸음 데이터로 다시 정한다
- **데모 전 마지막 확인:** 말할 때 입 벌림이 하품으로 잘못 세지지 않는지 (`MAR_YAWN`, `YAWN_MIN_SEC`). 하품 영상만 녹화했고 말하는 영상은 아직 확인하지 않았다

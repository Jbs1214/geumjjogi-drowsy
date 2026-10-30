# rule_based — 규칙 기반 졸음 판정과 경고

담당: 김민영. 지금은 **눈 감김 판정, 2초 경고, PERCLOS**까지 있다. 졸음 점수는 아직 없다.

## 맡은 일 (기획서 7절)
- 규칙 기반 기준선 (PERCLOS·졸음 점수)
- 경고 로직 (눈 감김 2초)

## 파일
| 파일 | 하는 일 |
| --- | --- |
| `config.py` | 임계값·설정. 코드에 값을 직접 쓰지 않고 여기서 가져다 쓴다 |
| `eyes.py` | 프레임별 눈 감음 판정 (`ear < EAR_CLOSED`, 얼굴 미검출은 False)과 연속 감음 구간 |
| `alarm.py` | 눈 감음이 `ALERT_SEC` 이상 이어지면 경고. 실시간은 `EyeClosureAlarm.update()` |
| `perclos.py` | 최근 `WINDOW_SEC`초 중 눈 감은 프레임 비율. 창이 차기 전에는 NaN |

| 설정 | 값 | 근거 |
| --- | --- | --- |
| `EAR_CLOSED` | 0.2 | 웹캠 검증: 감음 0.04, 뜬 상태 최저 0.25 |
| `ALERT_SEC` | 2.0 | 기획서 목표: 눈 감김 약 2초 이상 시 경고 |
| `WINDOW_SEC` | 7 | 팀 합의 (15 FPS 기준 105행) |

값을 바꾸면 이유와 결과를 README 실험 기록 표에 남긴다.

## 입력 / 출력
- 입력: `landmarks.extract` CSV (`add_features()`로 `ear` 계산)
- 출력: 경고가 울린 시각(초), 프레임별 PERCLOS

## 실행
```bash
python -m rule_based.eyes outputs/landmarks/<이름>.csv      # 눈 감음 구간
python -m rule_based.alarm outputs/landmarks/<이름>.csv     # 경고 시각
python -m rule_based.perclos outputs/landmarks/<이름>.csv   # 1초마다 PERCLOS
```

## 정할 것
- 얼굴 미검출 프레임 처리: 지금은 NaN이면 감음 구간이 끊긴다. 짧은 끊김을 채울 `MAX_GAP_SEC`를 정한다 (육주영 님 `sequence_model/config.py`의 `FILL_MAX_SEC`와 맞출 것)
- 졸음 점수: PERCLOS·가장 긴 눈 감김·하품·고개 숙임을 합친 0~100 점수와 정상/주의/졸음 기준
- 정면 pitch 보정: 정면 pitch가 녹화마다 +1~+9.4로 달라 `PITCH_DOWN`(15)까지 여유가 약 5도뿐이다. 시작할 때 몇 초간 정면 pitch를 재서 빼는 보정을 검토한다 (후속 과제)
- **데모 전 마지막 확인:** 말할 때 입 벌림이 하품으로 잘못 세지지 않는지 (`MAR_YAWN`, `YAWN_MIN_SEC`). 하품 영상만 녹화했고 말하는 영상은 아직 확인하지 않았다

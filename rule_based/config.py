# 규칙 기반 판정에서 쓰는 임계값과 설정. 코드에는 값을 직접 쓰지 않고 여기서 가져다 쓴다.
# 값을 바꾸면 이유와 결과를 README 실험 기록 표에 남긴다.

from pathlib import Path

BEST_PARAMS_PATH = Path(__file__).resolve().parent.parent / "auto_label" / "best_params.json"   # calibrate가 찾은 임계값. 기술서 3.1
EAR_BASE_PERCENTILE = 75  # 평소 EAR 분위. 눈 감음 임계값 = best_params의 ear_ratio × 평소 EAR
EAR_CLOSED = 0.2        # ear가 이보다 작으면 눈 감음(MVP 고정 기준). 검증: 감음 0.04, 뜬 상태 최저 0.25 (고개 들 때)
WARN_SEC = 1.5          # 눈 감음이 이 시간(초) 이상 이어지면 주의. 기획서 3-4의 시작값
ALERT_SEC = 3.0         # 눈 감음이 이 시간(초) 이상 이어지면 졸음 경고. 기획서 3-4
CAUTION_SEC, DROWSY_SEC = WARN_SEC, ALERT_SEC   # eval/window_eval.py가 쓰는 이름
PROB_ON, PROB_OFF = 0.6, 0.4    # 모델 눈 감김 켜짐·꺼짐 기준. 기술서 0장
CARRY_MIN_SEC = 0.5     # 판정 불가(눈이 안 보임)가 되기 전에 이만큼 감고 있었으면 타이머를 이어받는다. 수정 기획서 F 5-1
UNKNOWN_ALERT_SEC = 3.0 # 판정 불가가 이 시간(초) 이어지면 확인 필요 경고. 수정 기획서 F 5-3
UNKNOWN_EVENT_GAP_SEC = 1.0 # 판정 로그의 "판정 불가" 이벤트는 얼굴이 이 시간(초) 이상 보인 뒤 사라질 때만 새로 센다. 기술서 6.2, 11.06 팀 결정
RULE_MIN_SEC = 5.0          # 실시간 규칙: 평소 EAR을 정할 기록이 이보다 짧으면 규칙은 준비 중이다 (demo/webcam.py LiveRule)
BASELINE_EVERY_SEC = 0.5    # 실시간 규칙: 평소 EAR을 다시 구하는 간격(초)
YAWN_REPEAT = 3         # 하품이 YAWN_REPEAT_WINDOW_SEC 안에 이 횟수면 하품 주의. 수정 기획서 F 4·5-3
YAWN_REPEAT_WINDOW_SEC = 300.0
YAWN_COOLDOWN_SEC = 120.0   # 하품 주의 뒤 이 시간 동안 다시 알리지 않는다
YAWN_MERGE_SEC = 1.0        # 하품 판정 끊김이 이 시간(초) 이하이면 한 번으로 센다. 기술서 3.2
YAWN_MAX_SEC = 7.0          # 끊김을 이어 붙인 하품의 길이 상한(초). 하품 알림 계산에만 쓴다. 기술서 3.2
YAWN_DISPLAY_CLOSED_SEC = 0.5   # 입을 다문 지 이만큼 지나면 화면에서 하품을 끝낸다 (창 정답 50%로 생기는 지연을 화면에서 줄임)

MAR_YAWN = 0.3          # mar가 이보다 크면 입 벌림. 검증: 하품 최고 0.47~0.50, 입 다묾 최고 0.06 (yawn 영상)
YAWN_MIN_SEC = 1.0      # 입 벌림이 이 시간(초) 이상 이어지면 하품 1회. 짧은 입 벌림(말하기 등)과 구분. 검증: 하품 1.47~2.17초
PITCH_DOWN_RELATIVE = 17.5  # 정면 pitch보다 이만큼(도) 이상 숙이면 고개 숙임 (숙이면 +). 기술서 3.3
PITCH_DOWN_ABSOLUTE = 25.0  # 정면 pitch가 아직 없을 때의 고개 숙임 기준(도)
FRONT_PITCH_SEC = 60.0      # 세션 시작부터 이 시간 안의 눈 뜬 프레임으로 정면 pitch를 구한다

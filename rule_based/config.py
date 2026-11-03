# 규칙 기반 판정에서 쓰는 임계값과 설정. 코드에는 값을 직접 쓰지 않고 여기서 가져다 쓴다.
# 값을 바꾸면 이유와 결과를 README 실험 기록 표에 남긴다.

EAR_CLOSED = 0.2        # ear가 이보다 작으면 눈 감음. 검증: 감음 0.04, 뜬 상태 최저 0.25 (고개 들 때)
CAUTION_SEC = 1.5       # 연속 눈 감김이 이 시간(초) 이상이면 주의. 기획서 3-4
DROWSY_SEC = 3.0        # 연속 눈 감김이 이 시간(초) 이상이면 졸음. 기획서 3-4 (Euro NCAP SD202 수면 3초)
BEST_PARAMS_PATH = "auto_label/best_params.json"  # 타이머 입력의 임계값 파일. 없으면 EAR_CLOSED를 쓴다
EAR_BASE_PERCENTILE = 75  # 클립의 '눈 뜬' 기준 EAR. 임계값 = best_params의 ear_ratio × 이 퍼센타일 (auto_label.py와 같음)
PROB_ON = 0.6           # 모델 눈감김 확률이 이 이상이면 감음으로 켠다. 기획서 3-3
PROB_OFF = 0.4          # 모델 눈감김 확률이 이 이하이면 끈다. 사이 값은 이전 상태 유지

MAR_YAWN = 0.3          # mar가 이보다 크면 입 벌림. 검증: 하품 최고 0.47~0.50, 입 다묾 최고 0.06 (yawn 영상)
YAWN_MIN_SEC = 1.0      # 입 벌림이 이 시간(초) 이상 이어지면 하품 1회. 짧은 입 벌림(말하기 등)과 구분. 검증: 하품 1.47~2.17초
PITCH_DOWN = 15.0       # pitch(도)가 이보다 크면 고개 숙임 (숙이면 +). 검증: 숙임 중앙값 +19.6, 정면은 녹화마다 +1~+9.4라 보정 필요

WINDOW_SEC = 7          # 시간창 길이(초). 팀 합의 (15 FPS 기준 105행). sequence_model/config.py의 10과 다르다

# 졸음 점수 = 100 × Σ 가중치 × 지표(0~1). 지표는 아래 값에서 1.0이 되고 그 위는 1.0으로 자른다
PERCLOS_FULL = 0.5      # PERCLOS
CLOSURE_FULL_SEC = 3.0  # 가장 긴 눈 감김(초)
YAWN_FULL = 1           # 하품 횟수
                        # 고개 숙임 비율은 원래 0~1이라 그대로 쓴다
WEIGHTS = {"perclos": 0.25, "closure": 0.25, "yawn": 0.25, "head_down": 0.25}  # 합 1. 일단 모두 같게

STAGE_WARN = 40         # 점수가 이 이상이면 주의
STAGE_DROWSY = 70       # 점수가 이 이상이면 졸음

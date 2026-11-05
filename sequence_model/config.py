# 시퀀스 모델 파트에서 쓰는 설정값. 기획서 3-3(시간창)·3-5(라벨) 기준이고, 후보값은 실험으로 확정한다.

FPS = 30                                # 샘플 영상 FPS. 입력 FPS가 달라도 이 값으로 맞춘다
WINDOW_SEC = 2.0                        # 창 길이(초). 후보 2~3초, train.py --window-sec로 바꾼다
STRIDE_SEC = 0.5                        # 학습·평가 이동 간격. 겹쳐 잘라 경계에 걸린 눈 감김을 놓치지 않는다
REALTIME_STEP_SEC = 0.5                 # 실시간 데모: 0.5초마다 최근 창을 다시 판정

# auto_label 특징 11개. 자동 라벨 JSON의 X 열 순서와 같다고 가정한다 (기획서 3-1)
FACE_COLUMN = "face"                    # 얼굴 검출 여부. 정규화하지 않는다
FEATURE_COLUMNS: tuple[str, ...] = (
    FACE_COLUMN, "ear_l", "ear_r", "ear", "mar", "pitch", "yaw", "roll", "blink_l", "blink_r", "jaw_open",
)

# 자동 라벨 y 값 (기획서 3-5). 불확실은 모델 클래스가 아니라 학습·평가에서 뺀다
STATES = ("정상", "눈감김·고개숙임", "하품", "불확실")
UNCERTAIN = 3
CLASSES = STATES[:UNCERTAIN]
N_CLASSES = len(CLASSES)
LABEL_RATIO = 0.5                       # 창 안에서 한 상태의 비율이 이 값 이상이면 창의 정답으로 삼는다

FILL_MAX_SEC = 0.0                      # 미검출 구간을 앞뒤 값으로 채우지 않고 0으로 둔다 (기획서 3-3)

SPLIT_SEED = 42                         # 클립 그룹 분할 시드
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}    # 클립 수 기준 목표 비율
SPLIT_TRIALS = 1000                     # 그룹 배정을 시도할 횟수

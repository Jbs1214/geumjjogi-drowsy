# drowsy_v3 설정값 한 곳. 2026-11-02 기획서 최종 변경 9개 항목을 따른다.
# 값을 바꾸면 이유와 결과를 README.md 실험 기록 표에 남긴다.

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 입력 (PC마다 바꾸는 경로). 이전 폴더는 읽기만 하고 고치지 않는다. 원본 영상과 사람 라벨은 복사하지 않는다
OLD = Path(r"C:\Users\Woori\Desktop\drowsy_all_in_one")
NITYMED_DIR = OLD / "NITYMED"
REVIEWED_DIRS = (OLD / "data" / "reviewed_nitymed_v2", OLD / "data" / "reviewed_nitymed_yawn_v2")   # 재검수 라벨(segments 5종)
EYE_CHECK = OLD / "data" / "eye_check_segments.csv"         # 사람이 확인한 눈감김 구간 24개
DRIVER_SHEET = OLD / "data" / "driver_id.csv"               # 사람이 적은 운전자 번호 21명
OLD_SPLIT = OLD / "data" / "nitymed_split.csv"              # 운전자 단위 분할 (Train 22, Val 6, Test 13)
AIHUB_FEATURES = OLD / "data" / "processed" / "features"    # 같은 추출기로 뽑은 AI Hub 089 특징
AIHUB_LABELS = OLD / "data" / "processed" / "labels_v2"     # AI Hub 089 2차 자동 라벨
AIHUB_NEGATIVES = OLD / "data" / "negative_clips.txt"       # 하드 네거티브 클립 (전부 정상)
AIHUB_HOLDOUT = OLD / "data" / "test_groups.txt"            # 탑승자 B의 SGA 그룹. 쓰지 않는다
USER_VIDEO = Path(r"C:\Users\Woori\Downloads\나의 테스트영상.mp4")
USER_VIDEOS = (USER_VIDEO, Path(r"C:\Users\Woori\Pictures\Camera Roll\WIN_20261103_11_23_43_Pro.mp4"))   # 사용자가 찍은 영상. 학습·Val·Test에 쓰지 않는다
TEST2_DIR = Path(r"C:\Users\Woori\Videos\drowsy_test2")   # 새 평가 데이터(Test2) 영상. 최종 판정에 한 번만 쓴다
TEST2_ORIGINAL_DIR = Path(r"C:\Users\Woori\Videos\drowsy_test2_원본")   # 가변 FPS 원본 보관 (변환본만 Test2에 둔다)
WEBCAM_DIR = Path(r"C:\Users\Woori\Videos\drowsy_webcam")  # 판정 화면을 보며 녹화한 웹캠 영상(Val2). 선택에 써도 된다
RECORD_FPS = 30.0                                          # 녹화·변환 영상의 일정한 FPS (demo/cfr.py)
DMD_ARCHIVES = Path(r"C:\Users\Woori\Downloads\DMD")      # DMD 졸음 세트 압축 7개 (원본, 고치지 않는다)

# 미니 데이터 묶음 (dataset/import_mini.py). 읽기만 한다
MINI_DATA = Path(r"C:\Users\Woori\Desktop\datas\mini-data")
MINI_NITYMED_LABELS = MINI_DATA / "nitymed" / "labels" / "nitymed_labels_20261103"
MINI_BUNDLE = MINI_DATA / "dmd" / "labels" / "20261103_84e7bf4"     # DMD labels_v2·특징, NITYMED 특징
MINI_DMD_VAL_DRIVERS = 2                # DMD 운전자 6명 중 val 수. 나머지는 train, DMD test는 신규 7명 몫으로 남긴다

# 산출물
DATA = ROOT / "data"
FEATURE_DIR = DATA / "features"         # 영상별 특징 parquet (원본 FPS)
FEATURE_V8_DIR = DATA / "features_v8"   # v8로 판정할 영상의 특징. 좌우 눈 번호를 팀 기준으로 바꾼 뒤 다시 뽑았다 (README 실험 기록 11.04)
HEAD_DIR = DATA / "head_down"           # 사람이 표시한 고개숙임 구간
HEAD_DRAFT_DIR = DATA / "head_down_drafts"
USER_LABEL_DIR = DATA / "user_labels"   # 사용자 영상에 사람이 표시한 상태 구간 (참고용 채점)
USER_LABEL_DRAFT_DIR = DATA / "user_labels_drafts"
TEST2_LABEL_DIR = DATA / "test2_labels"     # Test2 영상에 사람이 표시한 상태 구간
TEST2_LABEL_DRAFT_DIR = DATA / "test2_labels_drafts"
WEBCAM_LABEL_DIR = DATA / "webcam_labels"   # Val2 웹캠 녹화에 사람이 표시한 상태 구간
WEBCAM_LABEL_DRAFT_DIR = DATA / "webcam_labels_drafts"
DMD_ANN_DIR = DATA / "dmd_ann"          # 압축에서 꺼낸 DMD 라벨 JSON
LABEL_DIR = DATA / "labels"             # 클립당 라벨 JSON 1개 (기획서 9번 양식)
SPLIT = DATA / "split.csv"              # 운전자 단위 분할. 모든 실험이 같이 쓴다
NORM = DATA / "norm.json"               # train 창 특징의 평균·표준편차
WINDOW_DIR = DATA / "windows"           # build_windows.py 결과
RUNS = ROOT / "runs"
YAWN_WEIGHTS = RUNS / "yawn1" / "cnn1d_v01.pt"   # 최종 구성의 하품 전용 모델 (기술서 4.1)

# 얼굴 검출 (landmarks/extract.py). main·김민영 특징 CSV(v8 학습 창)와 같은 값이다. 다르면 얼굴을 잡는 프레임이 학습 때와 달라진다
MIN_FACE_CONFIDENCE = 0.3

# 창 (기획서 6·7번)
FPS = 15.0                              # 공통 FPS. 원본이 25(NITYMED)·30(AI Hub)이어도 여기에 맞춘다
WINDOW_SEC = 3.0
WINDOW_FRAMES = round(FPS * WINDOW_SEC)     # 45프레임
STRIDE_SEC = 0.5                        # 창 끝 시각 간격
YAWN_RATIO = 0.5                        # 하품은 창 안 비율이 이 값 이상이면 정답
JUDGE_SEC = 0.2                         # 모델 판정 간격(초). 실시간과 파일 평가가 같다 (수정 기획서 F 5-3)
JUDGE_EVERY = round(JUDGE_SEC * FPS)    # 15 FPS 격자 3칸
DISPLAY_MIN_CLOSE_SEC = 0.5             # 화면에서 이보다 짧게 이어진 눈감김은 정상으로 보인다(깜빡임). 경고·채점과 무관
MAX_MISSING = 0.3                       # 얼굴 미검출 프레임 비율이 이보다 크면 창을 뺀다

# 클래스 (기획서 2·3번). 불확실은 클래스가 아니라 제외 표시다
CLASSES = ("정상", "눈감김", "하품", "고개숙임")
NORMAL, CLOSED, YAWN, HEAD = 0, 1, 2, 3
UNCERTAIN = -1
N_CLASSES = len(CLASSES)
# 겹침 우선순위 (기획서 4번): 불확실 > 눈감김 > 고개숙임 > 하품 > 정상. 앞의 채널이 이긴다
PRIORITY = (("uncertain", UNCERTAIN), ("eye_closed", CLOSED), ("head_down", HEAD), ("yawn", YAWN))
YAWN_STEP = 0.5                         # 하품 경계 단위(초, 기획서 라벨 정밀도)
YAWN_SELECT_THRESHOLD = 0.5             # 하품 전용 모델의 최고 epoch를 고를 때 쓰는 창 F1 기준 확률
YAWN_ENTER = 0.6                        # 하품 전용 모델 켜짐 기준. README "하품 전용 모델 비교 기준"
YAWN_MODEL_MIN_SEC = 0.4                # 이보다 짧은 하품 전용 모델 검출은 하품으로 보지 않는다

# 입력 특징 11개 (기획서 5번, 사람이 11개로 확정)
FACE_COLUMN = "face"
FEATURES: tuple[str, ...] = ("face", "ear_l", "ear_r", "ear", "mar", "pitch", "yaw", "roll", "blink_l", "blink_r", "jaw_open")

# 학습
SEEDS = (42, 43, 44)
EPOCHS = 50
BATCH_SIZE = 64                         # 팀 기획 표 "4.3 학습 설정(초기값)"의 값. 기술서 v2에는 값이 없다 (README에 표시)
LR = 1e-3
PATIENCE = 8                            # Val macro F1이 이만큼 안 오르면 멈춘다
AIHUB_SHARE = 0.5                       # AI Hub 창을 무작위로 줄여 train에서 이 비율을 넘지 않게 한다

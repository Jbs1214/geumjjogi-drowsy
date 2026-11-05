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
WEBCAM_DIR = Path.home() / "Videos" / "drowsy_webcam"      # 판정 화면을 보며 녹화한 웹캠 영상(Val2). 선택에 써도 된다. PC마다 사용자 폴더 아래
RECORD_FPS = 30.0                                          # 녹화·변환 영상의 일정한 FPS (demo/cfr.py)
# 웹캠 백엔드(OpenCV CAP_ 이름). 11.04 집 PC 측정: DSHOW 9 FPS(어두우면 자동 노출이 프레임을 줄임), MSMF 30 FPS. MediaPipe는 7ms/프레임
WEBCAM_BACKEND = "MSMF"
DMD_ARCHIVES = Path(r"C:\Users\Woori\Downloads\DMD")      # DMD 졸음 세트 압축 7개 (원본, 고치지 않는다)

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
DMD_VIDEO_DIR = DATA / "dmd_raw"        # 압축에서 꺼낸 DMD 얼굴 영상 (rgb_face.mp4만)
DMD_ANN_DIR = DATA / "dmd_ann"          # 압축에서 꺼낸 DMD 라벨 JSON
LABEL_DIR = DATA / "labels"             # 클립당 라벨 JSON 1개 (기획서 9번 양식)
SPLIT = DATA / "split.csv"              # 운전자 단위 분할. 모든 실험이 같이 쓴다
NORM = DATA / "norm.json"               # train 창 특징의 평균·표준편차
WINDOW_DIR = DATA / "windows"           # build_windows.py 결과
RUNS = ROOT / "runs"

# 타이머·알림 (rule_based/closure_timer.py, rule_based/alerts.py). 기술서 3.1~3.3.
# 눈 감김 임계값·하품 MAR·고개 숙임 pitch 등 규칙 값은 rule_based/config.py(김민영 님 담당)에서 읽는다
WARN_SEC = 1.5                  # 연속 눈 감김이 이 시간(초) 이상이면 주의
ALERT_SEC = 3.0                 # 연속 눈 감김이 이 시간(초) 이상이면 졸음
CARRY_MIN_SEC = 0.5             # 판정 불가·고개 숙임 직전에 이만큼 감고 있었으면 타이머를 이어받는다
UNKNOWN_ALERT_SEC = 3.0         # 판정 불가가 이 시간(초) 이어지면 확인 필요 경고
YAWN_REPEAT = 3                 # 하품이 YAWN_REPEAT_WINDOW_SEC 안에 이 횟수면 하품 주의
YAWN_REPEAT_WINDOW_SEC = 300.0
YAWN_COOLDOWN_SEC = 120.0       # 하품 주의 뒤 이 시간 동안 다시 알리지 않는다
YAWN_DISPLAY_CLOSED_SEC = 0.5   # 입을 다문 지 이만큼 지나면 화면에서 하품을 끝낸다 (창 정답 50%로 생기는 지연을 화면에서 줄임)
GAUGE_MAX_SEC = ALERT_SEC + 1.0     # Streamlit 계기판 게이지 눈금 끝(초). 화면 표시용이고 판정 기준이 아니다
SUMMARY_BUCKET_SEC = 300.0          # 운전 후 요약에서 이벤트 수를 묶는 구간(초). 기술서 6.3: 5분
# 고개 숙임은 모델 출력 대신 pitch 규칙(rule_based/head.py)으로 정한다. 기술서 7·8: pitch 규칙과 맞지 않으면 모델 출력은 쓰지 않는다.
# NITYMED Val에서 구간 F1 pitch 0.674 대 v8 모델 0.564, 둘 중 하나가 숙임이라 한 칸의 겹침 0.210 (runs/v8/eval_head_down)
HEAD_FROM_PITCH = True
# 고개 숙임 pitch 기준(도, 숙이면 +). 11.04 사용자 결정. 웹캠 2차 녹화 숙인 구간의 77%, NITYMED 고개숙임 창의 25%가 넘는다.
# rule_based/config.PITCH_DOWN(15도, 김민영 님 설정)과 따로 두고 내 판정 경로(모델·규칙 모두)는 이 값을 쓴다
HEAD_PITCH_DOWN = 25.0
# 정면 기준 상대 pitch (11.04 사용자 결정). 정면 pitch가 카메라 위치마다 +1~+9도로 달라 절대 기준이 흔들린다
# (집 웹캠 녹화: 정면 중앙값 7.5도, 숙임 중앙값 25.3도). 정면 기준 = 세션 시작 PITCH_BASE_SEC 안에서 두 눈을 뜬 검출 프레임의
# pitch 중앙값(sequence_model/postprocess.PitchBaseline). 기준이 아직 없으면 HEAD_PITCH_DOWN(절대)을 쓴다.
# 11.05: 상대 pitch는 기술서 v3에 없어 끈다(사용자 결정: 기획서·기술서 기준). 기술서 갱신이 합의되면 다시 켠다(README 갱신 안건 4)
HEAD_PITCH_RELATIVE = False
# 상대 pitch를 켰을 때만 쓴다. 정면보다 이만큼(도) 더 숙이면 고개 숙임(11.04 README 기준으로 고른 값)
HEAD_PITCH_DOWN_REL = 17.5
PITCH_BASE_SEC = 60.0
# v8부터 팀 좌우 기준(LEFT_EYE = 362쪽)으로 학습했다. v7까지는 ear_l·ear_r이 반대다(blink_l·blink_r은 같음, README 11.04)
TEAM_SIDES_FROM_TAG = 8
# 눈 상태 규칙 (11.04 사용자 결정, 기획서·기술서 갱신 안건). 눈 감김 EAR 기준은 rule_based/config.EAR_CLOSED(0.2)를 쓴다
#   한쪽 눈만 감았으면(두 눈 중 큰 EAR ≥ EAR_CLOSED이고 두 눈 블렌드셰이프 차 ≥ WINK_BLINK_DIFF) 앞을 보는 정상 상태라
#   모델이 눈감김이라 해도 정상으로 본다. 차이 조건이 없으면 실제 감김 중 순간적인 EAR 흔들림까지 '뜸'으로 봐서
#   NITYMED Val 1.5초 감김 재현율이 21/28 → 14/28로 떨어졌다(차이 조건을 넣으면 21/28 유지, 웹캠 한쪽 눈 구간 99% 정상)
#   고개를 숙인 상태에서는 모델이 감은 눈을 잘 못 보므로, 두 눈 EAR이 모두 기준 미만이고 두 눈 블렌드셰이프가 모두
#   BLINK_CLOSED 이상이면 눈감김으로 본다(우선순위 눈감김 > 고개숙임, 기술서 2.4)
BLINK_CLOSED = 0.5
WINK_BLINK_DIFF = 0.3

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

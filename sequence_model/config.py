# 시퀀스 모델 파트에서 쓰는 설정값. 기획서 3-3(시간창)·3-5(클래스)에 정해진 값만 채우고, 나머지는 TODO로 둔다.

FPS = 15                                # 원본 FPS가 달라도 15 FPS로 맞춘다
WINDOW_SEC = 10                         # 창 길이(초)
WINDOW_FRAMES = FPS * WINDOW_SEC        # 150프레임
STRIDE_SEC = 5                          # 학습·평가용 이동 간격(50% 겹침)
REALTIME_STEP_SEC = 1                   # 실시간 데모: 1초마다 최근 10초를 다시 판정

CLASSES = ("정상", "주의", "졸음")
N_CLASSES = 2                           # 2 = 정상 vs 졸음, 3 = 정상/주의/졸음 ('주의' 구간이 충분한지 확인 후 결정)

# TODO: landmarks/ 특징 DataFrame 열 이름이 정해지면 맞춘다 (EAR, MAR, 고개 각도 3개 등)
FEATURE_COLUMNS: tuple[str, ...] = ()

# TODO: 샘플 영상으로 확정 (짧은 끊김은 앞뒤 값 채움, 긴 끊김 창은 제외)
FILL_MAX_SEC: float | None = None       # 이 길이 이하의 미검출 구간은 앞뒤 값으로 채운다
DROP_MISSING_RATIO: float | None = None  # 창에서 미검출 비율이 이 값 이상이면 창을 제외한다

SPLIT_SEED = 42                         # 탑승자 단위 분할 시드

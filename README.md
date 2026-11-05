# 졸음운전 감지 (geumjjogi-drowsy)

졸음운전 경고: 웹캠 영상에서 눈 감김·하품·고개 숙임을 실시간으로 분석해 졸음을 감지하고 경고한다.

---

## <img src="assets/geumjjogi_logo.png" width="36" alt="GEUMJJOGI 로고" align="absmiddle"> 팀원(금쪽이 - GEUMJJOGI)

| 역할 | 이름 | 담당 |
| --- | --- | --- |
| 팀장 | 장범수 | 데이터 수집, 특징 추출, 라벨링, 학습 데이터 생성 |
| 팀원 | 김민영 | LSTM, 눈 감김 판정, 평가 |
| 팀원 | 육주영 | 1D-CNN, 하품·고개 숙임 판정, 데모 |

주요 작업은 최종 기획서 7-1을 본다.

기간: 2026.10.26 ~ 11.09 (발표 11.09). 자세한 계획은 최종 기획서(2026-11-03)를 본다.

## 📌 프로젝트 소개

1. MediaPipe Face Landmarker로 얼굴 좌표 17개(눈 12, 입 4, 코끝 1)를 뽑고, 프레임마다 특징 11개(`face, ear_l, ear_r, ear, mar, pitch, yaw, roll, blink_l, blink_r, jaw_open`)를 계산한다.
2. **졸음 단계**는 연속 눈 감김 시간으로 정한다. 규칙과 모델이 같은 타이머(`rule_based/alarm.py`의 `ClosureTimer`)를 쓴다.

    | 단계 | 연속 눈 감김 | 화면 |
    | --- | --- | --- |
    | 정상 | 1.5초 미만(깜빡임 포함) | 표시 없음 |
    | 주의(마이크로슬립) | 1.5초 이상, 3초 미만 | 주의 표시 |
    | 졸음(수면) | 3초 이상 | 경고음·화면 경고 |

    - 판정 불가(얼굴 미검출)와 고개 숙임은 직전까지 이어 감은 시간이 0.5초 이상이면 감은 것으로 이어서 센다(기획서 5-1).
    - 판정 불가가 3초 이어지면 확인 필요 경고를 낸다.
3. **학습 모델**: 3초 창(15 FPS, 45 × 11)으로 1D-CNN·LSTM을 학습한다. 출력은 4클래스 softmax(정상, 눈감김, 하품, 고개숙임)이고, 불확실은 클래스가 아니라 제외 표시다. 학습 창은 0.5초 간격, 실시간 판정은 0.2초 간격이다.
4. **별도 알림**: 하품이 5분에 3회면 하품 주의(알림 후 2분 쿨다운). 고개 숙임은 표시만 한다.
5. Streamlit 화면에 단계와 연속 눈 감김 시간을 실시간으로 보여 주고, 끝나면 주의·졸음·하품 이벤트를 시간순으로 요약한다.

## 🛠️ 개발 환경

| 구분 | 항목 | 버전 |
| --- | --- | --- |
| OS | Windows 10 Pro | 22H2 (빌드 19045) |
| 에디터 | VS Code | 1.144.0 |
| AI 코딩 | Claude Code | 2.1.316 |
| 가상환경 | Anaconda3 (`geumjjogi-drowsy`) | conda 26.5.3 |
| 언어 | Python | 3.11.18 |
| 딥러닝 프레임워크 | PyTorch | 2.7.1+cu118 |
| GPU | NVIDIA GeForce GTX 1060 3GB | 드라이버 560.94 |
| CUDA | CUDA Toolkit | 11.8 (nvcc V11.8.89) |

## 📦 라이브러리

| 라이브러리 | 버전 | 용도 |
| --- | --- | --- |
| `mediapipe` | 1.2.0 | 얼굴 랜드마크 추출 (Tasks API) |
| `opencv-python` | 4.10.0.84 | 영상·웹캠 프레임 읽기 |
| `opencv-contrib-python` | 4.10.0.84 | `mediapipe`가 함께 설치하는 패키지. 버전 고정용 |
| `numpy` | 2.3.5 | 특징 계산 |
| `pandas` | 3.0.6 | 특징 시계열 표 |
| `pyarrow` | 26.0.0 | 특징 표를 parquet로 저장·읽기 |
| `Pillow` | 13.0.0 | 데모 화면에 한글 글자 그리기 |
| `torch` (CUDA 11.8용) | 2.7.1+cu118 | 1D-CNN·LSTM 학습 |
| `streamlit`, `streamlit-webrtc` | 1.66.1, 0.79.0 | 실시간 데모 화면 |
| `scikit-learn` | 1.9.1 | 창 요약 특징 기준선 |
| `onnx`, `onnxruntime` | [미정] | (선택) 추론 속도 비교 |

경고음은 Windows 기본 `winsound`로 낸다. 목록에 없는 라이브러리는 팀에 먼저 공유하고 추가한다.

## 🗂️ 데이터

데이터(영상·특징·라벨)와 가중치는 레포에 포함되어 있지 않다. 비공개 팀 Google Drive에서 받는다.

| 데이터 | 내용 | 받는 곳 | 이용 조건 |
| --- | --- | --- | --- |
| NITYMED | 야간 실차 주행, 운전자 21명, 하품·마이크로슬립 영상, 25 FPS | Kaggle(720p), 연구실 사이트(1080p) | CC BY 4.0, 지정 논문 인용 |
| DMD (졸음 부분, RGB) | 주간 정차 실차 연기, 13명분, 프레임 단위 눈 상태·하품 라벨, 29.76 FPS | 신청 폼 → 메일 링크 | CC BY-NC-ND 4.0, 학술용, 가공본 배포 금지 |
| 자체 촬영 | 실내 연기(정상, 1~2초·3~7초 눈 감기), 참가자 동의서 | 팀 내부 | 원본은 외부 업로드 금지, 분석 후 폐기 |

- 원본 영상과 이전 라벨의 경로는 루트 `config.py`의 "입력" 부분에 있다. PC마다 바꾼다. 만든 특징·라벨·창은 `data/`, 가중치·채점 결과는 `runs/`에 쌓이며 둘 다 깃에 올리지 않는다.
- 얼굴 랜드마크 모델 `models/face_landmarker.task`는 MediaPipe에서 받아 이 위치에 둔다(깃에 올리지 않음).
- 라벨은 클립마다 JSON 1개다. 눈 감김(`eye_closed`)·고개 숙임(`head_down`)은 프레임 단위, 하품(`yawn`)은 0.5초 단위로 적는다. 겹치면 불확실 > 눈감김 > 고개숙임 > 하품 > 정상 순서로 정답을 정한다.
- NITYMED는 사람이 검수한 라벨에 고개 숙임을 사람이 따로 표시한다. DMD는 `dataset/dmd_convert.py`가 변환한다.
- 분할은 운전자 단위 파일 하나(`data/split.csv`)를 모든 실험이 같이 쓴다.
- **공개용 데모 영상은 NITYMED나 자체 촬영 영상으로만 만든다**(DMD는 가공본 배포 금지).

## ⚙️ 설치

Anaconda가 설치되어 있어야 합니다. Anaconda Prompt에서 실행합니다.

```bash
conda create -n geumjjogi-drowsy python=3.11 -y
conda activate geumjjogi-drowsy
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## ▶️ 실행

레포 루트에서 `python -m`으로 실행한다.

```bash
conda activate geumjjogi-drowsy

# 1. 특징 추출 (CPU) → data/features/*.parquet
python -m auto_label.extract

# 2. 사람 검수: NITYMED 고개 숙임 표시 → data/head_down/  (127.0.0.1 로컬 화면)
python -m auto_label.head_review.server

# 3. 라벨 JSON·운전자 단위 분할 → data/labels/, data/split.csv
python -m dataset.convert_labels
python -m dataset.dmd_convert --split train   # 새 DMD 클립만 변환, 기존 라벨은 덮어쓰지 않음

# 4. 3초 창·정규화 값 (v7: NITYMED + DMD, 세션 기준 정규화) → data/windows_session/
python -m sequence_model.build_windows --sources nitymed dmd --relative --out data/windows_session
python -m sequence_model.build_windows --export data/windows_session   # 기획서 이름 split.json·norm.npz

# 5. 1D-CNN·LSTM 학습 (GPU, 시드 3개) → runs/<tag>/
python -m sequence_model.train --tag v7 --windows data/windows_session
python -m sequence_model.train_yawn --tag yawn1   # 하품 전용 모델
python -m sequence_model.baseline --tag base1      # 창 요약 특징 + scikit-learn 기준선
python -m eval.window_eval --help                  # 창 단위 확률 파일 평가 (train_lstm 결과)

# 6. 평가: Val 선택 → 후처리 선택 → 최종 선택 → 비교표 (--with-webcam: Val2 웹캠 녹화 포함)
python -m eval.evaluate val --tag v7 --with-webcam
python -m eval.evaluate post --tag v7 --with-webcam
python -m eval.evaluate final --tags v6 v7
python -m eval.report                       # 규칙 대 학습 비교표 → runs/compare.md

# 7. 웹캠 녹화 영상 채점 (사람 정답 표시 후)
python -m auto_label.user_review.server --set webcam
python -m eval.score_user --set webcam --tags v7

# 8. 데모
python -m demo.webcam --camera 0                        # OpenCV 실시간 (영상 저장 안 함)
python -m demo.demo_video --input <영상> --weights runs/v7/cnn1d_v01.pt   # 판정 화면을 입힌 데모 영상
python -m streamlit run demo/streamlit_app.py           # Streamlit 데모 (http://127.0.0.1:8501)

# 테스트
python -m unittest discover -t . -s tests
```

## 🗂️ 폴더 구조

| 폴더 | 하는 일 | 주요 파일 |
| --- | --- | --- |
| (루트) | 설정값 한 곳 | `config.py`, `.streamlit/config.toml` |
| `dataset/` | NITYMED·DMD 라벨 → 클립당 JSON, 운전자 단위 분할 | `convert_labels.py`, `dmd_convert.py`, `import_mini.py` |
| `auto_label/` | 특징 추출, 사람 검수 도구(고개 숙임, 웹캠 정답) | `extract.py`, `head_review/`, `user_review/` |
| `landmarks/` | 좌표 17개 번호, 특징 계산, 세션 기준 정규화 | `points.py`, `extract.py`, `features.py`, `session.py` |
| `rule_based/` | 눈 감김 타이머, 하품·고개 숙임 규칙, 하품 5분 3회·판정 불가 경고, 운전 후 요약 | `alarm.py`, `alerts.py`, `eyes.py`, `yawn.py`, `head.py`, `config.py` |
| `sequence_model/` | 3초 창·4클래스 정답, 1D-CNN·LSTM 학습, 후처리 | `build_windows.py`, `models.py`, `train.py`, `postprocess.py` |
| `eval/` | 구간·경고 지표, Val·웹캠 채점, 규칙 대 학습 비교표 | `metrics.py`, `evaluate.py`, `score_user.py`, `report.py` |
| `demo/` | OpenCV 웹캠, 데모 영상, Streamlit 데모 | `webcam.py`, `demo_video.py`, `cfr.py`, `streamlit_app.py`, `charts.py` |
| `tests/` | 파이프라인 테스트 45개 | `test_v3.py` |

## ✅ 평가

| 지표 | 목표 |
| --- | --- |
| 졸음(3초) 이벤트 재현율 | 90% 이상 |
| 손 가린 하품 재현율 | MAR 규칙 초과 |
| 맨입 하품 재현율 | MAR 규칙과 차이 0.05 이내 |
| 하품 오탐 | 시간당 10회 이하 |
| 웹캠 처리 속도 | 15 FPS 이상 |
| 그 밖의 지표 | 값 보고: 규칙 대비 눈 감김 구간 F1, 주의 재현율, 고개 숙임, 오경보율, 경고 지연 |

## 창 단위 평가 기준

`eval/window_eval.py`(창 단위 평가)의 기준이다. 실제 결과를 보기 전에 정했다. 모델 선택은 Val로만 하고, Test는 `--final`일 때만 쓴다.

| 항목 | 기준 |
| --- | --- |
| 입력 | 창별 확률 (N,4). 정답과 시각은 `windows.npz` |
| 눈 감김 판정 | 눈 감김 확률 0.6 이상 켜짐, 0.4 미만 꺼짐, 사이는 이전 상태 유지 |
| 타이머 | `EyeClosureAlarm`, 주의 1.5초, 졸음 3.0초. 창 하나를 0.5초 칸으로 보고 칸 시작과 끝 시각을 모두 넣는다 (창 6개 = 3.0초) |
| 제외된 창 | 시각이 이어지지 않으면 켜짐·꺼짐 상태, 타이머, 정답 이벤트를 모두 끊는다 |
| 정답 이벤트 | `y==1`인 창이 이어진 구간. 길이는 창 수 × 0.5초 |
| 눈 감김 F1 | 구간 단위가 기본(허용 오차 `LABEL_TOLERANCE_SEC` 0.5초), 창 단위를 함께 적는다 |
| 이벤트 재현율 | 3초: 길이 3초 이상 이벤트 중 졸음 경고가 울린 비율. 1.5초: 길이 1.5초 이상 이벤트(3초 이상 포함) 중 주의 경고가 울린 비율 |
| 경고 지연 | 경고 시각 − (이벤트 시작 + 기준 시간)의 평균·중앙값·최대. 놓친 이벤트는 뺀다 |
| 오경보 | 그 단계 기준 이상인 정답 이벤트의 시작부터 끝 + `ALERT_TOLERANCE_SEC`(0.5초) 사이가 아닌 곳에서 울린 경고. 기준보다 짧은 감김에서 울린 경고도 포함한다 |
| 오경보 분모 | 평가 전체 시간 − 정답 눈 감김 시간 |
| `MIN_CLOSE_SEC` | 0.1초. 감김 길이가 창 수 × 0.5초라 0.5초 간격에서는 효과가 없다 |

- `ALERT_TOLERANCE_SEC`를 0.5초로 둔 이유는 창 간격 0.5초가 해상도 한계이기 때문이다. Euro NCAP 0.2초는 참고로만 본다.
- 한계: 0.5초 간격이라 경고 시각이 실제 감김 시작과 최대 0.5초 어긋나고, 지연 수치에 그게 포함된다.

- 판정 기준은 결과를 보기 전에 정했다. Val로 모델·후처리를 고르고, 새 평가 데이터(Test2, 결과를 보지 않고 녹화)는 최종 판정에 한 번만 쓴다.
- 현재 선택: **v7 1D-CNN + 후처리(exit 0.2, min_close 0.5), 하품은 MAR 규칙.** Val(NITYMED 6개) + Val2(웹캠 녹화) 기준 1.5초 재현율 0.894, 눈 감김 구간 F1 0.810, 3초 재현율 0.833이다. 웹캠 실시간 처리는 평균 22.1 FPS였다.
- Test2 최종 판정은 아직 하지 않았다.
- 테스트 코드: `tests/test_v3.py` (`python -m unittest discover -t . -s tests`)

## 😮 하품 전용 모델 비교 기준

눈 감김과 고개 숙임은 규칙에 맡기고 하품만 모델이 맡는 구성을 시험하는 기준이다. 결과를 보기 전(11.05)에 정했다.

- **모델:** `yawn1`. `sig3`와 같은 창에 정답만 하품 여부로 바꾼 이진 1D-CNN·LSTM, 시드 42·43·44. 눈 감김과 겹친 창도 하품이다. 최고 epoch는 모든 출처 val의 하품 F1로 고른다.
- **비교 상대:** MAR 규칙, `sig4` 4클래스 softmax의 하품 출력.
- **채점:** 하품 사건 단위. 검출이 하품 시작 뒤부터 끝 + 인정 시간 안에 겹치면 잡은 것이고, 어떤 하품과도 겹치지 않으면 오탐이다. 세 방법에 같은 인정 시간(1초, 3초)을 쓴다. 주 지표는 1초다.
- **기준값:** 모델 하품 확률 0.6(주), 0.8(참고). 0.4초보다 짧은 모델 검출은 지운다.
- **데이터:** val 9클립, NITYMED 맨입 9건, DMD 맨입 8건, DMD 손 가림 7건. Test는 읽지 않는다.
- **판정:** 맨입 재현율이 규칙보다 낮지 않고 손 가림 재현율이 규칙보다 높으면 "규칙보다 낫다"로 본다. 하품 전용은 4클래스보다 손 가림 재현율이 높고 맨입 재현율과 오탐이 나빠지지 않을 때 채택한다. 차이는 시드 표준편차보다 클 때만 본다.
- **한계:** 손 가림이 운전자 2명, 7건뿐이라 방향만 본다.

## 🎥 자체 영상 예비 비교 기준

결과를 보기 전(11.05)에 정했다. 운전자 1명뿐이라 Test2 최종 판정이 아니라 예비 비교다.

- **데이터:** self01 영상 4개(A 눈 감김, B 맨입 하품, C 손 가린 하품, D 종합). 실내 연기, 사람 검수 라벨. 분할 파일에 넣지 않고 학습에 쓰지 않는다.
- **고정:** 가중치 `sig5`, `yawn2`, 규칙 `best_params.json`, 모델 켜짐 0.6·꺼짐 0.4, 하품 전용 0.6·0.4초. 결과를 보고 바꾸지 않는다.
- **비교:** 실시간 규칙(`LiveRule`과 같은 눈 감김, MAR 하품, 상대 pitch), 4클래스 1D-CNN·LSTM(시드 3개 평균), 최종 구성(눈 감김·고개 숙임은 규칙, 하품은 `yawn2` 1D-CNN 시드 3개 평균).
- **클래스별 지표:** 눈 감김, 하품, 고개 숙임의 구간 F1(기존 `match_segments`). 첫 3초는 뺀다.
- **경고·하품 지표:** 주의(1.5초)·졸음(3초) 재현율과 오경보, 맨입·손 가린 하품 사건 재현율(인정 1초)과 시간당 오탐.
- **한계:** 사건 수가 적어 값은 방향만 본다. 3초 눈 감김이 20개 미만이면 졸음 재현율 90% 목표는 판정하지 않는다.

## 🧪 실험 기록

임계값, 좌표 번호, 시간창, 판정 주기, 라벨 변환 규칙을 바꾸면 여기에 남긴다.

| 날짜 | 바꾼 것 | 이유 | 결과 | 담당 |
| --- | --- | --- | --- | --- |
| 11.02 | 데이터: AI Hub 제외 → NITYMED + DMD | 연출 반복·부자연스러움, 3초 이상 눈 감김·시간 라벨·정상 클립 없음 | 기획서 3-5 | 팀 |
| 11.02 | 모델은 눈 감음만 판정, 감은 시간은 규칙 타이머(`EyeClosureAlarm`)로 측정 | 데모에서 경고가 0.5초 단위로 흔들리고 해제가 평균 0.51초 늦음 | 재측정 예정 | 팀 |
| 11.02 | DMD 변환: `opening`=뜸(A 기준), 평가 때 B 기준도 보고 | `opening`을 감음으로 보면 반쯤 뜬 눈까지 감음으로 학습 | gA_1 최장 감김 A 2.22초 / B 3.46초 | 팀 |
| 11.02 | 라벨 정밀도: 하품 0.5초, 나머지 프레임 단위 | 0.5초로는 깜빡임이 사라지고 1.5초 기준 대비 오차가 큼 | - | 팀 |
| 11.03 | 단계: 2초 경고 → 주의 1.5초·졸음 3초. 타이머 입력을 `closed + t`로 변경, 임계값은 `ear_ratio × 클립 EAR P75`(파일 없으면 0.2), 모델 확률 0.6/0.4 | 기획서 3-3·3-4. 규칙과 모델이 같은 타이머 사용 | eyes4s(3.067초) 주의 5.767초·졸음 7.267초, selftest(1.967초) 주의 7.967초만, yawn 없음 | 김민영 |
| 11.03 | 4클래스 softmax(정상·눈감김·하품·고개숙임), 3초 창·15 FPS·0.5초 간격. 정답은 눈감김·고개숙임 끝 프레임, 하품 50% 이상 | 최종 기획서 5-3 | 모델이 눈감김 외에 하품·고개숙임도 판정 | 육주영 |
| 11.03 | DMD 라벨: `close`만 감음(`closing`·`opening`=뜸), 0.5초 미만 `close`=정상, 손 하품=불확실, 하품은 MAR > 0.3 프레임만(0.5초 미만 끊김은 이음). B 기준 라벨은 만들지 않음 | NITYMED 사람 라벨 기준(0.5초)과 맞춤. DMD 하품이 평균 5.9초로 길고 49%가 손 하품 | DMD 하품 평균 5.93초 → 3.85초 | 육주영 |
| 11.03 | 타이머: 판정 불가·고개 숙임은 직전 감김 0.5초 이상이면 이어 셈. 모델 판정 0.2초 간격 | 최종 기획서 5-1·5-3 | 규칙·모델 모두 같은 타이머 | 육주영 |
| 11.03 | v7: 세션 기준 정규화(영상 시작 60초의 평소 값 대비), NITYMED + DMD 12명 | 카메라 각도·사람 차이 보정 | v6 대비 모든 선택 지표가 나아 v7 선택 | 육주영 |
| 11.03 | 하품 판정 후보 4개 중 MAR 규칙 선택, 화면은 입 다문 뒤 0.5초면 하품 끝 | 모델 하품 구간 F1이 MAR 규칙보다 낮음 | 하품 구간 F1 0.810 → 0.838 | 육주영 |
| 11.04 | 코드를 기획서 6장 폴더로 옮김. `dataset/convert_labels.py`의 DMD 규칙을 최종 라벨에 적힌 규칙대로 복원 | 로컬 원본 파일이 이전 규칙 버전으로 저장되어 있었음 | 다시 만든 DMD 라벨 15개가 최종 라벨과 같음, 테스트 45개 통과 | 육주영 |
| 11.04 | 좌표 번호: `LEFT_EYE`를 362번 쪽으로(팀 코드와 같게), 얼굴 검출 신뢰도 0.3(`config.MIN_FACE_CONFIDENCE`) | v8 학습 창(김민영 특징 CSV)과 같은 입력. 362번 쪽이 블렌드셰이프 `blink_l`과 같은 눈 | 같은 클립에서 김민영 `ear_l`과 기존 `ear_r` 평균 차 0.002. 기존 `data/features`는 좌우가 반대라 v8 계열에 쓸 때 다시 뽑음 | 육주영 |
| 11.04 | v8 1D-CNN: 김민영 창 묶음 `20261103_84e7bf4`(NITYMED 41클립, train 2,878·val 865 창, `norm.npz`), 세션 기준 정규화·DMD·AI Hub 없음. 창 제외는 끝 프레임 불확실·미검출, 불확실+미검출 30% 이상(임시 기준) | 기술서 v3 4.2: LSTM과 같은 창·분할·정규화 | Val macro F1 0.672 / 0.709 / 0.690(시드 42·43·44). 고개숙임은 train 16창이라 F1 0.45~0.52. 사건 평가는 김민영 평가 방식을 받은 뒤 | 육주영 |
| 11.04 | LSTM 풀링 mean → last (`--pool last`, 기본값은 mean 유지) | 눈 감김·고개 숙임 정답은 창 끝 프레임 기준(기술서 2.4)인데 평균 풀링은 창 전체를 섞음 | 시드 42 Val macro F1 0.433 → 0.668. CPU, Test 미사용 | 김민영 |
| 11.05 | 고개 숙임 기준: 정면 pitch보다 17.5도 이상 숙임, 정면 pitch가 아직 없으면 절대 25도(기술서 0장·3.3, 건의 승인) | 정면 pitch가 카메라 위치에 따라 +1~+9도로 달라 절대 기준은 숙임을 늦게 잡고 타이머가 끊김 | v8, grace 0: Val2 3초 재현율 0.559→0.725, Val2 3초 오경보 27.9→33.3회/시간, NITYMED Val 3초 오경보 8.0 유지, 고개숙임 F1 0.488→0.586 | 육주영 |
| 11.05 | 최종 구성: 눈 감김은 규칙과 타이머, 하품은 하품 전용 모델. 눈 감김 F1을 목표에서 값 보고로 바꾸고 하품 목표(손 가림 재현율 MAR 규칙 초과, 맨입 재현율 차이 0.05 이내, 오탐 시간당 10회 이하)를 정함 | val에서 눈 감김은 데이터마다 우열이 갈리고, 하품 전용 모델의 손 가림 재현율이 4클래스보다 높음 | val 1D-CNN(인정 1초, 기준 0.6): 손 가림 0.52 대 4클래스 0.33 대 규칙 0.00, 맨입 NITYMED 0.85·DMD 1.00, 오탐 NITYMED 16·DMD 6.5회/시간. NITYMED 오탐은 규칙도 16회라 목표 미달. Test2 전 변경이고 달성 여부는 Test2로만 판정 | 장범수 |
| 11.05 | 고개 숙임 기준을 `PITCH_DOWN` 15도에서 정면 pitch 대비 17.5도로 변경. 정면 pitch는 세션 시작 60초의 두 눈 EAR 0.2 이상 프레임 pitch 중앙값이고, 아직 없으면 절대 25도(`rule_based/head.py` `HeadDown`, 기술서 3.3) | 정면 pitch가 카메라마다 달라 절대 기준이 맞지 않음 | val 실시간 규칙 주의 재현율·시간당 오경보: NITYMED 0.595/56 → 0.595/48, DMD A 0.815/19.6 변화 없음, B 0.533/6.5 변화 없음. NITYMED 숙임 프레임 검출 val 0.632 → 0.589, train 0.438 → 0.365, 그 외 프레임 오탐 val 0.0057 → 0.0053, train 0.0150 → 0.0094 | 장범수 |
| 11.05 | DMD 변환을 `dataset/dmd_convert.py`로 분리(기술서 5.4). `convert_labels.py`의 옛 규칙(close만, 손 하품=불확실, `reviewed`)과 `--with-dmd` 제거 | 코드가 기술서·라벨 `rules`와 달랐다 | 기존 8클립 재생성: 5클립 동일, 3클립은 `rules`의 사람 수동 수정 자리만 다름. gC_14 재추출 EAR 차이 최대 0.0013 | 장범수 |
| 11.05 | `yawn1`: 하품 전용 이진 1D-CNN·LSTM, `eval/yawn_compare.py`로 MAR 규칙·`sig4` 하품 출력과 사건 단위 비교 | 눈 감김은 규칙, 하품만 모델이 맡는 구성 시험 | README 기준 충족(인정 1초, 기준 0.6, 1D-CNN). 재현율 NITYMED 맨입 0.85 대 규칙 0.89(시드 표준편차 0.06 안), DMD 맨입 1.00 대 0.88, DMD 손 가림 0.52 대 0.00, 4클래스 손 가림 0.33. 오탐 NITYMED 16 대 16, DMD 6.5 대 0회/시간(시드 표준편차 11.3). 기준 0.8 참고값: 손 가림 0.48, DMD 오탐 0. val 기준, 손 가림 7건·운전자 2명 | 장범수 |
| 11.05 | 최종 구성 실시간 연결: 하품은 하품 전용 모델(켜짐 0.6, 0.4초 이상), 눈 감김 경고는 규칙과 타이머, 최종 단계는 눈 감김 단계와 하품 알림 중 높은 쪽(`demo/webcam.py`, `demo/streamlit_app.py`). 하품 알림에 끊김 1.0초 병합과 길이 상한 7초 추가. `windows.npz`에 `yawn` 정답 추가 | 기술서 0장·3.2·4.1 반영 | `data/windows_sig3` 11,944창 정답 일치. `eval.yawn_compare`로 `yawn1` 수치 재현(인정 1초, 1D-CNN 0.6: 손 가림 0.52, 맨입 NITYMED 0.85·DMD 1.00, 오탐 16·6.5). 재학습 시드 42 val 하품 F1 0.742(`yawn1` 0.732). 기존 `windows_sig3`에는 `yawn`이 없어 재학습 전 `build_windows`를 다시 돌려야 함 | 장범수 |
| 11.05 | 영상 데모(`demo/demo_video.py`)와 Streamlit 결과 그래프를 최종 구성으로 연결, 마스크 착용 스위치 추가(`--mask`, Streamlit 사이드바). 자동 마스크 감지는 넣지 않음 | 기술서 3.2·8장 반영. 마스크 영상이 없어 자동 감지를 검증할 수 없음 | 파일과 실시간의 하품 판정이 같음을 테스트로 확인. P1043075 영상으로 데모 생성 확인, 이 클립은 하품 검출이 없어 마스크 동작은 테스트로만 확인 | 장범수 |
| 11.05 | `base1`: 창 요약 특징(평균·표준편차·최솟값·최댓값 44개) + 로지스틱 회귀·랜덤 포레스트 기준선(`sequence_model/baseline.py`, `origin/main`의 코드를 이 구조로 옮김). 데이터 `windows_sig3`, `scikit-learn` 1.9.1 추가 | 기술서 4.2 기준선 | val 창 macro F1(전체/NITYMED/DMD): 로지스틱 0.447/0.472/0.390, 랜덤 포레스트 0.417/0.434/0.533, `sig4` 1D-CNN 0.690/0.700/0.601, LSTM 0.607/0.637/0.566. 두 신경망 모두 기준선보다 높음 | 장범수 |
| 11.05 | 규칙 눈 감음 임계값을 0.48 × P90에서 `best_params.json`의 `ear_ratio`(0.6) × P75로 변경. 파일을 `auto_label/best_params.json`으로 저장소에 포함 | 기술서 3.1·5.6·9.1이 이 파일 하나로 공유하도록 정함 | val 실시간 규칙 주의 재현율·시간당 오경보: NITYMED 0.595/48 → 0.690/48, DMD A 0.815/19.6 → 0.926/26.2, B 0.533/6.5 → 0.600/13.1. 값은 AI Hub 30클립에서 고른 것이고 val로 고르지 않음. 이 행 이전의 규칙 비교 수치는 0.48 기준 | 장범수 |
| 11.05 | 분할과 파이프라인 기준 확정: `feat/seq-pipeline` 구조에 `main` 병합. 김민영 평가를 `eval/window_eval.py`로 옮기고 `train_lstm`·`share_lstm`을 이 구조에 연결. DMD 분할은 train 11명·val 2명 유지(기술서 2.6) | 기술서의 폴더 구조·`data/split.csv`가 `feat/seq-pipeline`과 같음. DMD는 학습에 쓰는 전제이고 손 가린 하품 정답이 DMD에만 있음 | `tests` 59개, `eval.tests.test_evaluate` 55개, `test_train_lstm`·`test_share_lstm` 59개 통과. `train_lstm`의 LSTM은 평균 하나만 쓰는 구조라 `models.LSTMClassifier`와 가중치가 호환되지 않아 파일 안에 둠 | 장범수 |
| 11.05 | `ear_ratio` 0.6 유지. DMD 합본 A 기준 교차 확인(`runs/calib_dmd_1105`, 육주영 PC): 운전자가 겹치지 않는 기존 8클립·신규 8클립에서 찾아 서로 채점 | 기술서 5.5의 재보정 여부 판단 | 찾은 값 0.70·0.61, 전체 0.70. 신규 8: 0.70 F1 0.944·3초 재현율 0.875·오경보 9.8, 0.60 0.946·0.750·4.9. 기존 8: 0.61 0.940·0.500·4.8, 0.60 0.937·0.400·4.8. F1은 0.60~0.75에서 거의 같고 올리면 오경보가 늘어 유지 | 육주영, 장범수 |
| 11.05 | DMD 합본 라벨 반영(Drive `DMD/합본 라벨`, gB_10 10;35 하품 시작 127.037초, gZ_37 눈 감김 122.547~122.782초 삭제), `data/windows_merged`, 재학습 `yawn2`·`sig5`. 최종 구성 가중치를 `yawn2`로 변경 | 사람이 영상으로 확인한 수정을 반영 | 16개 파일 크기가 Drive와 같음. 창 11,944개, 입력·정규화 값 같음, 정답은 train 2창만 달라짐. 인정 1초 `yawn2` 1D-CNN 0.6: 손 가림 0.48, 맨입 NITYMED 0.89·DMD 1.00, 오탐 16·6.5회/시간. `sig5` 4클래스 1D-CNN 손 가림 0.33. `yawn1` 대비 손 가림 0.52→0.48은 7건·시드 3개에서 나온 차이라 판단하지 않음 | 장범수 |
| 11.05 | 자체 영상 예비 비교(`eval/self_compare.py`): self01 4영상 약 15분, 규칙·`sig5`·최종 구성(`yawn2`) | 팀이 각자 같은 기준으로 돌리고 개수를 합치기 위함 | 구간 F1: 하품 규칙 0.67·1D-CNN 0.80·LSTM 0.73·최종 0.82, 고개 숙임 0.31·0.16·0.45, 눈 감김 0.12·0.26·0.53(라벨에 깜빡임이 없어 낮음, 사후 0.5초 이상 구간 0.57·0.66·0.63). 손 가린 하품 재현율(8건) 규칙 0.00, 1D-CNN 0.50, LSTM 0.54, 최종 0.58. 처음에는 화면 표시용 `shown_yawn`으로 채점해 4클래스가 0.00으로 나왔고, 입을 다문 것으로 보여 손 가린 하품이 지워지는 오류라 모델 출력(기준 0.6, 0.4초 이상)으로 고쳐 다시 채점했다. 졸음 재현율 모두 10/10, 시간당 오경보 40.4·24.3·22.9. 규칙 오경보 10건 중 5건은 하품 중 EAR 하락, 3건은 고개 숙인 구간, 2건은 EAR이 매우 낮아 라벨 확인 필요. 운전자 1명이라 예비값, 결과를 보고 가중치·임계값을 바꾸지 않음 | 장범수 |

### 팀 코드와 다른 점 (PR에서 정할 것)

이 브랜치는 `landmarks/`, `rule_based/`, `sequence_model/build_windows.py`를 v3 버전으로 바꿨다. 그래서 팀원 테스트(`landmarks/tests`, `rule_based/tests`, `sequence_model/tests/test_build_windows.py`)가 실패한다.

- **좌우 눈 번호:** 11.04부터 팀 코드와 같다(`LEFT_EYE = (362, …)`). v7까지의 가중치와 기존 `data/features`는 이전 순서(`LEFT_EYE = (33, …)`)로 만들었다.
- **특징 열:** v3 `landmarks/features.py`는 `ear, mar, pitch, yaw, roll`을 계산하고, `face`·`ear_l`·`ear_r`·blendshape는 `landmarks/extract.py`가 만든다.
- **단계 이름:** v3 `rule_based/alarm.py`에는 `CAUTION`이 없고 `WARN`·`DROWSY`를 쓴다.
- **창 제외 사유:** v3 `build_windows.py`에는 `EXCLUDE_END_MISSING` 같은 제외 사유 상수가 없다.

## 📄 출처 표기

- NITYMED: N. Petrellis et al., "High Speed Implementation of the Deformable Shape Tracking Face Alignment Algorithm", Euromicro DSD 2021.
- DMD: J. D. Ortega et al., "DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and Alertness Analysis", ECCV Workshops 2020.

## 🤝 협업 규칙

- 팀 공통 규칙(브랜치·커밋·Claude 작업 원칙): 전역지침 `~/.claude/CLAUDE.md`
- 이 프로젝트 규칙: `CLAUDE.md`
- `main`에 직접 push하지 않는다. `feat/`, `fix/`, `docs/` 브랜치에서 PR로 병합한다. 커밋은 `YYMMDD 태그 : 내용`.

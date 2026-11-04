이 레포에만 적용되는 지침이다. 팀 공통 규칙(브랜치·커밋·작업 원칙)은 전역지침 `~/.claude/CLAUDE.md`를 따르고, 둘이 다르면 이 파일이 우선한다. 배경·데이터·평가는 최종 기획서(2026-11-03)와 `README.md`를 본다.

졸음운전 감지: MediaPipe 얼굴 특징 → 규칙 기반 판정(MVP) → 1D-CNN·LSTM과 같은 입력으로 비교.

## 설계 규칙

- MediaPipe는 **Tasks API**로 쓴다(Solutions API 금지). 파일과 실시간 모두 `VIDEO` 모드로 프레임 시각을 넘겨 처리한다(`landmarks/extract.py`). `LIVE_STREAM` 전환은 [미정].
- 좌표는 17개(눈 12, 입 4, 코끝 1). 고개 각도는 점이 아니라 얼굴 변환 행렬에서 계산한다. 좌표 번호는 `landmarks/points.py`에 둔다.
- 프레임별 특징 11개: `face, ear_l, ear_r, ear, mar, pitch, yaw, roll, blink_l, blink_r, jaw_open`.
- 단계는 연속 눈 감김 시간으로 정한다: **주의 1.5초, 졸음 3초**.
- **모델 출력은 4클래스 softmax(정상, 눈감김, 하품, 고개숙임)다.** 불확실은 클래스가 아니라 제외 표시다. 고개숙임은 눈감김과 분리한다.
- 감은 시간은 창 개수로 세지 말고 `rule_based/alarm.py`의 `ClosureTimer`처럼 **프레임 시각(`프레임 번호 / fps`) 기준 타이머**로 잰다. 규칙과 모델은 같은 타이머와 같은 설정값을 쓴다.
- 판정 불가와 고개 숙임은 직전까지 이어 감은 시간이 0.5초 이상이면 감은 것으로 이어 센다. 판정 불가가 3초 이어지면 확인 필요 경고를 낸다.
- 하품은 5분에 3회면 하품 주의(알림 후 2분 쿨다운). 1회로는 단계를 바꾸지 않는다. 고개 숙임은 표시만 한다.
- 시간창은 **3초, 공통 15 FPS(45 × 11)**. 학습 창은 0.5초 간격, 실시간 판정은 0.2초 간격이다. 데이터마다 FPS가 다르므로(NITYMED 25, DMD 29.76, 웹캠) 15 FPS로 리샘플한다.
- 창 정답: 눈감김·고개숙임은 창 끝 프레임, 하품은 창의 50% 이상. 끝 프레임이 불확실이면 창을 뺀다.
- 설정값은 루트 `config.py`(창·클래스·학습)와 `rule_based/config.py`(규칙 임계값·시간 기준)에 둔다. 코드에 직접 쓰지 않는다.
- 얼굴 미검출 프레임이 30%를 넘는 창은 빼고 뺀 비율을 기록한다.
- 데모 타임라인 x축은 영상 시각 기준으로 그리고, 화면 값(EAR 등)은 현재값을 표시한다.

## 데이터·라벨 규칙

- 라벨은 클립마다 JSON 1개. 학습·평가 정답은 채널 라벨을 쓴다:
    - `eye_closed`: `[시작 초, 끝 초]` 목록, **프레임 단위**.
    - `head_down`: `[시작 초, 끝 초]` 목록, **프레임 단위**. 숙여서 눈이 안 보이는 구간은 여기에 넣고 `eye_closed`에서 뺀다.
    - `yawn`: `{start, end, hand}` 목록, **0.5초 단위**.
    - `uncertain`: 학습·평가에서 뺀다.
    - `segments`는 보기·호환용(겹치면 불확실 > 눈감김 > 고개숙임 > 하품 > 정상).
    - 그 밖: `clip`, `source`, `fps`, `frames`, `driver_id`, `conditions`, `rules`, `label_status`.
- DMD 라벨은 `dataset/convert_labels.py`로만 변환한다: `close`만 감음(0.5초 미만은 정상), `closing`·`opening`·`open`은 뜸, `undefined`와 손 하품은 불확실, 하품은 MAR > 0.3 프레임만 남긴다. DMD에는 고개 숙임 라벨이 없다.
- 사람 라벨(재검수, 고개 숙임 표시, 웹캠 정답, 운전자 번호)은 사람이 만든다. Claude가 대신 표시하지 않는다.
- 분할은 **운전자(`driver_id`) 단위**로 하고, 저장한 분할 파일(`data/split.csv`)을 모든 실험이 같이 쓴다. 같은 클립의 창이 학습·평가에 나뉘지 않는다. 정규화 값은 train으로만 구한다.
- 평가 기준은 결과를 보기 전에 README에 먼저 적는다. 최종 판정 데이터(Test2)는 한 번만 쓰고, 그것으로 모델이나 임계값을 고르지 않는다.

## 실행 순서

`auto_label.extract` → `auto_label.head_review.server`(사람 검수) → `dataset.convert_labels` → `sequence_model.build_windows` → `sequence_model.train` → `eval.evaluate` → `eval.report` → `demo/`

- 명령은 레포 루트에서 `python -m <폴더>.<파일>`로 실행한다. 자세한 명령은 `README.md`의 실행 절을 본다.
- 테스트: `python -m unittest discover -t . -s tests`
- 우선순위: ① 규칙 MVP → ② 1D-CNN·LSTM 비교 → ③ (선택) ONNX.

## 하지 말 것

- 영상·특징·라벨·가중치·`models/face_landmarker.task`를 레포에 올리지 않는다(비공개 Google Drive로 공유).
- DMD로 가공한 자료(잘라낸 영상, 변환 라벨, 오버레이 영상, DMD로 학습한 가중치)를 공개하지 않는다. 공개용 데모 영상은 NITYMED·자체 영상으로 만든다.
- README 라이브러리 목록에 없는 라이브러리를 팀 공유 없이 추가하지 않는다.

## 기록

- 좌표 번호, 임계값, 시간창, 판정 주기, 라벨 변환 규칙을 바꾸면 이유와 결과를 `README.md`의 **실험 기록** 표에 남긴다.
- 다시 학습할 때는 `--tag`에 새 이름을 쓴다. 가중치는 `runs/<tag>/<모델>_v01.pt`(`cnn1d_v01.pt`, `lstm_v01.pt`).

## 환경

- conda 환경 `geumjjogi-drowsy`. 고정 버전은 `README.md`의 라이브러리 표를 따른다.
- 모델 학습은 GPU로 한다. 특징 추출은 CPU로 한다.
- 좌표 추출은 오래 걸리므로 나눠 실행하고 결과 표(`data/features/*.parquet`)를 저장해 재사용한다.

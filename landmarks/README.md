# landmarks — 얼굴 랜드마크 추출과 특징 계산

담당: 김민영.

## 맡은 일 (기획서 7절)
- 랜드마크 17개 좌표 선정과 추출
- 시간 정보를 붙인 데이터프레임 생성
- 특징 계산 (EAR·MAR·고개 각도)

## 파일
| 파일 | 하는 일 |
| --- | --- |
| `points.py` | 좌표 17개 번호 (눈 12, 입 4, 코끝 1). 점을 바꾸면 이 파일만 고친다 |
| `extract.py` | 영상 → Face Landmarker → 프레임별 시각·검출 여부·좌표·고개 각도·블렌드셰이프 표 (CSV) |
| `features.py` | 좌표 표에 `ear_l`·`ear_r`·`ear`·`mar` 열을 더한다 (`add_features()`). 모델 입력 11개 순서는 `config.FEATURES`이고, `face`는 `sequence_model/build_windows.py`의 `feature_matrix()`가 `face_found`로 만든다 |

## 입력 / 출력
- 입력: 영상 파일 (mp4 등). 실시간 데모는 `LandmarkExtractor.process()`에 프레임을 하나씩 넣는다
- 출력: 프레임마다 한 행. 얼굴이 안 잡히면 좌표·각도·블렌드셰이프·EAR·MAR는 NaN (`face_found`는 False)

| 열 | 형식 | 설명 |
| --- | --- | --- |
| `t` | 초 | 영상 시각 = 프레임 번호 ÷ 원본 FPS (15 FPS 맞춤은 `sequence_model`) |
| `face_found` | bool | 얼굴 검출 여부 |
| `{이름}_x`, `{이름}_y` | 픽셀 | 좌표 17개. 이름은 `le1~le6`, `re1~re6`, `mouth_l/r/t/b`, `nose` |
| `pitch`, `yaw`, `roll` | 도 | 고개 각도. pitch는 **숙이면 +, 들면 −** |
| `blink_l`, `blink_r`, `jaw_open` | 0~1 | 블렌드셰이프 `eyeBlinkLeft`, `eyeBlinkRight`, `jawOpen` |
| `face` | 0/1 | `face_found`의 숫자형. 모델 입력용. 이 표에는 없고 `feature_matrix()`가 만든다 |
| `ear_l`, `ear_r`, `ear` | 비율 | 눈 종횡비 (좌, 우, 평균). 좌우는 `auto_label.py`와 같다. `add_features()`가 더함 |
| `mar` | 비율 | 입 종횡비. `add_features()`가 더함 |

## 실행
`models/face_landmarker.task`가 있어야 한다 (저장소에 올리지 않음).
```bash
python -m landmarks.extract <영상.mp4> -o outputs/landmarks
```

## 웹캠 검증 (2026-10-30, 1명)
얼굴 검출률 100%, 29.5 FPS. pitch는 숙이면 약 +20도, 들면 약 −20도.

## 변경 내용 (2026-11-02)
학습 데이터를 만든 `auto_label.py`(Drive)와 특징을 맞췄다. 웹캠 영상 3개에서 프레임별로 비교한 결과 EAR 차이는 0.002 이하, 각도 차이는 0.022도 이하다 (OpenCV 4.10.0, 얼굴 검출률 둘 다 100%).
- 블렌드셰이프 `blink_l`, `blink_r`, `jaw_open` 추가 (`extract.py`). 실시간 입력에 11개 특징을 모두 넣을 수 있다
- 얼굴 검출 신뢰도 세 가지를 0.3으로 맞춤 (`extract.py`)
- 왼쪽·오른쪽 눈 좌표를 `auto_label.py`와 같게 맞바꿈. 바뀌기 전에는 `ear_l`과 `ear_r`이 학습 데이터와 반대였다 (`points.py`)
- `face`(0/1) 열 추가, 모델 입력 11개 순서를 `FEATURE_COLUMNS`로 고정 (`features.py`). `extract.py`의 `face_found`는 그대로 둔다
- 고개 각도 공식은 바꾸지 않았다. 합성 회전에서 `auto_label.py`와 값이 같았다
- 테스트 추가: `python -m unittest discover -t . -s landmarks/tests`

## 정할 것
- AI Hub 영상에서 얼굴이 작으면 바운딩박스로 잘라 넣을지 (샘플 데이터로 확인)
- 안경·조명·다른 사람에서 검증

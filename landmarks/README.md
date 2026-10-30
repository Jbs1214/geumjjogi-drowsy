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
| `extract.py` | 영상 → Face Landmarker → 프레임별 시각·검출 여부·좌표·고개 각도 표 (CSV) |
| `features.py` | 좌표 표에 EAR·MAR 열을 더한다 (`add_features()`) |

## 입력 / 출력
- 입력: 영상 파일 (mp4 등). 실시간 데모는 `LandmarkExtractor.process()`에 프레임을 하나씩 넣는다
- 출력: 프레임마다 한 행. 얼굴이 안 잡히면 좌표·각도·특징은 NaN

| 열 | 형식 | 설명 |
| --- | --- | --- |
| `t` | 초 | 영상 시각 = 프레임 번호 ÷ 원본 FPS (15 FPS 맞춤은 `sequence_model`) |
| `face_found` | bool | 얼굴 검출 여부 |
| `{이름}_x`, `{이름}_y` | 픽셀 | 좌표 17개. 이름은 `le1~le6`, `re1~re6`, `mouth_l/r/t/b`, `nose` |
| `pitch`, `yaw`, `roll` | 도 | 고개 각도. pitch는 **숙이면 +, 들면 −** |
| `ear_l`, `ear_r`, `ear` | 비율 | 눈 종횡비 (좌, 우, 평균). `add_features()`가 더함 |
| `mar` | 비율 | 입 종횡비. `add_features()`가 더함 |

## 실행
`models/face_landmarker.task`가 있어야 한다 (저장소에 올리지 않음).
```bash
python -m landmarks.extract <영상.mp4> -o outputs/landmarks
```

## 웹캠 검증 (2026-10-30, 1명)
얼굴 검출률 100%, 29.5 FPS. pitch는 숙이면 약 +20도, 들면 약 −20도. 자세한 값은 `docs/landmarks_검증결과.md`.

## 정할 것
- AI Hub 영상에서 얼굴이 작으면 바운딩박스로 잘라 넣을지 (샘플 데이터로 확인)
- 안경·조명·다른 사람에서 검증

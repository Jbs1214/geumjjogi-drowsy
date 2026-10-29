# sequence_model — 10초 시간창 시퀀스 분류기

담당: 육주영. 지금은 **파일 뼈대만** 있다. 함수는 모두 `NotImplementedError`를 낸다.

## 맡은 일 (기획서 7절)
- AI Hub 라벨의 정상/주의/졸음 대응 정의
- 10초 시간창 정의와 데이터 구성
- 시퀀스 분류기 학습
- 평가 지표와 비교표 (`eval/`)

## 파일
| 파일 | 하는 일 |
| --- | --- |
| `config.py` | 창 길이·FPS·이동 간격·클래스 등 설정값 |
| `labels.py` | AI Hub 라벨 → 정상/주의/졸음, 탑승자 단위 분할 |
| `windows.py` | 특징 시계열 → 15 FPS 맞춤 → 10초 창 + 길이 마스크 |
| `models.py` | 분류기 생성 (모델 미정) |
| `train.py` | 학습 → 가중치 저장 |
| `predictor.py` | 가중치 불러와 창 하나의 클래스 확률 계산 (데모에서 사용) |

## 시간창 (기획서 3-3)
| 항목 | 값 |
| --- | --- |
| 창 길이 | 10초 (클립이 짧으면 클립 길이에 맞춤) |
| 프레임 속도 | 15 FPS → 150프레임 |
| 이동 간격 | 학습·평가 5초, 실시간 1초 |
| 입력 | 150행 × 특징 열 (17개 좌표가 아니라 계산된 특징) |
| 분할 | 탑승자 단위 |

## 입력 인터페이스
`landmarks/`가 만드는 특징 DataFrame(영상 시각 `t`, 얼굴 검출 여부, EAR·MAR·고개 각도 등)을 받는다.
열 이름은 김민영 님과 확정한 뒤 `config.FEATURE_COLUMNS`에 맞춘다.

## 정할 것
- 학습 모델: **미정** (논문 검토 후 결정)
- '정상' 클래스 정의: 이 데이터에는 '운전하다'만 있는 장면이 없다.
  후보안은 "졸음 관련 행동 0개 + 키프레임 5장 중 '운전하다' 3장 이상"
- 2클래스 / 3클래스: '주의' 구간이 충분한지 확인 후 결정
- 얼굴 미검출 기준: 샘플 영상으로 확정

## 참고 논문 (기획서 3-6)
- Ghoddoosian et al. (2019), A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection
- Bai et al. (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

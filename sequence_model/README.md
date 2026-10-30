# sequence_model — 시간창 구성과 1D-CNN·LSTM

담당: 육주영. 개정 기획서(2026-10-30) 기준으로 구현했다.

**지금 상태:** 코드는 합성 데이터로만 검증했다. 팀 산출물(자동 라벨, 검수 라벨)을 받기 전이라 **실데이터 학습 결과는 아직 없다.** 테스트 점수는 성능 수치가 아니다.

## 맡은 일 (기획서 7절)
- 시간창·데이터셋 구성, 클립 단위 분할
- 1D-CNN·LSTM 학습
- 평가 지표와 비교표 (`eval/`)

## 파일
| 파일 | 하는 일 |
| --- | --- |
| `config.py` | 창 길이·FPS·이동 간격·특징·상태 등 설정값 |
| `labels.py` | 자동·검수 라벨 JSON 읽기, 구간 → 프레임 상태, SGA 그룹 단위 분할. **JSON 파싱은 이 파일에만 있다** |
| `windows.py` | FPS 맞춤, 얼굴 미검출 처리, 창 자르기, 데이터셋 구성, 정규화 |
| `models.py` | 1D-CNN(`cnn1d`), LSTM(`lstm`) |
| `baseline.py` | 창 요약 특징 + 로지스틱 회귀·랜덤 포레스트 기준선 (scikit-learn) |
| `train.py` | 학습 → `checkpoints/best_model_vNN.pt` 저장 |
| `predictor.py` | 가중치를 불러와 창의 클래스 확률 계산 (데모에서 사용) |
| `tests/` | 단위 테스트와 합성 데이터 종단 테스트 |

## 시간창 (기획서 3-3)
| 항목 | 값 |
| --- | --- |
| 창 길이 | 2초(60프레임). 후보 2~3초, `--window-sec`로 바꾼다 |
| 프레임 속도 | 30 FPS. 다른 FPS 입력은 시각 기준으로 가장 가까운 프레임을 가져와 맞춘다 |
| 이동 간격 | 학습·평가·실시간 모두 0.5초 |
| 입력 모양 | `(창 프레임, 11)`. 11개 특징 순서는 `config.FEATURE_COLUMNS` |
| 클래스 | 0 정상, 1 눈감김·고개숙임, 2 하품. 3 불확실은 클래스가 아니라 창을 뺀다 |
| 창 정답 | 창 안 상태 비율이 `LABEL_RATIO`(0.5) 이상. 눈감김, 하품, 불확실 순서로 본다 |
| 짧은 클립 | 창보다 짧으면 제외 |
| 얼굴 미검출 | 특징은 0, `face`는 0. 정상으로 표시된 프레임은 불확실로 바꾼다 |
| 정규화 | train 분할의 얼굴 검출 프레임으로 평균·표준편차 계산. 값은 가중치 파일에 함께 저장 |
| 분할 | SGA 번호 그룹 단위. 보정에 쓴 클립의 그룹은 항상 train |

LSTM은 마지막 상태가 아니라 전체 출력의 평균을 쓴다. 합성 데이터에서 마지막 상태는 수렴에 약 50 epoch가 걸려 조기 종료에 걸렸고, 평균은 세 시드 모두 6 epoch 안에 수렴했다. 실데이터에서 다시 확인한다.

## 입력 형식 (가정)
샘플을 받기 전이라 기획서의 필드 이름으로 읽는다. 다르면 `labels.py`만 고친다.

| 파일 | 가정한 형식 |
| --- | --- |
| 자동 라벨 `<클립>_자동라벨.json` | `X`: 프레임별 11개 값(미검출은 null), 순서는 `FEATURE_COLUMNS`. `y`: 프레임별 0~3. `fps`: 없으면 30 |
| 검수 라벨 `<클립>.json` | `segments`: `{"start": 초, "end": 초, "state": 상태 이름}` 목록. 상태 이름은 `labels.STATE_BY_NAME` |
| 보정 클립 목록 `.txt` | 한 줄에 클립 번호 하나 |

라벨 폴더 기본값은 `data/processed/labels_auto`다. 이 경로는 `.gitignore`에 있다. 라벨 JSON은 다른 폴더에 두면 깃에 올라갈 수 있으니 주의한다.

## 실행
```bash
python -m sequence_model.train --model cnn1d --calibration-clips <보정 클립 목록.txt>
python -m sequence_model.train --model lstm --window-sec 3 --calibration-clips <보정 클립 목록.txt>
python -m unittest discover -t . -s sequence_model/tests    # 테스트
```
학습 점수의 정답은 자동 라벨이라 규칙을 얼마나 따라 배웠는지만 보여 준다. 평가는 `eval/`에서 검수 라벨로 한다.

학습은 GPU로만 한다. GPU가 없으면 `train.py`는 시작하지 않고 멈추며, CPU로 바꾸는 옵션은 없다. 종단 테스트도 GPU가 없으면 건너뛴다. 기준선(scikit-learn)은 GPU를 쓰지 않는다.

## 실시간 사용
```python
predictor = SeqPredictor(Path("checkpoints/best_model_v01.pt"))
recent = fill_missing(resample(최근_프레임_표, predictor.fps))     # 열: t, FEATURE_COLUMNS
window = recent[list(config.FEATURE_COLUMNS)].to_numpy(dtype="float32")[-window_frames(predictor.window_sec):]
proba = predictor.predict_proba(window)                             # 0.5초마다 호출
```

## 팀 작업이 필요한 것
- 자동 라벨·검수 라벨 샘플, 보정 30클립 목록 (김민영): 있어야 실데이터 학습과 형식 확인을 할 수 있다
- 추가 검수 클립: 지금 검수 라벨이 있는 30클립은 보정에 쓴 클립이라 평가에 못 쓴다. test 그룹에서 골라 검수해야 한다
- 실시간 특징 11개 (김민영): `landmarks/extract.py`에 `blink_l`, `blink_r`, `jaw_open`이 없어 이대로는 웹캠 입력을 모델에 넣을 수 없다
- scikit-learn: 팀 라이브러리 목록에 없어 공유가 필요하다

## 참고 논문 (기획서 3-6)
- Ghoddoosian et al. (2019), A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection
- Bai et al. (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

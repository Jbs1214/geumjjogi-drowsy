# 졸음운전 경고 (geumjjogi-drowsy)

YOLO로 눈·입 상태를 학습하고, 웹캠으로 추론해 졸음이면 경고음을 낸다.

- 클래스 4개: `eye_open`, `eye_closed`, `mouth_open`, `mouth_closed`
- 화면: OpenCV 창

## 환경 설정

```bash
conda activate geumjjogi-drowsy
```

사용 라이브러리: `ultralytics`, `torch`·`torchvision` (CUDA 11.8용), `opencv-python`, `pygame`

## 실행 방법

| 단계 | 파일 | 하는 일 |
| --- | --- | --- |
| 1 | `convert.py` | AI Hub 라벨 JSON → YOLO 형식 변환 (4개 클래스) |
| 2 | `train.py` | YOLO 학습 → 가중치 저장 |
| 3 | `app.py` | 웹캠 추론 → 졸음이면 경고음 (OpenCV 창) |

```bash
python convert.py
python train.py
python app.py
```

## 데이터

- 출처: AI Hub #173 「졸음운전 예방을 위한 운전자 상태 정보 영상」
- 원본 데이터와 학습 가중치(`best_model_vXX.pt`)는 레포에 올리지 않는다. Google Drive에서 공유한다.

## 팀원

| 역할 | 이름 |
| --- | --- |
| 팀장 | 장범수 |
| 팀원 | 김민영 |
| 팀원 | 육주영 |

# 영상·웹캠 프레임 → Face Landmarker → 프레임마다 시각·검출 여부·17개 좌표·고개 각도 한 행 (기획서 3-1 ①~③).
# 실시간 데모도 LandmarkExtractor.process()를 그대로 쓴다.
#
# 실행: python -m landmarks.extract <영상.mp4> [-o outputs/landmarks]

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision

from landmarks.points import POINTS

# 모델 파일은 저장소에 올리지 않는다. Google Drive에서 받아 models/에 둔다.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"

COORD_COLUMNS = [f"{name}_{axis}" for name in POINTS for axis in ("x", "y")]
POSE_COLUMNS = ["pitch", "yaw", "roll"]     # 도(degree)
COLUMNS = ["t", "face_found", *COORD_COLUMNS, *POSE_COLUMNS]


def matrix_to_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """얼굴 변환 행렬 (4, 4) → (pitch, yaw, roll) 도 단위."""
    angles, *_ = cv2.RQDecomp3x3(matrix[:3, :3])
    return tuple(float(a) for a in angles)


class LandmarkExtractor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        # 고개 각도는 17개 좌표 대신 모델이 내는 얼굴 변환 행렬에서 구한다 (점 수가 적어 solvePnP가 불안정하다)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> dict:
        """BGR 프레임 하나와 영상 시각 t(초) → 표 한 행. 얼굴이 없으면 좌표·각도는 NaN."""
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        # VIDEO 모드는 타임스탬프가 계속 커져야 한다
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)

        row = {"t": t, "face_found": bool(result.face_landmarks)}
        if not result.face_landmarks:
            row.update(dict.fromkeys(COORD_COLUMNS + POSE_COLUMNS, np.nan))
            return row

        # 정규화 좌표를 픽셀로 바꾼다. 가로·세로 비율이 달라도 EAR·MAR가 찌그러지지 않게
        landmarks = result.face_landmarks[0]
        for name, idx in POINTS.items():
            row[f"{name}_x"] = landmarks[idx].x * w
            row[f"{name}_y"] = landmarks[idx].y * h
        row.update(zip(POSE_COLUMNS, matrix_to_angles(np.asarray(result.facial_transformation_matrixes[0]))))
        return row

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def extract_video(path: Path, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """영상 파일 → 프레임별 표 (열: COLUMNS). 시각 t는 프레임 번호 ÷ 원본 FPS. 15 FPS 맞춤은 sequence_model에서 한다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise ValueError(f"영상 FPS를 읽을 수 없습니다: {path}")

    rows = []
    with LandmarkExtractor(model_path) as extractor:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rows.append(extractor.process(frame, i / fps))
            i += 1
    cap.release()
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="영상에서 랜드마크 17개·고개 각도 표(CSV)를 만든다")
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("outputs/landmarks"))
    args = parser.parse_args()

    df = extract_video(args.video)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.video.stem}.csv"
    df.to_csv(out, index=False)
    print(f"{out}: {len(df)}프레임, 얼굴 검출 {df['face_found'].mean():.1%}")


if __name__ == "__main__":
    main()

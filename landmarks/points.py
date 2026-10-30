# 졸음 판단에 쓰는 Face Landmarker 좌표 17개 (기획서 3-1). 478개 중 이 번호만 표에 남긴다.
# 번호는 MediaPipe 얼굴 메시 기준이다. 점을 바꾸거나 늘리면 이 파일만 고친다.

# 눈 6개씩, EAR 순서 p1~p6 (p1·p4 = 가로 양끝, p2-p6·p3-p5 = 세로 쌍). 좌우는 화면 기준.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

# 입 4개: 좌우 입꼬리, 윗입술·아랫입술 안쪽 가운데
MOUTH = {"mouth_l": 61, "mouth_r": 291, "mouth_t": 13, "mouth_b": 14}

NOSE = {"nose": 1}

# 이름 → 랜드마크 번호. 좌표 열 이름은 f"{이름}_x", f"{이름}_y" (픽셀 단위)
POINTS: dict[str, int] = {
    **{f"le{i}": idx for i, idx in enumerate(LEFT_EYE, 1)},
    **{f"re{i}": idx for i, idx in enumerate(RIGHT_EYE, 1)},
    **MOUTH,
    **NOSE,
}

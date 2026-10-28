These are local guidelines that apply to this repo only. Team-wide rules (branches, commits, working principles) follow `~/.claude/CLAUDE.md` (global guidelines). If this file conflicts with the global guidelines, this file takes precedence.

- **Repo:** https://github.com/Jbs1214/geumjjogi-drowsy
- **Topic:** Drowsy driving warning

---

## 1. Approach

**Train YOLO on eye and mouth states, run inference on a webcam, and play a warning sound when the driver is drowsy.**

- Model: YOLO (`ultralytics`). Model size is [TBD].
- 4 classes: `eye_open`, `eye_closed`, `mouth_open`, `mouth_closed`
- Warning threshold: [TBD: how many seconds of closed eyes trigger a warning]
- Display: OpenCV window

## 2. Data

**Do not put the raw data in the repo.**

- Source: AI Hub #173 "Driver State Information Videos for Preventing Drowsy Driving" (졸음운전 예방을 위한 운전자 상태 정보 영상)
- Training: images (21GB) + label JSON. Use only the data that has bounding box labels (real roads and controlled environments). Number of images to use is [TBD].
- Videos (100GB): back up to an external hard drive. Whether to use them for training is [TBD].
- Only 1 person keeps the raw data (images and videos): [TBD: data keeper]. Whether to upload to Google Drive will be decided after discussing with the team [TBD].

## 3. Environment

- conda environment: `geumjjogi-drowsy`
- Shared environment settings (Python, GPU, CUDA, etc.) follow the global guidelines in `~/.claude/CLAUDE.md`.

## 4. How to Run

| Step | File | What it does |
| --- | --- | --- |
| 1 | `convert.py` | Convert AI Hub label JSON to YOLO format (4 classes) |
| 2 | `train.py` | Train YOLO and save the weights |
| 3 | `app.py` | Run webcam inference and play a warning sound when drowsy (OpenCV window) |

```bash
conda activate geumjjogi-drowsy
python convert.py
python train.py
python app.py
```

## 5. Weight Versioning

- Naming: `best_model_v01.pt`, `best_model_v02.pt`, ... Bump the version on every training run.
- Do not upload weights to GitHub. Share them on Google Drive.

## 6. Libraries

- **Agreed to use:**
    - `ultralytics` — YOLO training and inference
    - `torch`, `torchvision` (for CUDA 11.8) — GPU training. Versions are [TBD: record after confirming installation]
    - `opencv-python` — read webcam frames and display output
    - `pygame` — play the warning sound
- **Agreed not to use:** [TBD]

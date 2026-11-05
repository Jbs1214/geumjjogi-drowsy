# CLAUDE.md

These instructions apply to this repo only. Team-wide rules (branches, commits, working principles) follow `~/.claude/CLAUDE.md`; if the two differ, this file takes precedence.

Drowsy driving detection: MediaPipe face features → rule-based judgment (MVP) → compare against 1D-CNN and LSTM on the same input. Final setup: rule and timer for eye closure, yawn-only binary model for yawns; the 4-class model stays for the rule comparison.

## Where to look

Read the relevant file directly when needed. If it disagrees with this file, follow the source order below and fix this file.

- Numbers (window, FPS, thresholds, time criteria): `config.py`, `rule_based/config.py`, `auto_label/best_params.json`  
- DMD label conversion rules: `docs/spec.md` (technical specification) section 5.4 and `rules` in the converted label JSON. The JSON produced by `dmd_convert` is the reference for DMD labels; `dataset/convert_labels.py` must not apply DMD rules on its own (if it differs, fix the code).  
- Design background, judgment logic, label JSON format, evaluation plan: `docs/plan.md` (project proposal), `docs/spec.md` (technical specification)  
- Run commands, library versions, experiment log: `README.md`

## Design rules

- Use the MediaPipe **Tasks API** (the Solutions API is forbidden). For both files and real time, use `VIDEO` mode and pass the frame timestamp.  
- Compute head angles from the face transformation matrix, not from landmark points.  
- Drowsiness stage is decided by **continuous eye-closure time**. Do not measure closure time by counting windows; measure it with a **timer based on frame time (frame number / fps)**, like `ClosureTimer` in `rule_based/alarm.py`. The rule-based path and the model use the same timer and the same settings.  
- Models answer only the current state: the 4-class model gives normal, eye closure, yawn, or head-down, and the yawn model gives yawn or not. The timer measures duration. Yawn targets are in `docs/spec.md`. **Uncertain is an exclusion mark, not a class.**  
- If the eyes close during a yawn, judge it as eye closure (priority: eye closure \> yawn) and keep the timer running. Do not exclude eye closure during a yawn from the timer.  
- Do not hard-code numbers, thresholds, or time criteria; keep them in config files.  
- Draw the demo timeline x-axis in video time, and show current values (EAR, etc.) on screen.

## Data and label rules

- DMD eye closure counts `close` and `closing` as closed and `opening` as open (Basis A). Evaluation also reports Basis B, which counts `opening` as closed too.  
- Blinks also count as eye closure (short closures stay in `eye_closed`). Existing NITYMED labels, however, keep blinks as normal; re-labeling is postponed until after the presentation, and newly reviewed files use this rule. Keeping blinks from becoming warnings is the timer's job.  
- Spans where the head is lowered so far that the eyes are not visible go in `head_down` and are removed from `eye_closed`.  
- Make DMD labels only with the conversion tool, and record the conversion rules in `rules`. If the conversion code differs from `docs/spec.md` section 5.4 or the label's `rules`, fix the code. Keep `label_status` of converted DMD labels as `auto`.  
- AI Hub data is not used for now. The team will announce it if that changes.  
- Human labels (re-review, head-down marking, webcam ground truth, driver IDs) are made by humans. Claude does not mark them on their behalf.  
- Auto labels (`auto_label`) are the output of rules, so **never use them as evaluation ground truth**.  
- Do not change the `driver_id` of labels that already exist (NITYMED `md01` format; DMD uses the driver number from the file name). Never let two sources produce the same string.  
- Split by **driver (`driver_id`)**, and have every experiment share the saved split file (`data/split.csv`). DMD drivers stay in the train/val split in `docs/spec.md` section 2.6; do not treat all DMD clips as a separate evaluation set. Compute normalization values from train only.  
- Write the evaluation criteria in `README.md` before looking at results. Use the final judgment data (Test2) only once, and never use it to choose models or thresholds.

## Run order

`auto_label.extract` → `auto_label.head_review.server` (human review) → `dataset.convert_labels`, `dataset.dmd_convert` → `sequence_model.build_windows` → `sequence_model.train`, `sequence_model.train_yawn` → `eval.evaluate`, `eval.yawn_compare` → `eval.report` → `demo/`

- Run commands from the repo root as `python -m <folder>.<file>`. Tests: `python -m unittest discover -t . -s tests`  
- Priority: ① rule-based MVP → ② 1D-CNN/LSTM comparison → ③ (optional) ONNX. Do not build sequence models before the MVP.

## Do not

- Do not upload videos, features, labels, weights, or `models/face_landmarker.task` to the repo (share via the private Google Drive).  
- Do not publish material processed from DMD (cut videos, converted labels, overlay videos, weights trained on DMD). Make public demo videos from NITYMED and our own footage.  
- Do not add libraries that are not in the README library list without sharing with the team first.

## Records

- When you change landmark indices, thresholds, window, judgment interval, or label conversion rules, record the reason and result in the **experiment log** table in `README.md`.  
- When retraining, use a new name for `--tag`.

## Environment

- conda environment `geumjjogi-drowsy`. Train models on GPU, extract features on CPU.  
- Landmark extraction is slow, so run it in parts and save the result tables (`data/features/*.parquet`) for reuse.
# Drowsy Driving Detection: Project Proposal

Team 금쪽같은 우리 사이 (기획서): Jang Beomsu (lead), Kim Minyoung, Yuk Juyoung | Version 2026-11-04 | Presentation 2026-11-09 (Mon)

## 1 Overview

From webcam video, the system analyzes eye closure, yawning, and head-down posture in real time to detect drowsiness and warn the driver. We build a rule-based MVP first and compare it with 1D-CNN and LSTM. Technical details are in the separate technical specification (기술서).

| Item | Content |
|---|---|
| Data | NITYMED (night, real car, 21 drivers), DMD drowsiness part (day, 13 drivers' worth), self-recorded (indoor acting) |
| Goals | Drowsy (3 s) event recall of 90% or higher; eye-closure F1 better than the rule-based baseline; webcam 15 FPS or higher |
| Deliverables | Drowsiness stage warnings, yawn alerts, rule vs. learned comparison table, label data, Streamlit demo, post-drive summary |
| Technology | MediaPipe Face Landmarker, OpenCV, PyTorch (1D-CNN, LSTM), Streamlit |
| Out of scope | Regulatory certification, in-vehicle validation, drowsiness prediction, Eye Gaze, IR cameras, PERCLOS / drowsiness score |
| Data terms | DMD is CC BY-NC-ND, so processed data and labels are not published. The public demo is built from NITYMED and our own footage. |

## 2 Problems

| Problem | Detail |
|---|---|
| Tighter regulation | The EU GSR made drowsiness and attention warning (DDAW) mandatory for all new cars (2024.07), and Euro NCAP 2026 requires direct monitoring of eyes and head. |
| Semi-autonomous driving | When a semi-autonomous vehicle hands control back to the driver, it must confirm the driver is awake. |
| Hard to notice | Drivers have trouble noticing their own drowsiness, and indirect signals such as steering cannot directly see eye closure or yawning. |

## 3 Solution

| Step | What it does | Key criterion |
|---|---|---|
| Webcam video | Receives the driver's face frame by frame. | No extra hardware |
| Face landmarks | MediaPipe outputs 478 points per frame. | Pretrained model, no training of our own |
| Landmark selection | Uses only eye, mouth, and nose-tip points. | Only the points needed |
| Feature computation | Computes features such as EAR, MAR, and head angles. | Every frame |
| Judgment | Sets the drowsiness stage from continuous eye-closure time. | Rule-based MVP, then compare 1D-CNN and LSTM |
| Warning | Shows on-screen indicators and sounds an alarm by stage. | Drowsiness stages below |

| Drowsiness stage | Continuous eye closure | Action |
|---|---|---|
| Normal | Under 1.5 s (including blinks) | No indicator |
| Caution (microsleep) | 1.5 s or more, under 3 s | On-screen caution |
| Drowsy (sleep) | 3 s or more | Alarm sound, on-screen warning |

Euro NCAP SD-202 (reference 5) defines microsleep as 1–2 s and sleep as 3 s. The team chose 1.5 s within that range.

## 4 User scenario

| Step | Stage | Content |
|---|---|---|
| 1 | Turn on webcam | Turn on the webcam in the Streamlit demo. |
| 2 | Real-time judgment | Analyzes face features and shows the continuous eye-closure time. |
| 3 | Stage warning | Closing the eyes for 1.5 s shows a caution; 3 s gives an alarm and an on-screen warning. |
| 4 | Separate alert | Repeated yawning shows a caution; head-down is only indicated. |
| 5 | End of drive | Ending the demo gathers the judgment log into a summary. |
| 6 | Review summary | Caution, drowsy, and yawn events are collected in time order. |

| User | What they get |
|---|---|
| Driver | Stage warnings and continuous eye-closure time, repeated-yawn alerts, head-down indication |
| Transport / logistics manager | Drowsiness event records and a post-drive summary |
| Project team | Rule vs. learned comparison, tables by lighting and glasses condition |

## 5 System design

### Overall structure: development 1–4, operation 5–8

| Order | Stage | Content | Kind |
|---|---|---|---|
| 1 | Data and labels | Auto labels, then human review | External input / data |
| 2 | Feature extraction | Per-frame features | |
| 3 | Training | Time window, 1D-CNN / LSTM | |
| 4 | Evaluation | F1, warning delay, FPS | |
| 5 | Deployment | Streamlit demo | |
| 6 | Real-time inference | Time window, periodic judgment | |
| 7 | Monitoring | Judgment log, FPS, delay | |
| 8 | Improvement | Re-label false alarms and misses | Final result of each flow |

The dotted line is the path that feeds improvement results back into the data (8 Improvement → 1 Data and labels).

### 5-1. Real-time judgment

1. Per-frame features (external input / data) come in.
2. Check "can the eyes be seen?"
    - No: **cannot judge**. If the eyes were closed up to just before, counting continues even if the head drops.
    - Yes: pass to the **continuous eye-closure timer** (shared by rules and model).
3. The timer value sets the drowsiness stage.

| Stage | Continuous eye closure | Action |
|---|---|---|
| Normal | Under 1.5 s | No indicator |
| Caution | 1.5–3 s | On-screen indicator |
| Drowsy | 3 s or more | Alarm sound / on-screen |

The model only answers whether the eyes are closed right now; time is measured by the same timer.

### 5-2. Labeling and training data

| Order | Stage | Content | Kind |
|---|---|---|---|
| 1 | Source video and labels | NITYMED, DMD | External input / data |
| 2 | Label conversion | Convert DMD label format | |
| 3 | Feature extraction | Per-frame features | |
| 4 | Threshold calibration | Threshold search | |
| 5 | Auto labeling | Generate draft labels | |
| 6 | Human review | Review eye-closure and yawn spans | Human step |
| 7 | Training window generation | Time windows, 4-class ground truth | Final result of each flow |

Videos, features, and labels are not uploaded to git; they are shared through the private team Google Drive.

Legend: external input / data; human step; final result of each flow.

### 5-3 Design criteria

| Criterion | Choice | Rationale |
|---|---|---|
| Training window | Time window in seconds, common FPS | Data FPS differs (25 and 29.76), so we cut by seconds and then resample. |
| Stride | Separate settings for training and real time | Adjacent windows overlap heavily, so training uses a wide stride and real time matches the response speed. |
| Model output | 4-class softmax | Normal, eye closure, yawn, head-down. Uncertain is an exclusion mask, not a class. |
| Answer rule | Based on the current state | Eye closure and head-down are labeled by the current state; yawn by the event that continues. |
| Data split | By driver | The same person never appears in both training and evaluation; normalization values are computed from training data only. |
| Head-down | Separate from drowsiness stage | The protocol has no criterion for it. Head-down with open eyes is only indicated; head-down during eye closure keeps counting; prolonged "cannot judge" gives a check-needed warning. |
| Yawn | Caution when repeated | People yawn without being sleepy, so one yawn does not change the stage (stage unchanged). A cooldown follows each alert. |

Numbers, formulas, and exception handling for each item are in the technical specification.

## 6 GitHub

Repository: https://github.com/Jbs1214/geumjjogi-drowsy

| Folder | Role |
|---|---|
| `dataset/` | Pair videos and labels, convert labels, split |
| `auto_label/` | Feature extraction, thresholds, auto-labeling tools |
| `landmarks/` | Landmark indices, feature computation |
| `rule_based/` | Eye-closure timer, yawn and head-down handling, warnings |
| `sequence_model/` | 1D-CNN / LSTM training and evaluation |
| `eval/` | Evaluation scripts, per-condition result tables |
| `demo/` | Streamlit real-time demo |

| Rule | Content |
|---|---|
| Branch | No direct push to `main`; merge through PRs. `feat/`, `fix/`, `docs/` |
| Commit | `YYMMDD type : description` |
| Excluded | Source videos showing faces and dataset files are not uploaded. |

### Technology

| Category | Choice | Used for |
|---|---|---|
| Landmarks | MediaPipe Face Landmarker | Extract face points every frame (pretrained) |
| Video / data | OpenCV, pandas, NumPy | Read video, compute features, create labels and windows |
| Learning models | PyTorch (1D-CNN, LSTM), scikit-learn | Models to compare with rule-based judgment; baseline |
| Demo | Streamlit, streamlit-webrtc | Live webcam demonstration (local) |
| Collaboration | GitHub, Notion, Google Drive, Discord | Code, documents, data sharing, communication |

## 7 Other

### 7-1 Roles

| Name | Responsibility | Main tasks |
|---|---|---|
| Jang Beomsu (lead) | Data collection, feature extraction, labeling, training data generation | Obtain NITYMED and DMD, extract features, convert labels, auto-label and review, generate training windows |
| Kim Minyoung (member) | LSTM, eye-closure judgment, evaluation | LSTM training, eye-closure timer and thresholds, rule vs. learned comparison table, FPS and warning delay |
| Yuk Juyoung (member) | 1D-CNN, yawn and head-down judgment, demo | 1D-CNN training, yawn alerts and head-down handling, Streamlit demo, log, and summary |

### 7-2 Schedule

| Period | Content |
|---|---|
| 10/26 – 11/02 | Finalize data, auto-labeling tool, acted recordings |
| 11/03 – 11/05 | Label conversion, auto labeling and review, rule-based MVP and demo |
| 11/06 – 11/08 | Model training and evaluation, demo video, summary screen |
| 11/09 | Final presentation |

### 7-3 Evaluation plan

| Metric | Target |
|---|---|
| Drowsy (3 s) event recall | 90% or higher |
| Eye-closure span F1 | Better than the rule-based baseline |
| Processing speed (measured on webcam) | 15 FPS or higher |
| Other metrics | Reported as values: caution recall, yawn, head-down, false alarm rate, warning delay |

### 7-4 Limitations and responses

| Limitation | Response |
|---|---|
| Generalization | With only 34 drivers, results are reported separately by lighting. |
| Few 3 s closures | Label the 21 microsleep clips first and supplement with acted footage. |
| Sunglasses / masks | If the eye signal barely changes, show "cannot judge"; with a mask, turn off yawn alerts. |
| Not validated at highway speeds | State that this is a research demo and name the limitation in the presentation. |
| Recording safety | Never record while driving drowsy; use indoor acting and consent forms. |

### 7-5 References

| No. | Reference | Used for |
|---|---|---|
| 1 | 공도현·곽근창 (2016), 눈 개폐 빈도수 기반 졸음 판단 (Drowsiness judgment based on eye open/close frequency) | Eye-closure-based drowsiness judgment |
| 2 | Soukupová & Čech (2016), Real-Time Eye Blink Detection using Facial Landmarks | EAR (eye openness) formula |
| 3 | Bai et al. (2018), Empirical Evaluation of Generic Convolutional and Recurrent Networks | Comparing 1D-CNN and LSTM together |
| 4 | Ghoddoosian et al. (2019), A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection | Comparison with prior-work performance |
| 5 | Euro NCAP, Driver Monitoring Test Procedure SD-202 v1.2 (2026.07) | Drowsiness stage criteria (1.5 s, 3 s) |
| 6 | Petrellis et al. (2021), NITYMED / Ortega et al. (2020), DMD | Sources of the NITYMED and DMD data |

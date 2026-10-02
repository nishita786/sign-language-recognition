# ASL Alphabet Recognition

Real-time recognition of ten American Sign Language letters from the webcam. The site uses a small slice of the [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) photos: **A, B, C, D, F, I, L, V, W, Y**. Those letters have distinct hand shapes, so a landmark model stays accurate.

Each photo is turned into 21 MediaPipe hand landmarks, centered on the wrist and scaled by the palm. On the held-out photos the model reaches **99.4% accuracy**.

## Run the website

From the project root, with the existing environment:

```bash
source .live_env/bin/activate
python src/app.py
```

Open http://127.0.0.1:7860 and allow the camera. Hold one of the ten letters still until it appears, then lower your hand. That letter is added to the word on screen and printed in the terminal. **Clear word** starts over.

## Results

| Letter | Held-out photos |
| --- | ---: |
| A | 29/29 |
| B | 30/30 |
| C | 30/30 |
| D | 35/35 |
| F | 39/39 |
| I | 32/33 |
| L | 34/34 |
| V | 35/35 |
| W | 33/34 |
| Y | 34/34 |
| **Overall** | **99.4%** |

The full report is `results/asl_letters_report.txt`.

## Data

200 photos per letter, copied from the Kaggle training set into `dataset/asl_letters/`. Hands MediaPipe could not see were dropped before training. Photos, features, and weights live under `dataset/` and `models/` and are not committed.

## Reproduce training

The ten-letter subset is already in `dataset/asl_letters/`. To rebuild it from the full Kaggle download, place that download under `dataset/kaggle` and run:

```bash
export PYTHONPATH=src
python src/train_asl_letters.py
```

## Earlier Indian Sign Language experiment

A 49-sign video model is still in the repo. Its best test accuracy was 80.0% on wrist-normalized landmarks. The webcam demo no longer uses that model, because similar signs such as boy, friend, and sorry were not reliable live. The old OpenCV script is `python src/live_recognition.py`.

## Reproduce training

Run these from the project root after `pip install -r requirements.txt`. Scripts that import each other need `src` on the path:

```bash
export PYTHONPATH=src

python src/create_split.py
python src/extract_hand_landmarks.py
python src/normalize_landmarks.py
python src/train_normalized_landmark_lstm.py
python src/evaluate_normalized_landmark_lstm.py
```

The motion experiment, kept for comparison:

```bash
python src/create_motion_features.py
python src/train_motion_lstm.py
python src/evaluate_motion_lstm.py
```

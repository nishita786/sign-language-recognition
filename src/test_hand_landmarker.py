import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/mediapipe/hand_landmarker.task"

VIDEO_PATH = (
    "dataset/raw/sign project/"
    "1.bird/"
    "Bird2.mp4"
)

# Check model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Check video
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "We will locate an actual video filename if needed."
    )

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_count = 0
frames_with_hands = 0
max_hands = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # OpenCV BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Detect hands
    result = detector.detect(mp_image)

    num_hands = len(result.hand_landmarks)

    if num_hands > 0:
        frames_with_hands += 1

    max_hands = max(max_hands, num_hands)

    if frame_count <= 5:
        print(
            f"Frame {frame_count}: "
            f"{num_hands} hand(s) detected"
        )

cap.release()
detector.close()

print("\n========== RESULT ==========")
print("Video:", VIDEO_PATH)
print("Total frames:", frame_count)
print("Frames with hands:", frames_with_hands)
print("Maximum hands detected:", max_hands)

if frame_count > 0:
    percentage = (frames_with_hands / frame_count) * 100
    print(f"Detection rate: {percentage:.2f}%")

print("============================")

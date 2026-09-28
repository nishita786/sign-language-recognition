import os
import cv2
import numpy as np
import mediapipe as mp

VIDEO_PATH = "dataset/raw/sign project/16.eat/0403(38).mp4"

OUTPUT_PATH = "dataset/landmarks/val/16.eat__0403(38).npy"

SEQUENCE_LENGTH = 32
FEATURES_PER_FRAME = 126

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video frames:", total_frames)

# Read ALL frames first
all_frames = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    all_frames.append(frame)

cap.release()

print("Frames successfully read:", len(all_frames))

if len(all_frames) == 0:
    raise RuntimeError("No frames could be read.")

# Uniformly select exactly 32 frames.
# If the video is shorter than 32 frames,
# np.linspace repeats/duplicates indices.
indices = np.linspace(
    0,
    len(all_frames) - 1,
    SEQUENCE_LENGTH
).astype(int)

features = []

for i, frame_index in enumerate(indices):

    frame = all_frames[frame_index]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    frame_features = np.zeros(
        FEATURES_PER_FRAME,
        dtype=np.float32
    )

    if results.multi_hand_landmarks:

        for hand_idx, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            if hand_idx >= 2:
                break

            start = hand_idx * 63

            for landmark_idx, landmark in enumerate(
                hand_landmarks.landmark
            ):

                frame_features[
                    start + landmark_idx * 3
                ] = landmark.x

                frame_features[
                    start + landmark_idx * 3 + 1
                ] = landmark.y

                frame_features[
                    start + landmark_idx * 3 + 2
                ] = landmark.z

    features.append(frame_features)

hands.close()

sequence = np.asarray(
    features,
    dtype=np.float32
)

print("Final shape:", sequence.shape)
print("Non-zero values:", np.count_nonzero(sequence))

if sequence.shape != (32, 126):
    raise RuntimeError(
        f"Wrong shape: {sequence.shape}"
    )

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

np.save(
    OUTPUT_PATH,
    sequence
)

print()
print("SUCCESS")
print("Recovered:", OUTPUT_PATH)
print("Shape:", sequence.shape)

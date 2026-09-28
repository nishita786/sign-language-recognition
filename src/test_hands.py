import cv2
import mediapipe as mp

VIDEO_PATH = "dataset/raw/sign project/1.bird/Bird2.mp4"

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

frame_count = 0
frames_with_hands = 0
max_hands = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if num_hands > 0:
        frames_with_hands += 1

    max_hands = max(max_hands, num_hands)

    if frame_count <= 5:
        print(f"Frame {frame_count}: {num_hands} hand(s) detected")

cap.release()
hands.close()

print("\n========== RESULT ==========")
print("Video:", VIDEO_PATH)
print("Total frames:", frame_count)
print("Frames with hands:", frames_with_hands)
print("Maximum hands detected:", max_hands)

if frame_count:
    detection_rate = frames_with_hands / frame_count * 100
    print(f"Detection rate: {detection_rate:.2f}%")

print("============================")

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque


# =========================
# Configuration
# =========================

MODEL_PATH = "models/normalized_landmark_lstm_best.keras"

SEQUENCE_LENGTH = 32
FEATURES = 126

CONFIDENCE_THRESHOLD = 0.60

# Your 49 class names
CLASS_NAMES = [
    "1.bird",
    "10.youareperfect",
    "11.monsoon",
    "12.afternoon",
    "13.angry",
    "14.bad",
    "15.boy",
    "16.eat",
    "17.friend",
    "18.drinking",
    "19.girl",
    "2.black",
    "20.brother",
    "21.good",
    "22.father",
    "23.evening",
    "24.help",
    "25.mother",
    "26.NAME",
    "27.Night",
    "28.Music",
    "29.nose",
    "3.cat",
    "30.sleep",
    "31.sit",
    "32.sorry",
    "33.stand",
    "34.stop",
    "35.student",
    "36.study",
    "37.teacher",
    "38.thankyou",
    "39.today",
    "4.cow",
    "40.tommorow",
    "41.welcome",
    "42.work",
    "43.yesterday",
    "44.teeth",
    "45.hand",
    "46.write",
    "47.umberlla",
    "48.ring",
    "5.dog",
    "50.power",
    "6.fish",
    "7.goodmorning",
    "8.grey",
    "9.hello"
]


# =========================
# Load model
# =========================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================
# MediaPipe
# =========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


# =========================
# Frame buffer
# =========================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=5
)

print("\nLive recognition started.")
print("Perform a sign after the buffer fills.")
print("Press Q to quit.\n")


# =========================
# Main loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)


    # =========================
    # Extract landmarks
    # =========================

    frame_features = np.zeros(
        FEATURES,
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


            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # =========================
    # Normalize landmarks
    # Same preprocessing used
    # during training
    # =========================

    normalized = frame_features.copy()

    for hand_idx in range(2):

        start = hand_idx * 63
        end = start + 63

        hand = frame_features[
            start:end
        ].reshape(21, 3)

        wrist = hand[0].copy()

        # If no hand was detected,
        # leave zeros unchanged.
        if np.any(hand):

            hand = hand - wrist

        normalized[
            start:end
        ] = hand.reshape(63)


    # Add frame to sequence
    sequence.append(normalized)


    # =========================
    # Prediction
    # =========================

    predicted_label = "Collecting frames..."
    confidence = 0.0

    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(
            sequence,
            dtype=np.float32
        )

        input_data = np.expand_dims(
            input_data,
            axis=0
        )

        probabilities = model.predict(
            input_data,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            probabilities
        )

        confidence = float(
            probabilities[predicted_index]
        )

        current_prediction = (
            predicted_index
        )

        prediction_history.append(
            current_prediction
        )

        # Majority vote across recent predictions
        if len(prediction_history) >= 3:

            counts = np.bincount(
                prediction_history,
                minlength=len(CLASS_NAMES)
            )

            stable_prediction = np.argmax(
                counts
            )

        else:

            stable_prediction = (
                current_prediction
            )


        if confidence >= CONFIDENCE_THRESHOLD:

            predicted_label = CLASS_NAMES[
                stable_prediction
            ]

        else:

            predicted_label = "Uncertain"


    # =========================
    # Display
    # =========================

    cv2.rectangle(
        frame,
        (10, 10),
        (700, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Prediction: {predicted_label}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )


    cv2.imshow(
        "Real-Time ISL Recognition",
        frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
hands.close()

cv2.destroyAllWindows()

print("Recognition stopped.")

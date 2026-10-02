"""Real-time Indian Sign Language recognition.

The demo uses the normalized landmark LSTM, the strongest model in this
project (80% test accuracy). Each sign is captured the same way the
model was trained: 32 frames, both hands, wrist-centered landmarks.

Hold a sign in front of the camera. The word is shown on the video
and printed in the terminal. Press Q and that word is printed again.

Controls
    Q  quit
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from sign_features import hands_to_features, sample_sequence


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "normalized_landmark_lstm_best.keras"
SPECIALIST_PATH = ROOT / "models" / "boy_friend_sorry.keras"
CLASS_DIR = ROOT / "dataset" / "landmarks_normalized" / "train"

SEQUENCE_LENGTH = 32
PREDICT_EVERY = 4
MIN_FRAMES = 16

DISPLAY_FIXES = {
    "youareperfect": "you are perfect",
    "goodmorning": "good morning",
    "thankyou": "thank you",
    "tommorow": "tomorrow",
    "umberlla": "umbrella",
}

FALLBACK_CLASS_NAMES = [
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
    "9.hello",
]


def load_class_names():
    """Class order must match training: sorted label prefixes."""
    if CLASS_DIR.is_dir():
        names = sorted({
            path.name.split("__")[0]
            for path in CLASS_DIR.glob("*.npy")
        })
        if len(names) == 49:
            return names
    return list(FALLBACK_CLASS_NAMES)


def display_name(class_name):
    raw = class_name.split(".", 1)[-1]
    raw = DISPLAY_FIXES.get(raw, raw)
    return raw.replace("_", " ").lower()


def landmarks_to_features(hand_landmarks_list):
    """Pack hands into the 132-d frame used by the position model."""
    return hands_to_features(hand_landmarks_list)


def wrist_normalize(frame_features):
    """Subtract each hand's wrist, matching normalize_landmarks.py."""
    normalized = frame_features.copy()

    for hand_idx in range(2):
        start = hand_idx * 63
        end = start + 63
        hand = frame_features[start:end].reshape(21, 3)

        if np.any(hand):
            hand = hand - hand[0]

        normalized[start:end] = hand.reshape(63)

    return normalized.astype(np.float32)


def hand_shape(sequence):
    """Hand size and how often both hands are visible.

    Sorry in this dataset is one closed fist. Friend is both hands
    clasped, so the shape is wider or a second hand is detected.
    """
    sizes = []
    two_hands = 0
    used = 0

    for frame in sequence:
        present = []
        for hand_index in range(2):
            start = hand_index * 63
            hand = frame[start:start + 63].reshape(21, 3)
            if np.any(hand):
                present.append(hand)
        if not present:
            continue
        used += 1
        if len(present) == 2:
            two_hands += 1
        points = present[0][:, :2]
        sizes.append(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    if not sizes:
        return 1.0, 0.0
    return float(np.mean(sizes)), two_hands / used


def disambiguate_sorry_friend(sequence, probabilities, class_names, predicted_index):
    """Prefer sorry when a single small fist is only weakly called friend."""
    names = list(class_names)
    if "32.sorry" not in names or "17.friend" not in names:
        return predicted_index

    sorry_index = names.index("32.sorry")
    friend_index = names.index("17.friend")
    if predicted_index != friend_index:
        return predicted_index

    size, two_hand_fraction = hand_shape(sequence)
    friend_probability = float(probabilities[friend_index])
    sorry_probability = float(probabilities[sorry_index])
    one_fist = two_hand_fraction < 0.12 and size <= 0.16
    uncertain = (
        friend_probability < 0.78
        or friend_probability - sorry_probability < 0.25
    )
    if one_fist and uncertain:
        return sorry_index
    return predicted_index


_SPECIALIST = None
TRIO = ("15.boy", "17.friend", "32.sorry")
TRIO_WORDS = ("boy", "friend", "sorry")


def get_specialist():
    global _SPECIALIST
    if _SPECIALIST is None and SPECIALIST_PATH.exists():
        _SPECIALIST = tf.keras.models.load_model(SPECIALIST_PATH)
    return _SPECIALIST


def predict_sequence(model, class_names, frames):
    """Return the top label and confidence for a 32-frame window.

    Boy, friend, and sorry are passed to a smaller model that also sees
    where the hands are. That is what separates a hand at the chin, a
    fist on the chest, and two clasped hands.
    """
    sequence = sample_sequence(frames)
    shape = sequence[:, :126]
    probabilities = model.predict(
        shape[None, ...],
        verbose=0
    )[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    name = display_name(class_names[predicted_index])

    specialist = get_specialist()
    if (
        specialist is not None
        and class_names[predicted_index] in TRIO
        and sequence.shape[-1] >= 132
    ):
        trio_probabilities = specialist.predict(
            sequence[None, ...],
            verbose=0
        )[0]
        trio_index = int(np.argmax(trio_probabilities))
        name = TRIO_WORDS[trio_index]
        confidence = float(trio_probabilities[trio_index])

    return name, confidence


def draw_overlay(frame, word):
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.putText(
        frame,
        word,
        (24, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )


def run_check(model, class_names):
    """Confirm preprocessing matches the saved training features."""
    sample_name = "1.bird__0403(78).npy"
    raw_path = ROOT / "dataset" / "landmarks" / "test" / sample_name
    norm_path = ROOT / "dataset" / "landmarks_normalized" / "test" / sample_name

    if not raw_path.exists() or not norm_path.exists():
        print("Sample landmarks not found; model load check only.")
        print("Classes:", len(class_names))
        print("Check passed.")
        return

    raw = np.load(raw_path)
    expected = np.load(norm_path)
    rebuilt = np.stack([wrist_normalize(frame) for frame in raw])
    delta = float(np.max(np.abs(rebuilt - expected)))

    probabilities = model.predict(expected[None, ...], verbose=0)[0]
    predicted = class_names[int(np.argmax(probabilities))]
    confidence = float(np.max(probabilities))

    print(f"Normalization max abs error: {delta:.8f}")
    print(f"Sample prediction: {display_name(predicted)} ({confidence * 100:.1f}%)")

    if delta > 1e-5:
        raise RuntimeError(
            "Live normalization does not match the training features."
        )

    if model.output_shape[-1] != len(class_names):
        raise RuntimeError("Model classes do not match the label list.")

    print("Check passed.")


def open_webcam(index=0, attempts=30):
    """Open the camera, waiting if macOS is still asking for permission.

    OpenCV requests access and then fails immediately while the prompt is
    unanswered. Retrying gives the Allow button time to take effect.
    """
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    for attempt in range(attempts):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()

        if attempt == 0:
            print(
                "\nmacOS has not allowed camera access yet.\n"
                "Click Allow if a prompt is showing.\n"
                "If there is no prompt: System Settings → Privacy & Security "
                "→ Camera, and turn on Cursor.\n"
                "Waiting..."
            )
        time.sleep(1)

    return None


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            f"Model has {model.output_shape[-1]} classes, "
            f"but {len(class_names)} labels were loaded."
        )

    print("Model loaded.")
    print(f"Classes: {len(class_names)}")

    if "--check" in sys.argv:
        run_check(model, class_names)
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = open_webcam(0)
    if cap is None:
        hands.close()
        raise RuntimeError(
            "Could not open the webcam. On this Mac, allow Camera access "
            "for Cursor in System Settings → Privacy & Security → Camera, "
            "then run this script again. If Cursor was already listed, "
            "turn it off and on, quit Cursor, and reopen it."
        )

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    word = "..."
    frame_index = 0
    empty_frames = 0
    last_printed = None

    print("\nLive recognition started.", flush=True)
    print("Hold a sign, then press Q in the video window.\n", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame_index += 1

            detected = results.multi_hand_landmarks or []
            for hand_landmarks in detected:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            if detected:
                empty_frames = 0
                sequence.append(landmarks_to_features(detected))
            else:
                empty_frames += 1
                if empty_frames == 20:
                    sequence.clear()
                    word = "..."

            if (
                detected
                and len(sequence) >= MIN_FRAMES
                and frame_index % PREDICT_EVERY == 0
            ):
                word, _confidence = predict_sequence(
                    model,
                    class_names,
                    sequence
                )
                if word != last_printed:
                    print(word, flush=True)
                    last_printed = word

            draw_overlay(frame, word)

            cv2.imshow("Indian Sign Language Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        if last_printed:
            print(last_printed)


if __name__ == "__main__":
    main()

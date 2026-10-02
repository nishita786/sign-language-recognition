"""Train a small ASL letter model from the Kaggle ASL Alphabet photos.

Keeps a few visually distinct letters and a few hundred photos each, then
learns the hand shape from MediaPipe landmarks. Landmarks transfer to a
webcam better than a network trained on the studio backgrounds.
"""

import json
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from asl_letters import hand_features

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_TRAIN = (
    ROOT
    / "dataset"
    / "kaggle"
    / "datasets"
    / "grassknoted"
    / "asl-alphabet"
    / "versions"
    / "1"
    / "asl_alphabet_train"
    / "asl_alphabet_train"
)
SMALL_DIR = ROOT / "dataset" / "asl_letters"
MODEL_PATH = ROOT / "models" / "asl_letters.keras"
CLASS_PATH = ROOT / "models" / "asl_letters_classes.json"
REPORT_PATH = ROOT / "results" / "asl_letters_report.txt"

LETTERS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
PER_CLASS = 200
SEED = 7


def mirror_features(features):
    points = features.reshape(21, 3).copy()
    points[:, 0] *= -1.0
    return points.reshape(-1)


def copy_subset():
    already_copied = all(
        (SMALL_DIR / letter).is_dir() and any((SMALL_DIR / letter).glob("*.jpg"))
        for letter in LETTERS
    )
    if already_copied and not KAGGLE_TRAIN.exists():
        print("using the small set already in", SMALL_DIR)
        return
    if not KAGGLE_TRAIN.exists():
        raise SystemExit(f"Kaggle train folder not found: {KAGGLE_TRAIN}")
    rng = np.random.RandomState(SEED)
    SMALL_DIR.mkdir(parents=True, exist_ok=True)
    for letter in LETTERS:
        source = KAGGLE_TRAIN / letter
        target = SMALL_DIR / letter
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        names = sorted(path.name for path in source.glob("*.jpg"))
        chosen = rng.choice(names, size=PER_CLASS, replace=False)
        for name in chosen:
            shutil.copy2(source / name, target / name)
        print(f"copied {letter}: {PER_CLASS}")


def extract_features():
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    rows = []
    labels = []
    missed = {letter: 0 for letter in LETTERS}
    for class_index, letter in enumerate(LETTERS):
        for path in sorted((SMALL_DIR / letter).glob("*.jpg")):
            image = cv2.imread(str(path))
            if image is None:
                missed[letter] += 1
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detected = hands.process(rgb).multi_hand_landmarks
            if not detected:
                missed[letter] += 1
                continue
            rows.append(hand_features(detected[0]))
            labels.append(class_index)
        print(f"extracted {letter}: {labels.count(class_index)} kept, {missed[letter]} missed")
    hands.close()
    features = np.asarray(rows, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    np.savez_compressed(SMALL_DIR / "features.npz", features=features, labels=labels)
    return features, labels


def split_and_augment(features, labels):
    rng = np.random.RandomState(SEED)
    train_x, train_y, test_x, test_y = [], [], [], []
    for class_index in range(len(LETTERS)):
        indices = np.where(labels == class_index)[0]
        rng.shuffle(indices)
        cut = int(len(indices) * 0.8)
        train_idx, test_idx = indices[:cut], indices[cut:]
        train_x.append(features[train_idx])
        train_y.append(labels[train_idx])
        test_x.append(features[test_idx])
        test_y.append(labels[test_idx])
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    mirrored = np.stack([mirror_features(row) for row in train_x])
    train_x = np.concatenate([train_x, mirrored])
    train_y = np.concatenate([train_y, train_y])
    order = rng.permutation(len(train_x))
    return train_x[order], train_y[order], test_x, test_y


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(LETTERS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def write_report(test_x, test_y, probabilities):
    predicted = np.argmax(probabilities, axis=1)
    lines = [f"overall {float(np.mean(predicted == test_y)):.4f}", ""]
    for class_index, letter in enumerate(LETTERS):
        mask = test_y == class_index
        correct = int(np.sum(predicted[mask] == class_index))
        total = int(np.sum(mask))
        lines.append(f"{letter} {correct}/{total}")
    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)


def main():
    copy_subset()
    features, labels = extract_features()
    train_x, train_y, test_x, test_y = split_and_augment(features, labels)
    model = build_model()
    model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        epochs=40,
        batch_size=32,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                restore_best_weights=True,
            )
        ],
    )
    probabilities = model.predict(test_x, verbose=0)
    write_report(test_x, test_y, probabilities)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    CLASS_PATH.write_text(json.dumps(LETTERS))
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()

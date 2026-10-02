"""Train an LSTM that sees hand shape and where the hands are.

Run from the project root:

    python src/train_position_lstm.py
"""

import glob
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sign_features import (
    FEATURES,
    SEQUENCE_LENGTH,
    mirror_raw,
    to_features,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BATCH_SIZE = 16
EPOCHS = 40


def load_split(split, mirror=False):
    paths = sorted(glob.glob(os.path.join("dataset", "landmarks", split, "*.npy")))
    class_names = sorted({os.path.basename(path).split("__")[0] for path in paths})
    class_to_index = {name: index for index, name in enumerate(class_names)}

    sequences = []
    labels = []
    for path in paths:
        raw = np.load(path)
        label = class_to_index[os.path.basename(path).split("__")[0]]
        sequences.append(np.stack([to_features(frame) for frame in raw]))
        labels.append(label)
        if mirror:
            flipped = np.stack([to_features(mirror_raw(frame)) for frame in raw])
            sequences.append(flipped)
            labels.append(label)

    return (
        np.stack(sequences).astype(np.float32),
        np.array(labels, dtype=np.int32),
        class_names,
    )


def build_model(num_classes):
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, FEATURES))
    x = layers.LayerNormalization()(inputs)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading landmarks with hand position...")
    x_train, y_train, class_names = load_split("train", mirror=True)
    x_val, y_val, val_names = load_split("val", mirror=True)
    x_test, y_test, test_names = load_split("test", mirror=False)

    if class_names != val_names or class_names != test_names:
        raise RuntimeError("Class lists do not match across splits.")

    print(f"Train {len(y_train)}  Val {len(y_val)}  Test {len(y_test)}")
    print(f"Classes {len(class_names)}  Features {FEATURES}")

    model = build_model(len(class_names))
    checkpoint = "models/position_lstm_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    best = tf.keras.models.load_model(checkpoint)
    loss, accuracy = best.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    probabilities = best.predict(x_test, verbose=0)
    predicted = probabilities.argmax(axis=1)
    for label in ("15.boy", "17.friend", "32.sorry"):
        index = class_names.index(label)
        mask = y_test == index
        correct = np.sum(predicted[mask] == index)
        print(f"  {label}: {correct}/{mask.sum()}")

    with open("results/position_lstm_history.json", "w") as handle:
        json.dump(
            {key: [float(value) for value in values] for key, values in history.history.items()},
            handle,
            indent=2,
        )
    with open("models/position_lstm_classes.json", "w") as handle:
        json.dump(class_names, handle, indent=2)

    print("Saved", checkpoint)


if __name__ == "__main__":
    main()

import os
import json
import tensorflow as tf

from motion_landmark_loader import (
    MotionLandmarkDataGenerator
)

from model_motion_lstm import (
    build_motion_model
)


BATCH_SIZE = 16
EPOCHS = 50

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


print("\nLoading motion + landmark datasets...")

train_gen = MotionLandmarkDataGenerator(
    "train",
    batch_size=BATCH_SIZE
)

val_gen = MotionLandmarkDataGenerator(
    "val",
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_gen = MotionLandmarkDataGenerator(
    "test",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Train batches:", len(train_gen))
print("Validation batches:", len(val_gen))
print("Test batches:", len(test_gen))
print("Classes:", len(train_gen.class_names))


print("\nBuilding motion LSTM...")

model = build_motion_model()

model.summary()


checkpoint_path = (
    "models/motion_landmark_lstm_best.keras"
)


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


print("\nStarting training...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(
    test_gen,
    verbose=1
)


print("\n==============================")
print("MOTION + LANDMARK LSTM")
print("==============================")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("==============================")


final_path = (
    "models/motion_landmark_lstm_final.keras"
)

model.save(final_path)


history_path = (
    "results/motion_landmark_lstm_history.json"
)

history_data = {
    key: [float(v) for v in values]
    for key, values in history.history.items()
}

with open(history_path, "w") as f:
    json.dump(
        history_data,
        f,
        indent=2
    )


print("\nFinal model saved to:")
print(final_path)

print("\nTraining history saved to:")
print(history_path)

print("\nTraining complete.")

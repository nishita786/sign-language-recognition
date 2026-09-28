import os
import json
import tensorflow as tf

from landmark_data_loader import LandmarkDataGenerator
from model_landmark_lstm import build_model


# =========================
# Configuration
# =========================

BATCH_SIZE = 16
EPOCHS = 50

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# Load data
# =========================

print("\nLoading landmark datasets...")

train_gen = LandmarkDataGenerator(
    "train",
    batch_size=BATCH_SIZE
)

val_gen = LandmarkDataGenerator(
    "val",
    batch_size=BATCH_SIZE
)

test_gen = LandmarkDataGenerator(
    "test",
    batch_size=BATCH_SIZE
)

print(f"Train batches: {len(train_gen)}")
print(f"Validation batches: {len(val_gen)}")
print(f"Test batches: {len(test_gen)}")
print("Classes: 49")


# =========================
# Build model
# =========================

print("\nBuilding Landmark LSTM model...")

model = build_model()

model.summary()


# =========================
# Callbacks
# =========================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "landmark_lstm_best.keras"
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


# =========================
# Train
# =========================

print("\nStarting training...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# =========================
# Evaluate
# =========================

print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(
    test_gen,
    verbose=1
)

print("\n==============================")
print("LANDMARK LSTM RESULTS")
print("==============================")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("==============================")


# =========================
# Save final model
# =========================

final_model_path = os.path.join(
    MODEL_DIR,
    "landmark_lstm_final.keras"
)

model.save(final_model_path)

print(f"\nFinal model saved to:")
print(final_model_path)


# =========================
# Save training history
# =========================

history_path = os.path.join(
    RESULTS_DIR,
    "landmark_lstm_history.json"
)

history_data = {
    key: [float(x) for x in values]
    for key, values in history.history.items()
}

with open(history_path, "w") as f:
    json.dump(history_data, f, indent=2)

print(f"Training history saved to:")
print(history_path)

print("\nTraining complete.")

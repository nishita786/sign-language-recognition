import tensorflow as tf
import numpy as np

from data_loader import ISLDataGenerator
from model_cnn_lstm import build_cnn_lstm


# ============================================================
# SANITY CHECK
# ============================================================

print("=" * 60)
print("CNN + LSTM SANITY CHECK")
print("=" * 60)

# ------------------------------------------------------------
# Load the normal training generator
# ------------------------------------------------------------

generator = ISLDataGenerator(
    split="train",
    batch_size=4,
    shuffle=False
)

# ------------------------------------------------------------
# Take only the first 8 samples
# ------------------------------------------------------------

X = []
y = []

for batch_index in range(2):

    batch_X, batch_y = generator[batch_index]

    X.append(batch_X)
    y.append(batch_y)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

print("\nSanity-check data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nLabels:")
print(y)

# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------

model = build_cnn_lstm()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created.")

# ------------------------------------------------------------
# Train only on these 8 samples
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING ON 8 SAMPLES")
print("=" * 60)

history = model.fit(
    X,
    y,
    batch_size=2,
    epochs=20,
    shuffle=True,
    verbose=1
)

# ------------------------------------------------------------
# Final evaluation
# ------------------------------------------------------------

final_accuracy = history.history["accuracy"][-1]

best_accuracy = max(
    history.history["accuracy"]
)

print("\n" + "=" * 60)
print("SANITY CHECK RESULT")
print("=" * 60)

print(
    f"\nFinal training accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Best training accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

if best_accuracy >= 0.90:

    print("\n✓ SANITY CHECK PASSED")
    print(
        "The model can memorize a tiny dataset."
    )
    print(
        "The data/model pipeline is fundamentally working."
    )

else:

    print("\n✗ SANITY CHECK FAILED")
    print(
        "The model could not memorize the tiny dataset."
    )
    print(
        "We need to investigate the model or data pipeline."
    )
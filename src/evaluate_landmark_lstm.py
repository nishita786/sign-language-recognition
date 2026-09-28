import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from landmark_data_loader import LandmarkDataGenerator


MODEL_PATH = "models/landmark_lstm_best.keras"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading test dataset...")

test_gen = LandmarkDataGenerator(
    "test",
    batch_size=16
)

print("Test batches:", len(test_gen))

print("\nLoading best model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")


# =========================
# Generate predictions
# =========================

y_true = []
y_pred = []

for i in range(len(test_gen)):

    X_batch, y_batch = test_gen[i]

    predictions = model.predict(
        X_batch,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(y_batch)
    y_pred.extend(predicted_labels)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =========================
# Accuracy
# =========================

accuracy = np.mean(
    y_true == y_pred
)

print("\n==============================")
print("TEST ACCURACY")
print("==============================")
print(f"{accuracy * 100:.2f}%")
print("==============================")


# =========================
# Classification report
# =========================

class_names = test_gen.class_names

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(report)


with open(
    os.path.join(
        RESULTS_DIR,
        "landmark_classification_report.txt"
    ),
    "w"
) as f:
    f.write(report)


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(20, 18)
)

plt.imshow(cm)

plt.title(
    "Landmark LSTM Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.colorbar()

plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "landmark_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200
)

plt.close()

print("\nConfusion matrix saved to:")
print(cm_path)

print("\nEvaluation complete.")

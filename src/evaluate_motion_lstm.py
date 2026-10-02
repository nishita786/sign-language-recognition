import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_landmark_loader import MotionLandmarkDataGenerator


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MODEL_PATH = "models/motion_landmark_lstm_best.keras"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


print("\nLoading motion test dataset...")

test_gen = MotionLandmarkDataGenerator(
    "test",
    batch_size=16,
    shuffle=False
)

print("Test batches:", len(test_gen))
print("Test samples:", len(test_gen.files))
print("Classes:", len(test_gen.class_names))


print("\nLoading best motion model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


print("\nGenerating predictions...")

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

    y_true.extend(y_batch.tolist())
    y_pred.extend(predicted_labels.tolist())


y_true = np.array(y_true)
y_pred = np.array(y_pred)


accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n==============================")
print("MOTION + LANDMARK RESULTS")
print("==============================")
print(f"Accuracy:    {accuracy * 100:.2f}%")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("==============================")


report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(test_gen.class_names)),
    target_names=test_gen.class_names,
    zero_division=0
)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "motion_landmark_classification_report.txt"
)

with open(report_path, "w") as f:

    f.write("Motion + Landmark LSTM Evaluation\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write(report)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(test_gen.class_names))
)

plt.figure(figsize=(22, 20))
plt.imshow(cm)
plt.title("Motion + Landmark LSTM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(
    range(len(test_gen.class_names)),
    test_gen.class_names,
    rotation=90
)
plt.yticks(
    range(len(test_gen.class_names)),
    test_gen.class_names
)
plt.colorbar()
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "motion_landmark_confusion_matrix.png"
)

plt.savefig(cm_path, dpi=200)
plt.close()


print("\nClassification report saved to:")
print(report_path)
print("\nConfusion matrix saved to:")
print(cm_path)
print("\nEvaluation complete.")

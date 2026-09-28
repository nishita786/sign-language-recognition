import numpy as np
import tensorflow as tf
from collections import Counter

from normalized_landmark_loader import NormalizedLandmarkDataGenerator

MODEL_PATH = "models/normalized_landmark_lstm_best.keras"

test_gen = NormalizedLandmarkDataGenerator(
    "test",
    batch_size=16,
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for i in range(len(test_gen)):
    X, y = test_gen[i]

    predictions = model.predict(X, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    y_true.extend(y)
    y_pred.extend(pred_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

pairs = Counter()

for actual, predicted in zip(y_true, y_pred):
    if actual != predicted:
        actual_name = test_gen.class_names[actual]
        predicted_name = test_gen.class_names[predicted]
        pairs[(actual_name, predicted_name)] += 1

print("\n======================================")
print("TOP CONFUSIONS")
print("======================================")

if not pairs:
    print("No misclassifications!")
else:
    for (actual, predicted), count in pairs.most_common(15):
        print(f"{actual} -> {predicted} : {count}")

print("\n======================================")
print(f"Total test samples: {len(y_true)}")
print(f"Correct: {np.sum(y_true == y_pred)}")
print(f"Incorrect: {np.sum(y_true != y_pred)}")
print("======================================")

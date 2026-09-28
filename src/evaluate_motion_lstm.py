import numpy as np
import tensorflow as tf

from motion_landmark_loader import MotionLandmarkDataGenerator

MODEL_PATH = "models/motion_landmark_lstm_best.keras"

test_gen = MotionLandmarkDataGenerator(
    "test",
    batch_size=16,
    shuffle=False
)

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")

loss, accuracy = model.evaluate(
    test_gen,
    verbose=1
)

print("\n==============================")
print("MOTION + LANDMARK LSTM")
print("==============================")
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("==============================")

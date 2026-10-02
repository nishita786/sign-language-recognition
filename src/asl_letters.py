"""Hand-shape features for the small ASL letter model."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "asl_letters.keras"
CLASS_PATH = ROOT / "models" / "asl_letters_classes.json"


def load_letters():
    return json.loads(CLASS_PATH.read_text())


def hand_features(hand_landmarks):
    """63 values: wrist-centered landmarks, scaled by the palm size."""
    points = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
        dtype=np.float32,
    )
    points = points - points[0]
    scale = np.linalg.norm(points[9])
    if scale > 1e-6:
        points = points / scale
    return points.reshape(-1)

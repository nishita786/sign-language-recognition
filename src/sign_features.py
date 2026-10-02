"""Landmark features for the live model.

The 126 wrist-centered coordinates describe the hand shape. Six extra
values keep each hand's position in the frame, which is how boy (hand
near the face), sorry (one fist on the chest), and friend (both hands
clasped) stay apart.
"""

import numpy as np

RAW_FEATURES = 126
EXTRA_FEATURES = 6
FEATURES = RAW_FEATURES + EXTRA_FEATURES
SEQUENCE_LENGTH = 32


def mirror_raw(raw):
    """Flip a frame left-right, matching a mirrored webcam."""
    out = np.array(raw, dtype=np.float32, copy=True)
    for hand_index in range(2):
        start = hand_index * 63
        hand = out[start:start + 63].reshape(21, 3)
        if np.any(hand):
            hand[:, 0] = 1.0 - hand[:, 0]
            out[start:start + 63] = hand.reshape(63)
    return out


def wrist_normalize(raw):
    normalized = np.array(raw, dtype=np.float32, copy=True)
    for hand_index in range(2):
        start = hand_index * 63
        hand = normalized[start:start + 63].reshape(21, 3)
        if np.any(hand):
            hand = hand - hand[0]
            normalized[start:start + 63] = hand.reshape(63)
    return normalized


def to_features(raw):
    """Turn one raw 126-d landmark frame into the 132-d model input."""
    raw = np.asarray(raw, dtype=np.float32)
    extra = np.zeros(EXTRA_FEATURES, dtype=np.float32)
    for hand_index in range(2):
        start = hand_index * 63
        hand = raw[start:start + 63].reshape(21, 3)
        slot = hand_index * 3
        if np.any(hand):
            extra[slot] = 1.0
            extra[slot + 1] = hand[0, 0]
            extra[slot + 2] = hand[0, 1]
    return np.concatenate([wrist_normalize(raw), extra])


def hands_to_features(hand_landmarks_list):
    """Pack MediaPipe hands the same way the training videos were packed."""
    raw = np.zeros(RAW_FEATURES, dtype=np.float32)
    for hand_index, hand_landmarks in enumerate(hand_landmarks_list):
        if hand_index >= 2:
            break
        start = hand_index * 63
        for landmark_index, landmark in enumerate(hand_landmarks.landmark):
            raw[start + landmark_index * 3] = landmark.x
            raw[start + landmark_index * 3 + 1] = landmark.y
            raw[start + landmark_index * 3 + 2] = landmark.z
    return to_features(raw)


def sample_sequence(frames):
    frames = np.asarray(frames, dtype=np.float32)
    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH).astype(int)
    return frames[indices]

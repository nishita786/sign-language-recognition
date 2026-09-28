import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

SPLITS_DIR = "dataset/splits"
OUTPUT_ROOT = "dataset/landmarks"

SEQUENCE_LENGTH = 32
FEATURES_PER_FRAME = 126

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================
# MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return None

    # Select 32 frames uniformly
    indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    index_set = set(indices.tolist())

    features = []
    current_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if current_index in index_set:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            # ------------------------------------------------
            # 126 values:
            #
            # Left hand:
            #   21 landmarks × 3 = 63
            #
            # Right hand:
            #   21 landmarks × 3 = 63
            #
            # Total = 126
            # ------------------------------------------------

            frame_features = np.zeros(
                FEATURES_PER_FRAME,
                dtype=np.float32
            )

            if results.multi_hand_landmarks:

                for hand_idx, hand_landmarks in enumerate(
                    results.multi_hand_landmarks
                ):

                    if hand_idx >= 2:
                        break

                    start = hand_idx * 63

                    for landmark_idx, landmark in enumerate(
                        hand_landmarks.landmark
                    ):

                        frame_features[
                            start + landmark_idx * 3
                        ] = landmark.x

                        frame_features[
                            start + landmark_idx * 3 + 1
                        ] = landmark.y

                        frame_features[
                            start + landmark_idx * 3 + 2
                        ] = landmark.z

            features.append(frame_features)

        current_index += 1

        if len(features) == SEQUENCE_LENGTH:
            break

    cap.release()

    if len(features) != SEQUENCE_LENGTH:
        return None

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(split_name):

    csv_path = os.path.join(
        SPLITS_DIR,
        f"{split_name}.csv"
    )

    df = pd.read_csv(csv_path)

    output_dir = os.path.join(
        OUTPUT_ROOT,
        split_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    successful = 0
    failed = 0

    print(
        f"\nProcessing {split_name}: "
        f"{len(df)} videos"
    )

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name
    ):

        # CSV columns are:
        # video
        # label

        video_path = row["video"]
        label = row["label"]

        if not os.path.exists(video_path):

            print(
                f"\nWARNING: Video not found:"
                f"\n{video_path}"
            )

            failed += 1
            continue

        sequence = extract_video(
            video_path
        )

        if sequence is None:

            print(
                f"\nWARNING: Failed:"
                f"\n{video_path}"
            )

            failed += 1
            continue

        # ----------------------------------------------------
        # Create unique output filename
        # ----------------------------------------------------

        base_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        safe_label = str(label).replace(
            " ",
            "_"
        )

        output_filename = (
            f"{safe_label}__"
            f"{base_name}.npy"
        )

        output_path = os.path.join(
            output_dir,
            output_filename
        )

        np.save(
            output_path,
            sequence
        )

        successful += 1

    print(
        f"\n{split_name} complete"
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "ISL HAND LANDMARK EXTRACTION"
    )

    print(
        "======================================"
    )

    print(
        "Sequence length:",
        SEQUENCE_LENGTH
    )

    print(
        "Features per frame:",
        FEATURES_PER_FRAME
    )

    for split in [
        "train",
        "val",
        "test"
    ]:

        process_split(split)

    hands.close()

    print(
        "\n======================================"
    )

    print(
        "LANDMARK EXTRACTION COMPLETE"
    )

    print(
        "======================================"
    )

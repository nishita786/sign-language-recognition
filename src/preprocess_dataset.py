import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"

IMG_SIZE = 224
SEQUENCE_LENGTH = 32


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(video_path, sequence_length=32, img_size=224):
    """
    Extract exactly sequence_length frames from a video.

    Frames are uniformly sampled across the complete video.
    Videos with fewer frames are handled using repeated frames.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found: {video_path}")

    # Uniformly select frame positions
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length
    ).astype(int)

    frames = []

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        success, frame = cap.read()

        if not success:
            # If reading fails, use the previous frame
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros(
                    (img_size, img_size, 3),
                    dtype=np.uint8
                )

        else:
            # OpenCV reads BGR → convert to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

        # Resize
        frame = cv2.resize(
            frame,
            (img_size, img_size)
        )

        # Normalize 0–255 → 0–1
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

    cap.release()

    return np.array(frames, dtype=np.float32)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    csv_path = SPLIT_DIR / f"{split_name}.csv"

    output_path = OUTPUT_DIR / split_name

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print(f"PROCESSING {split_name.upper()} DATA")
    print("=" * 60)

    print(f"Videos: {len(df)}")

    successful = 0
    failed = 0

    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{split_name}"
    ):

        video_path = PROJECT_ROOT / row["video"]

        label = row["label"]

        try:

            frames = extract_frames(
                video_path,
                SEQUENCE_LENGTH,
                IMG_SIZE
            )

            # Create a safe filename
            video_name = Path(video_path).stem

            # Include index to prevent duplicate filenames
            output_file = (
                output_path /
                f"{index:04d}_{video_name}.npy"
            )

            np.save(
                output_file,
                frames
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFailed: {video_path}"
            )

            print(
                f"Error: {e}"
            )

    print("\n" + "-" * 60)
    print(f"{split_name.upper()} COMPLETE")
    print("-" * 60)

    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")

    return successful, failed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ISL DATASET PREPROCESSING")
    print("=" * 60)

    print(f"\nImage size       : {IMG_SIZE} x {IMG_SIZE}")
    print(f"Frames/video     : {SEQUENCE_LENGTH}")
    print("Normalization    : 0-1")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_successful = 0
    total_failed = 0

    # Process using the already-created splits
    for split in ["train", "val", "test"]:

        successful, failed = process_split(split)

        total_successful += successful
        total_failed += failed

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"\nTotal successful : {total_successful}")
    print(f"Total failed     : {total_failed}")

    print("\nProcessed data saved to:")
    print(OUTPUT_DIR)

    print("\nExpected structure:")

    print("""
dataset/
└── processed/
    ├── train/
    ├── val/
    └── test/
""")


if __name__ == "__main__":
    main()
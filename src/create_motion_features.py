import os
import glob
import numpy as np

INPUT_ROOT = "dataset/landmarks_normalized"
OUTPUT_ROOT = "dataset/landmarks_motion"

for split in ["train", "val", "test"]:

    input_dir = os.path.join(INPUT_ROOT, split)
    output_dir = os.path.join(OUTPUT_ROOT, split)

    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(
        os.path.join(input_dir, "*.npy")
    )

    print(f"\nProcessing {split}: {len(files)} files")

    successful = 0

    for file_path in files:

        landmarks = np.load(file_path)

        if landmarks.shape != (32, 126):
            print(
                "Skipping:",
                file_path,
                landmarks.shape
            )
            continue

        # Calculate frame-to-frame movement
        motion = np.zeros_like(landmarks)

        motion[1:] = (
            landmarks[1:] -
            landmarks[:-1]
        )

        # Combine:
        # normalized landmarks = 126
        # motion features       = 126
        # total                 = 252
        combined = np.concatenate(
            [landmarks, motion],
            axis=1
        )

        output_path = os.path.join(
            output_dir,
            os.path.basename(file_path)
        )

        np.save(
            output_path,
            combined.astype(np.float32)
        )

        successful += 1

    print(
        f"{split}: {successful}/{len(files)} processed"
    )

print("\nMotion feature creation complete.")

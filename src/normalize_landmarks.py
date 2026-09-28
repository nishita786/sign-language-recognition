import os
import glob
import numpy as np

INPUT_ROOT = "dataset/landmarks"
OUTPUT_ROOT = "dataset/landmarks_normalized"

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

        data = np.load(file_path)

        # Expected:
        # (32, 126)
        # 2 hands × 21 landmarks × 3 coordinates

        if data.shape != (32, 126):
            print(
                "Skipping wrong shape:",
                file_path,
                data.shape
            )
            continue

        normalized = data.copy()

        # Process both hands
        for hand_idx in range(2):

            start = hand_idx * 63
            end = start + 63

            hand = data[:, start:end].reshape(
                32, 21, 3
            )

            # Wrist = landmark 0
            wrist = hand[:, 0:1, :]

            # Make every landmark relative to wrist
            hand_normalized = hand - wrist

            normalized[
                :, start:end
            ] = hand_normalized.reshape(
                32, 63
            )

        output_path = os.path.join(
            output_dir,
            os.path.basename(file_path)
        )

        np.save(
            output_path,
            normalized.astype(np.float32)
        )

        successful += 1

    print(
        f"{split}: {successful}/{len(files)} processed"
    )

print("\nNormalization complete.")

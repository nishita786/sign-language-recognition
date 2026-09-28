import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

BATCH_SIZE = 4
NUM_CLASSES = 49

SEQUENCE_LENGTH = 32
IMG_SIZE = 224


# ============================================================
# LABEL MAPPING
# ============================================================

def create_label_mapping():

    train_df = pd.read_csv(
        SPLIT_DIR / "train.csv"
    )

    classes = sorted(
        train_df["label"].unique()
    )

    label_to_index = {
        label: index
        for index, label in enumerate(classes)
    }

    index_to_label = {
        index: label
        for label, index in label_to_index.items()
    }

    return label_to_index, index_to_label


# ============================================================
# MEMORY-EFFICIENT DATA GENERATOR
# ============================================================

class ISLDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        split,
        batch_size=4,
        shuffle=False
    ):

        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.df = pd.read_csv(
            SPLIT_DIR / f"{split}.csv"
        )

        self.label_to_index, self.index_to_label = (
            create_label_mapping()
        )

        self.indices = np.arange(
            len(self.df)
        )

        self.on_epoch_end()

        print(
            f"{split} dataset: "
            f"{len(self.df)} videos"
        )

    def __len__(self):

        return int(
            np.ceil(
                len(self.df) / self.batch_size
            )
        )

    def __getitem__(self, batch_index):

        start = batch_index * self.batch_size

        end = min(
            start + self.batch_size,
            len(self.df)
        )

        batch_indices = self.indices[
            start:end
        ]

        batch_size_actual = len(
            batch_indices
        )

        X = np.empty(
            (
                batch_size_actual,
                SEQUENCE_LENGTH,
                IMG_SIZE,
                IMG_SIZE,
                3
            ),
            dtype=np.float32
        )

        y = np.empty(
            batch_size_actual,
            dtype=np.int32
        )

        for i, original_index in enumerate(
            batch_indices
        ):

            row = self.df.iloc[
                original_index
            ]

            video_path = Path(
                row["video"]
            )

            video_name = video_path.stem

            processed_file = (
                PROCESSED_DIR
                / self.split
                / f"{original_index:04d}_{video_name}.npy"
            )

            if not processed_file.exists():

                raise FileNotFoundError(
                    f"Processed file not found:\n"
                    f"{processed_file}"
                )

            X[i] = np.load(
                processed_file
            )

            y[i] = self.label_to_index[
                row["label"]
            ]

        return X, y

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    train_generator = ISLDataGenerator(
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_generator = ISLDataGenerator(
        split="val",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_generator = ISLDataGenerator(
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_generator,
        val_generator,
        test_generator
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ISL MEMORY-EFFICIENT DATA LOADER TEST")
    print("=" * 60)

    train_generator, val_generator, test_generator = (
        create_datasets()
    )

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(
        f"\nTrain samples : {len(train_generator.df)}"
    )

    print(
        f"Validation    : {len(val_generator.df)}"
    )

    print(
        f"Test samples   : {len(test_generator.df)}"
    )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    print(
        f"Classes        : {NUM_CLASSES}"
    )

    # --------------------------------------------------------
    # Load ONLY ONE batch
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LOADING ONE TRAINING BATCH")
    print("=" * 60)

    X_batch, y_batch = (
        train_generator[0]
    )

    print(
        "\nBatch X shape:",
        X_batch.shape
    )

    print(
        "Batch y shape:",
        y_batch.shape
    )

    print(
        "X dtype:",
        X_batch.dtype
    )

    print(
        "y dtype:",
        y_batch.dtype
    )

    print(
        "X min:",
        X_batch.min()
    )

    print(
        "X max:",
        X_batch.max()
    )

    print(
        "\nBatch labels:"
    )

    print(y_batch)

    print("\n" + "=" * 60)
    print("DATA LOADER TEST COMPLETE")
    print("=" * 60)
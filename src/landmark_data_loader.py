import os
import glob
import numpy as np
import tensorflow as tf

BATCH_SIZE = 16
SEQUENCE_LENGTH = 32
FEATURES = 126
NUM_CLASSES = 49


class LandmarkDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, split, batch_size=BATCH_SIZE, shuffle=True, **kwargs):
        super().__init__(**kwargs)

        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.files = sorted(
            glob.glob(
                f"dataset/landmarks/{split}/*.npy"
            )
        )

        # Build label mapping from folder-independent filename
        self.class_names = sorted(
            set(
                os.path.basename(f).split("__")[0]
                for f in self.files
            )
        )

        self.class_to_index = {
            name: i
            for i, name in enumerate(self.class_names)
        }

        self.indices = np.arange(len(self.files))

        self.on_epoch_end()

        print(
            f"{split}: {len(self.files)} sequences"
        )

    def __len__(self):
        return int(
            np.ceil(
                len(self.files) / self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        X = []
        y = []

        for idx in batch_indices:

            file_path = self.files[idx]

            sequence = np.load(
                file_path
            ).astype(np.float32)

            label_name = os.path.basename(
                file_path
            ).split("__")[0]

            label = self.class_to_index[
                label_name
            ]

            X.append(sequence)
            y.append(label)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.int32)
        )

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":

    train = LandmarkDataGenerator(
        "train"
    )

    val = LandmarkDataGenerator(
        "val"
    )

    test = LandmarkDataGenerator(
        "test",
        shuffle=False
    )

    X, y = train[0]

    print("\n========== CHECK ==========")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)
    print("X min:", X.min())
    print("X max:", X.max())
    print("Labels:", y)
    print("Classes:", train.class_names)
    print("Number of classes:", len(train.class_names))
    print("===========================")

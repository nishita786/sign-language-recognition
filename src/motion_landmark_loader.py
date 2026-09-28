import os
import glob
import numpy as np
import tensorflow as tf


class MotionLandmarkDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        split,
        batch_size=16,
        shuffle=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_dir = os.path.join(
            "dataset",
            "landmarks_motion",
            split
        )

        self.files = sorted(
            glob.glob(
                os.path.join(
                    self.data_dir,
                    "*.npy"
                )
            )
        )

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

        self.labels = np.array([
            self.class_to_index[
                os.path.basename(f).split("__")[0]
            ]
            for f in self.files
        ])

        self.indices = np.arange(
            len(self.files)
        )

        self.on_epoch_end()

        print(
            f"{split}: {len(self.files)} sequences"
        )

    def __len__(self):
        return int(
            np.ceil(
                len(self.files) /
                self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        X = np.zeros(
            (
                len(batch_indices),
                32,
                252
            ),
            dtype=np.float32
        )

        y = np.zeros(
            len(batch_indices),
            dtype=np.int32
        )

        for i, idx in enumerate(batch_indices):

            X[i] = np.load(
                self.files[idx]
            )

            y[i] = self.labels[idx]

        return X, y

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(
                self.indices
            )

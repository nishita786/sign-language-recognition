import tensorflow as tf
from tensorflow.keras import layers, models

SEQUENCE_LENGTH = 32
FEATURES = 126
NUM_CLASSES = 49


def build_model():

    inputs = layers.Input(
        shape=(SEQUENCE_LENGTH, FEATURES)
    )

    x = layers.LayerNormalization()(inputs)

    x = layers.LSTM(
        128,
        return_sequences=True
    )(x)

    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(
        64
    )(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()

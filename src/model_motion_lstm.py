import tensorflow as tf
from tensorflow.keras import layers, models


def build_motion_model():

    inputs = layers.Input(
        shape=(32, 252)
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
        49,
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

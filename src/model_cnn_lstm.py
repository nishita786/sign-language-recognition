import tensorflow as tf

from tensorflow.keras import layers, models


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 32
IMG_SIZE = 224
NUM_CLASSES = 49


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

def build_frame_cnn():

    cnn = models.Sequential(
        [
            layers.Input(
                shape=(IMG_SIZE, IMG_SIZE, 3)
            ),

            # Block 1
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(
                (2, 2)
            ),

            # Block 2
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(
                (2, 2)
            ),

            # Block 3
            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(
                (2, 2)
            ),

            # Convert feature maps into a feature vector
            layers.GlobalAveragePooling2D(),

            layers.Dropout(0.3)
        ],
        name="custom_cnn"
    )

    return cnn


# ============================================================
# CNN + LSTM MODEL
# ============================================================

def build_cnn_lstm():

    frame_cnn = build_frame_cnn()

    inputs = layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        name="video_input"
    )

    # Apply the CNN independently to every frame
    frame_features = layers.TimeDistributed(
        frame_cnn,
        name="frame_feature_extractor"
    )(inputs)

    # Learn temporal relationships between frames
    x = layers.LSTM(
        128,
        return_sequences=False,
        name="temporal_lstm"
    )(frame_features)

    x = layers.Dropout(
        0.4
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.3
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="sign_prediction"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Custom_CNN_LSTM"
    )

    return model


# ============================================================
# BUILD MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CUSTOM CNN + LSTM")
    print("=" * 60)

    model = build_cnn_lstm()

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nMODEL SUMMARY")
    print("=" * 60)

    model.summary()

    print("\n" + "=" * 60)
    print("MODEL CREATED SUCCESSFULLY")
    print("=" * 60)
import tensorflow as tf

from tensorflow.keras import layers, models


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 32
IMG_SIZE = 224
NUM_CLASSES = 49


# ============================================================
# MOBILENETV2 FEATURE EXTRACTOR
# ============================================================

def build_mobilenet():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained weights initially
    base_model.trainable = False

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # MobileNetV2 expects inputs in approximately [-1, 1]
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(inputs)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dropout(0.2)(x)

    return models.Model(
        inputs,
        outputs,
        name="MobileNetV2_Feature_Extractor"
    )


# ============================================================
# MOBILENETV2 + LSTM
# ============================================================

def build_mobilenet_lstm():

    frame_model = build_mobilenet()

    inputs = layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        name="video_input"
    )

    # Extract features from every frame
    frame_features = layers.TimeDistributed(
        frame_model,
        name="frame_feature_extractor"
    )(inputs)

    # Learn temporal relationships
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
        name="MobileNetV2_LSTM"
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MOBILENETV2 + LSTM")
    print("=" * 60)

    model = build_mobilenet_lstm()

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
    print("MOBILENETV2 + LSTM CREATED SUCCESSFULLY")
    print("=" * 60)
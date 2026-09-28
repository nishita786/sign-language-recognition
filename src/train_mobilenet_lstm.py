import tensorflow as tf
from pathlib import Path
import json

from data_loader import create_datasets
from model_mobilenet_lstm import build_mobilenet_lstm


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRAINING MOBILENETV2 + LSTM")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nCreating data generators...")

    train_generator, val_generator, test_generator = (
        create_datasets()
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding MobileNetV2 + LSTM...")

    model = build_mobilenet_lstm()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel ready.")

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR /
        "mobilenet_lstm_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR /
        "mobilenet_lstm_final.keras"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        test_generator,
        verbose=1
    )

    print(
        f"\nTest Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR /
        "mobilenet_lstm_history.json"
    )

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            history.history,
            f
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nBest model saved to:\n"
        f"{checkpoint_path}"
    )

    print(
        f"\nFinal model saved to:\n"
        f"{final_model_path}"
    )

    print(
        f"\nHistory saved to:\n"
        f"{history_path}"
    )


if __name__ == "__main__":
    main()
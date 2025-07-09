import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
DATASET_DIR = Path("braille_dataset")
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 50  # Increased epochs for better convergence
SEED = 42

MODEL_OUTPUT_PATH = Path("braille_model")
BEST_MODEL_FILE = MODEL_OUTPUT_PATH / "braille_cnn_best.keras"  # Use .keras format
FINAL_MODEL_FILE = MODEL_OUTPUT_PATH / "braille_cnn_final.keras"


def main():
    """Main function to run the training pipeline."""
    # Create output directory if it doesn't exist
    MODEL_OUTPUT_PATH.mkdir(exist_ok=True)

    # 1. Create tf.data Datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "val",
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Define Data Augmentation as a Keras Layer
    data_augmentation = models.Sequential(
        [
            layers.Rescaling(1.0 / 255),
            layers.RandomRotation(0.07),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    # 3. Build the CNN Model with Augmentation Layer
    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
            data_augmentation,
            # Block 1
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2, 2),
            # Block 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2, 2),
            # Block 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2, 2),
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Slower learning rate
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # 4. Setup Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_FILE,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
    )
    earlystop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,  # Increased patience
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
    )

    # 5. Train the Model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    )
    print("--- Model Training Finished ---")

    # 6. Evaluate on Test Set
    print("\n--- Evaluating on Test Set ---\n")
    # Load the best model saved by ModelCheckpoint
    best_model = models.load_model(BEST_MODEL_FILE)
    test_loss, test_acc = best_model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # 7. Save Final Model and Mappings
    print(f"\nSaving final model to {FINAL_MODEL_FILE}")
    best_model.save(FINAL_MODEL_FILE)

    # Save the class indices and character mappings for inference
    class_indices = {name: i for i, name in enumerate(class_names)}
    with open(MODEL_OUTPUT_PATH / "class_indices.json", "w") as f:
        json.dump(class_indices, f)

    print("Model and mappings saved successfully.")

    # 8. Plot and Save Training Metrics
    plot_history(history)


def plot_history(history):
    """Plots training and validation accuracy and loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.savefig(MODEL_OUTPUT_PATH / "training_history.png")
    print(
        f"Training history plot saved to {MODEL_OUTPUT_PATH / 'training_history.png'}"
    )
    # plt.show()


if __name__ == "__main__":
    main()

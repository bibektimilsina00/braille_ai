import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurations
DATASET_DIR = "braille_dataset"
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# Folder-safe mapping used in dataset generation
folder_name_map = {
    "ksha": "क्ष",
    "tra": "त्र",
    "gya": "ज्ञ",
}

# 1. Prepare ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# 2. Map ASCII-safe folder names back to Nepali characters for classes
inv_map = {}
for folder_name, index in train_generator.class_indices.items():
    nep_char = folder_name_map.get(
        folder_name, folder_name
    )  # default to folder_name if not in map
    inv_map[index] = nep_char

print(f"Detected {len(train_generator.class_indices)} classes.")
print("Class index to Nepali character mapping:")
for idx in sorted(inv_map.keys()):
    print(f"  {idx}: {inv_map[idx]}")

num_classes = len(train_generator.class_indices)

# 3. Build CNN Model
model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 4. Setup callbacks
checkpoint_cb = callbacks.ModelCheckpoint(
    "braille_cnn_best.h5", save_best_only=True, monitor="val_accuracy", mode="max"
)
earlystop_cb = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

# 5. Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb],
)

# 6. Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Save final model
model.save("braille_cnn_model.h5")

# 8. Plot metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()

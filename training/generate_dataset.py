import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- Configuration ---
IMAGE_SIZE = 100
DOT_RADIUS = 10
SAMPLES_PER_CLASS = 1000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO is implicitly 0.1

# --- Braille Mappings ---

# Dot positions in a 2x3 Braille cell, numbered 1-6
# 1 4
# 2 5
# 3 6
DOT_POSITIONS = [
    (30, 30),  # dot 1
    (30, 50),  # dot 2
    (30, 70),  # dot 3
    (70, 30),  # dot 4
    (70, 50),  # dot 5
    (70, 70),  # dot 6
]

# Corrected Nepali Braille character mappings (Bharati Braille)
# Using dot numbers (1-6) as per user request and standard charts.
NEPALI_BRAILLE_MAP_NUMERIC = {
    "क": "15", "ख": "26", "ग": "1234", "घ": "136", "ङ": "256", "च": "12", "छ": "16", "ज": "234", "झ": "456", "ञ": "34",
    "ट": "23456", "ठ": "2346", "ड": "1236", "ढ": "123456", "ण": "2456", "त": "2345", "थ": "1246", "द": "124",
    "ध": "2356", "न": "1245", "प": "1235", "फ": "345", "ब": "13", "भ": "24", "म": "125", "य": "12456", "र": "1345",
    "ल": "135", "व": "1356", "श": "126", "ष": "12356", "स": "235", "ह": "134", "क्ष": "12345", "ज्ञ": "146",
    "अ": "1", "आ": "245", "इ": "23", "ई": "45", "उ": "156", "ऊ": "1346", "ए": "14", "ऐ": "25",
    "ओ": "145", "औ": "236", "अं": "46", "अः": "6", "ऋ": "1256",
    "virama": "2",      # ् (Halant)
    "danda": "346",     # । (Purna Viram)
}

# Mapping for creating folder names that are filesystem-safe
FOLDER_NAME_MAP = {
    "क": "ka", "ख": "kha", "ग": "ga", "घ": "gha", "ङ": "nga", "च": "cha", "छ": "chha", "ज": "ja", "झ": "jha", "ञ": "yna",
    "ट": "ta", "ठ": "tha", "ड": "da", "ढ": "dha", "ण": "naa", "त": "taa", "थ": "thaa", "द": "daa", "ध": "dhaa", "न": "na",
    "प": "pa", "फ": "pha", "ब": "ba", "भ": "bha", "म": "ma", "य": "ya", "र": "ra", "ल": "la", "व": "wa",
    "श": "sha", "ष": "shha", "स": "sa", "ह": "ha",
    "क्ष": "ksha", "ज्ञ": "gya",
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai",
    "ओ": "o", "औ": "au", "अं": "am", "अः": "ah", "ऋ": "ri",
}


def convert_numeric_to_binary(numeric_pattern):
    """Converts a numeric dot pattern (e.g., '15') to a 6-bit binary string (e.g., '100010')."""
    binary_pattern = ['0'] * 6
    for digit in numeric_pattern:
        dot_index = int(digit) - 1
        if 0 <= dot_index < 6:
            binary_pattern[dot_index] = '1'
    return "".join(binary_pattern)


def create_braille_image(binary_dot_pattern):
    """Creates a visual image of a Braille character."""
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255  # White background
    for i, val in enumerate(binary_dot_pattern):
        if val == '1':
            center = DOT_POSITIONS[i]
            # Add slight random offset to dot position for variety
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            cv2.circle(img, (center[0] + offset_x, center[1] + offset_y), DOT_RADIUS, 0, -1)  # Black filled circle
    return img


def augment_image(img):
    """Applies random augmentations to an image to increase dataset variety."""
    # Brightness
    brightness = random.uniform(0.7, 1.3)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Rotation
    angle = random.uniform(-7, 7)
    M = cv2.getRotationMatrix2D((IMAGE_SIZE // 2, IMAGE_SIZE // 2), angle, 1)
    img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), borderValue=255)

    # Blur
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Noise
    if random.random() > 0.5:
        noise = np.random.randint(0, 15, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        img = cv2.add(img, noise, dtype=cv2.CV_8U)
        img[img > 255] = 255

    return img


def generate_dataset():
    """Generates and saves the complete Braille dataset."""
    base_path = Path("braille_dataset")
    if base_path.exists():
        print(f"Dataset directory '{base_path}' already exists. Removing it to regenerate.")
        shutil.rmtree(base_path)

    print("Generating new Braille dataset...")

    # Convert main map to use folder-safe names
    braille_map_for_folders = {
        FOLDER_NAME_MAP.get(char, char): pattern
        for char, pattern in NEPALI_BRAILLE_MAP_NUMERIC.items()
    }

    for label, numeric_pattern in tqdm(braille_map_for_folders.items(), desc="Generating dataset"):
        binary_pattern = convert_numeric_to_binary(numeric_pattern)

        for i in range(SAMPLES_PER_CLASS):
            img = create_braille_image(binary_pattern)
            img = augment_image(img)

            # Determine split (train/val/test)
            rand_val = random.random()
            if rand_val < TRAIN_RATIO:
                split = "train"
            elif rand_val < TRAIN_RATIO + VAL_RATIO:
                split = "val"
            else:
                split = "test"

            out_dir = base_path / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{label}_{i}.png"
            cv2.imwrite(str(out_path), img)

    print(f"Dataset generation complete. Saved in '{base_path}'.")

    # Create a file with class names (using the folder-safe names)
    class_names = sorted(braille_map_for_folders.keys())
    with open(base_path / "class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print("Class names saved to class_names.txt")

    # Create a mapping file from folder name back to Nepali character
    folder_to_char_map = {v: k for k, v in FOLDER_NAME_MAP.items()}
    # Add the ones that were not in the map
    for char in NEPALI_BRAILLE_MAP_NUMERIC:
        if char not in FOLDER_NAME_MAP:
            folder_to_char_map[char] = char

    with open(base_path / "folder_to_char_map.txt", "w", encoding="utf-8") as f:
        import json
        json.dump(folder_to_char_map, f, ensure_ascii=False, indent=2)
    print("Folder to character map saved to folder_to_char_map.txt")


if __name__ == "__main__":
    generate_dataset()
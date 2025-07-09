import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Configuration
IMAGE_SIZE = 100
DOT_RADIUS = 10
SAMPLES_PER_CLASS = 1000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Dot positions in a 2x3 Braille cell
DOT_POSITIONS = [
    (30, 20),  # dot 1
    (30, 50),  # dot 2
    (30, 80),  # dot 3
    (70, 20),  # dot 4
    (70, 50),  # dot 5
    (70, 80),  # dot 6
]

# Nepali Braille characters only (Bharati Braille)
braille_map = {
    # Nepali consonants
    "क": "101000",
    "ख": "111000",
    "ग": "101100",
    "घ": "111100",
    "ङ": "100110",
    "च": "101010",
    "छ": "111010",
    "ज": "101110",
    "झ": "111110",
    "ञ": "100010",
    "ट": "101001",
    "ठ": "111001",
    "ड": "101101",
    "ढ": "111101",
    "ण": "101011",
    "त": "101111",
    "थ": "111111",
    "द": "101100",
    "ध": "111100",
    "न": "100011",
    "प": "110000",
    "फ": "110001",
    "ब": "110010",
    "भ": "110011",
    "म": "100101",
    "य": "110100",
    "र": "110101",
    "ल": "110110",
    "व": "100111",
    "श": "110111",
    "ष": "111000",
    "स": "111001",
    "ह": "111010",
    "क्ष": "111011",
    "त्र": "111100",
    "ज्ञ": "111101",
    # Nepali vowels
    "अ": "100000",
    "आ": "101000",
    "इ": "110000",
    "ई": "110100",
    "उ": "100100",
    "ऊ": "111000",
    "ए": "111100",
    "ऐ": "101100",
    "ओ": "011000",
    "औ": "011100",
    "ऋ": "100010",
    "ॠ": "101010",
}

# Folder-safe mapping for compound characters
folder_name_map = {
    "क्ष": "ksha",
    "त्र": "tra",
    "ज्ञ": "gya",
}


def create_braille_image(dot_pattern):
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255  # white background
    for i, val in enumerate(dot_pattern):
        if val == "1":
            center = DOT_POSITIONS[i]
            cv2.circle(img, center, DOT_RADIUS, 0, -1)  # black filled circle
    return img


def augment_image(img):
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((IMAGE_SIZE // 2, IMAGE_SIZE // 2), angle, 1)
    img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), borderValue=255)
    return img


def save_dataset(mapping):
    base_path = Path("braille_dataset")
    for label, pattern in tqdm(mapping.items(), desc="Generating Nepali dataset"):
        for i in range(SAMPLES_PER_CLASS):
            img = create_braille_image(pattern)
            img = augment_image(img)
            if i < SAMPLES_PER_CLASS * TRAIN_RATIO:
                split = "train"
            elif i < SAMPLES_PER_CLASS * (TRAIN_RATIO + VAL_RATIO):
                split = "val"
            else:
                split = "test"
            folder_label = folder_name_map.get(label, label)
            out_dir = base_path / split / folder_label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{folder_label}_{i}.png"
            cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    save_dataset(braille_map)

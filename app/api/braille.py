import io

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter()

# Load model once at startup
import os

# Construct absolute path relative to the app directory
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(app_dir, "models", "braille_cnn_best.h5")

#MODEL_PATH = "../models/braille_cnn_best.h5"  # Comment out the relative path

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Failed to load model from {MODEL_PATH}: {e}")

# Class index to Nepali character mapping (update as needed)
inv_map = {
    0: "अ",
    1: "अः",
    2: "ऐ",
    3: "अं",
    4: "औ",
    5: "भ",
    6: "ब",
    7: "च",
    8: "छ",
    9: "ड",
    10: "द",
    11: "danda",
    12: "ढ",
    13: "ध",
    14: "ए",
    15: "ई",
    16: "ग",
    17: "घ",
    18: "ज्ञ",
    19: "ह",
    20: "इ",
    21: "ज",
    22: "झ",
    23: "क",
    24: "ख",
    25: "क्ष",
    26: "ल",
    27: "म",
    28: "न",
    29: "ण",
    30: "ङ",
    31: "ओ",
    32: "ऊ",
    33: "प",
    34: "फ",
    35: "र",
    36: "ऋ",
    37: "स",
    38: "श",
    39: "ष",
    40: "ट",
    41: "त",
    42: "ठ",
    43: "थ",
    44: "उ",
    45: "virama",
    46: "व",
    47: "य",
    48: "ञ",
}

IMG_SIZE = (100, 100)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize(IMG_SIZE)
    arr = np.array(image)
    arr = arr.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 1))
    return arr


@router.post("/predict_braille/")
async def predict_braille(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        contents = await file.read()
        img_arr = preprocess_image(contents)
        preds = model.predict(img_arr)
        pred_idx = int(np.argmax(preds))
        pred_char = inv_map.get(pred_idx, str(pred_idx))
        return JSONResponse(
            {"predicted_class": pred_idx, "predicted_character": pred_char}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

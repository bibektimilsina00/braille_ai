import io

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter()

# Load model once at startup
MODEL_PATH = "braille_model/braille_cnn_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

# Class index to Nepali character mapping (update as needed)
inv_map = {
    0: "क",
    1: "ख",
    2: "ग",
    3: "घ",
    4: "ङ",
    5: "च",
    6: "छ",
    7: "ज",
    8: "झ",
    9: "ञ",
    10: "ट",
    11: "ठ",
    12: "ड",
    13: "ढ",
    14: "ण",
    15: "त",
    16: "थ",
    17: "द",
    18: "ध",
    19: "न",
    20: "प",
    21: "फ",
    22: "ब",
    23: "भ",
    24: "म",
    25: "य",
    26: "र",
    27: "ल",
    28: "व",
    29: "श",
    30: "ष",
    31: "स",
    32: "ह",
    33: "क्ष",
    34: "त्र",
    35: "ज्ञ",
    36: "अ",
    37: "आ",
    38: "इ",
    39: "ई",
    40: "उ",
    41: "ऊ",
    42: "ऋ",
    43: "ॠ",
    44: "ए",
    45: "ऐ",
    46: "ओ",
    47: "औ",
}

IMG_SIZE = (100, 100)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize(IMG_SIZE)
    arr = np.array(image) / 255.0
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

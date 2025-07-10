import io

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter()

import json

# Load model and class mappings once at startup
MODEL_PATH = "braille_model/braille_cnn_final.keras"
CLASS_INDICES_PATH = "braille_model/class_indices.json"
FOLDER_TO_CHAR_MAP_PATH = "braille_dataset/folder_to_char_map.txt"

model = None
inv_map = {}
nepali_char_map = {}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
        # Invert the map for prediction: index -> character
        inv_map = {v: k for k, v in class_indices.items()}
    with open(FOLDER_TO_CHAR_MAP_PATH, "r", encoding="utf-8") as f:
        nepali_char_map = json.load(f)
except Exception as e:
    print(f"Failed to load model or class indices: {e}")

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
        nepali_char = nepali_char_map.get(pred_char, "N/A")
        return JSONResponse(
            {"predicted_class": pred_idx, "predicted_character": pred_char, "nepali_character": nepali_char}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

# Load model once
model = tf.keras.models.load_model("alzheimers_cnn_model.h5")
class_names = ["Normal", "Mild", "Moderate", "Severe"]

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((128, 128))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 128, 128, 1)
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    x = preprocess(img_bytes)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return {
        "prediction": class_names[idx],
        "index": idx,
        "probabilities": probs.tolist()
    }

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI(title="Alzheimer's MRI Predictor API")

# Enable CORS so Streamlit frontend can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once at startup
model = tf.keras.models.load_model("alzheimers_cnn_model.h5")
class_names = ["Normal", "Mild", "Moderate", "Severe"]

def preprocess(image_bytes):
    """Preprocess uploaded image for model prediction"""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((128, 128))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 128, 128, 1)
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict Alzheimer stage from uploaded MRI"""
    img_bytes = await file.read()
    x = preprocess(img_bytes)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return {
        "prediction": class_names[idx],
        "index": idx,
        "probabilities": probs.tolist()
    }

@app.get("/")
def read_root():
    return {"message": "Alzheimer's MRI Predictor API is running."}

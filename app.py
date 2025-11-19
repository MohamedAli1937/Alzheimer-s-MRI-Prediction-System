import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf  


# Load trained model
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("alzheimers_cnn_model.h5")
    return model

model = load_cnn_model()


# Classes
class_names = ["Normal", "Mild", "Moderate", "Severe"]


# Predict function
def predict_alzheimers(img: Image.Image):
    img = img.convert('L')  # convert to grayscale
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = img_array[np.newaxis, ..., np.newaxis]  # shape (1,128,128,1)
    
    probs = model.predict(img_array)[0]
    pred_class = np.argmax(probs)
    pred_label = class_names[pred_class]
    
    return pred_class, pred_label, probs

# Streamlit App Layout
st.set_page_config(page_title="Alzheimer's MRI Predictor", layout="centered")

st.title("🧠 Alzheimer's MRI Prediction System")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", width=250)  # smaller display

    if st.button("Predict"):
        with st.spinner("Analyzing MRI..."):
            pred_class, pred_label, probs = predict_alzheimers(image)
        
        st.success(f"Prediction: **{pred_label}** (Class {pred_class})")
        
        st.markdown("### Class Probabilities:")
        for i, prob in enumerate(probs):
            st.write(f"{class_names[i]}: {prob:.2f}")
            st.progress(float(prob))
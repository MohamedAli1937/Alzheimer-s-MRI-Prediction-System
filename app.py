import streamlit as st
from PIL import Image
import requests

# ✅ Update this to deployed FastAPI URL
API_URL = "https://alzheimer-s-mri-prediction-system.onrender.com/predict"
class_names = ["Normal", "Mild", "Moderate", "Severe"]

st.set_page_config(page_title="Alzheimer's MRI Predictor", layout="centered")
st.title("🧠 Alzheimer's MRI Prediction System")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", width=300)

    if st.button("Predict"):
        with st.spinner("Analyzing MRI..."):
            # Send the file to FastAPI correctly
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
            else:
                st.success(f"Prediction: **{data['prediction']}** (Class {data['index']})")
                
                st.markdown("### Class Probabilities:")
                for i, prob in enumerate(data['probabilities']):
                    st.write(f"{class_names[i]}: {prob:.2f}")
                    st.progress(float(prob))



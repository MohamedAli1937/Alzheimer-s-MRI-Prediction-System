import streamlit as st
from PIL import Image
import requests

API_URL = "http://127.0.0.1:8000/predict"
class_names = ["Normal", "Mild", "Moderate", "Severe"]

st.set_page_config(page_title="Alzheimer's MRI Predictor", layout="centered")
st.title("🧠 Alzheimer's MRI Prediction System")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", width=250)

    if st.button("Predict"):
        with st.spinner("Analyzing MRI..."):
            response = requests.post(API_URL, files={"file": uploaded_file.getvalue()})
            data = response.json()

        st.success(f"Prediction: **{data['prediction']}** (Class {data['index']})")
        
        st.markdown("### Class Probabilities:")
        for i, prob in enumerate(data['probabilities']):
            st.write(f"{class_names[i]}: {prob:.2f}")
            st.progress(float(prob))

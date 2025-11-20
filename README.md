**🧠 Alzheimer’s MRI Prediction – Cognify**
=======================================

A deep learning web application for predicting Alzheimer’s disease stage from MRI scans.

## 📌 **Overview**


Cognify is a **CNN-based medical image classification system** designed to identify Alzheimer’s Disease stages using MRI scans.
It provides fast, user-friendly predictions useful for educational and research purposes.

> ⚠️ **This tool is not intended for clinical or medical diagnosis.**

## 🎯 **Objective**
Cognify is a deep learning web application that classifies 128×128 grayscale MRI images into four Alzheimer’s stages:

* 😊 Normal

* 🙂 Mild

* 😐 Moderate

* 😱 Severe

It predicts the most likely stage and shows class probabilities, helping visualize risk and understanding of the disease.
## 🧩 **How It Works**

### 🏗️ **Project Structure**

```
📦 Alzheimer-s-MRI-Prediction-System/
│
├── app.py                    # Streamlit frontend
├── backend.py                # FastAPI backend
├── alzheimers_cnn_model.h5   # Trained CNN model
├── train.parquet             # Training dataset
├── test.parquet              # Test dataset
├── cognify.ipynb             # Training / EDA notebook
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── images/
      └── streamlit_demo.jpg

```
### 🌐 **Streamlit Web App**

🖼️ **Screenshot**

![Streamlit Screenshot](Images/streamlit_demo.jpg) 

## **🌐 Deployment**

The project is fully deployed on Render:

Backend API:
https://alzheimer-s-mri-prediction-system.onrender.com

Streamlit frontend:
https://alzheimer-mri-streamlit.onrender.com


## 🚀 **How to Deploy Locally**
### **1️⃣ Clone the Repository**
```bash
git Clone https://github.com/MohamedAli1937/Alzheimer-s-MRI-Prediction-System.git
cd Alzheimer-s-MRI-Prediction-System
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### **3️⃣ Install Requirements**
```bash
pip install -r requirements.txt  
```
### **4️⃣ Run the FastAPI backend**
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### **5️⃣ Run the Streamlit App**

```bash
streamlit run app.py  
```
    

## 📈 **Model Performance**

Example: 

**Training Accuracy: 81.59%**

**Validation Accuracy: 73.8%**
## 🧰 **Tech Stack**
*   **Python 3.10+**
    
*   **TensorFlow / Keras**
    
*   **NumPy**
    
*   **Pandas**
    
*   **PyArrow**
    
*   **Streamlit**
    
*   **Pillow**
    


## 🔮 **Future Improvements**

*   Add Grad-CAM heatmaps for explainability
    
*   Support for 3D MRI volumes (NIfTI)
    
*   Deployment on HuggingFace Spaces
    
*   Faster inference using TensorFlow Lite
    

## 🙌 **Acknowledgements**

Special thanks to all open MRI datasets used in research and experimentation.

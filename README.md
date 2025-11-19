🧠 Alzheimer’s MRI Prediction – Cognify
=======================================

A deep learning web application for predicting Alzheimer’s disease stage from MRI scans.

📌 **Overview**
---------------

Cognify is a **CNN-based medical image classification system** designed to identify Alzheimer’s Disease stages using MRI scans.The project includes:

*   A **trained TensorFlow model (alzheimers\_cnn\_model.h5)**
    
*   A **Streamlit web interface (app.py)**
    
*   **Training and test datasets (train.parquet, test.parquet)**
    
*   **A fully reproducible pipeline (cognify.ipynb)**
    

It provides fast, user-friendly predictions useful for educational and research purposes.

> ⚠️ **This tool is not intended for clinical or medical diagnosis.**

🎯 **Objective**
----------------

The goal of this project is to:

1.  Build a **Convolutional Neural Network (CNN)** capable of classifying MRI images into Alzheimer’s stages:
    
    *   Normal
        
    *   Mild
        
    *   Moderate
        
    *   Severe
        
2.  Provide a **simple web interface** where users can upload an MRI image and receive a prediction in seconds.
    
3.  Demonstrate a full ML workflow:
    
    *   Data preprocessing
        
    *   Training
        
    *   Evaluation
        
    *   Deployment with Streamlit
        

🧩 **How It Works**
-------------------

### **1️⃣ Model Training**

The CNN model was trained on 128×128 grayscale MRI images using a dataset stored in the parquet files:

*   **train.parquet** — training samples
    
*   **test.parquet** — evaluation samples
    

Training code is available in cognify.ipynb.

### **2️⃣ Preprocessing**

Uploaded MRI images are:

*   Converted to grayscale
    
*   Resized to **128×128**
    
*   Normalized (divided by 255.0)
    
*   Expanded to shape (1, 128, 128, 1)
    

### **3️⃣ Prediction**

The model outputs 4 probabilities:

Class :

0 → Normal

1 → Mild

2 → Moderate

3 → Severe

The Streamlit app displays both:

*   The predicted class
    
*   Probabilities for each stage
    

🏗️ **Project Structure**
-------------------------
```
📦 Alzheimer-s-MRI-Prediction-System /
│
├── app.py                    # Streamlit web app
├── alzheimers_cnn_model.h5   # Trained CNN model
├── train.parquet             # Training dataset
├── test.parquet              # Test dataset
├── cognify.ipynb             # Training / EDA notebook
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
└── images/
      streamlit_demo.jpg
```

🌐 **Streamlit Web App**
------------------------

### 🖼️ **Screenshot**

![Streamlit Screenshot](Images/streamlit_demo.jpg) 

🚀 **How to Run the App**
-------------------------

### **1️⃣ Clone the Repository**
```bash
git Clone https://github.com/MohamedAli1937/Alzheimer-s-MRI-Prediction-System.git
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

### **4️⃣ Run the Streamlit App**

```bash
streamlit run app.py  
```


🧰 **Tech Stack**
-----------------

*   **Python 3.10+**
    
*   **TensorFlow / Keras**
    
*   **NumPy**
    
*   **Pandas**
    
*   **PyArrow**
    
*   **Streamlit**
    
*   **Pillow**
    

📈 **Model Performance**
------------------------

**Training Accuracy: 81.59%**

**Validation Accuracy: 73.8%**


🔮 **Future Improvements**
--------------------------

*   Add Grad-CAM heatmaps for explainability
    
*   Support for 3D MRI volumes (NIfTI)
    
*   Deployment on HuggingFace Spaces
    
*   Faster inference using TensorFlow Lite
    

🙌 **Acknowledgements**
-----------------------

Special thanks to all open MRI datasets used in research and experimentation.

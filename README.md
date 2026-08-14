Lung Cancer Classification using  XAI Techniques

🧠 Project Overview

This project aims to classify lung cancer subtypes (Adenocarcinoma, Benign, and Squamous Cell Carcinoma) using a ResNet50-based deep learning model. To enhance transparency, it integrates Explainable AI (XAI) techniques – LIME and Grad-CAM – that visually and interpretably justify the model's predictions.

🔧 Technologies Used

Python 3.8+
PyTorch
Flask
OpenCV
NumPy, Matplotlib
LIME (for XAI)
Grad-CAM (for heatmap)
HTML, CSS, JS (for frontend)

📁 Folder Structure

Lung-Cancer-Detection-XAI/

├── Codes/                    
├── templates/               
├── Pre-Trained_Model/        
├── Phase1_Code/              
├── Docs/                     
├── requirements.txt
└── README.md

🚀 How to Run the Project

Clone this repo

Create a virtual environment

Install dependencies

Run pre-train.py to train the model

Run LIME_XAI.py and GRAD-CAM.py to visualize

Launch the web app via app.py

# Example
python Codes/pre-train.py

python Codes/LIME_XAI.py

python Codes/GRAD-CAM.py

python Codes/app.py

Then open your browser:
http://127.0.0.1:5000



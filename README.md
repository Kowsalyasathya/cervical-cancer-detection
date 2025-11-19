AI Based Multi-class Ensemble System For Cervical Cancer Detection

This project is an AI-Based Multi-Class Ensemble System designed to detect cervical cancer cell types from microscope images.
The system uses a deep learning ensemble model with 95% accuracy, combining EfficientNetB0 and Fine-Tuned MobileNetV2, and is deployed as a web application built with Flask.

📌 Overview

The system classifies cervical cell images into:

Malignant

Precancerous

Normal

The user provides basic details, uploads an image, and receives the prediction along with a downloadable PDF medical report.

The web app features a clean pink-themed UI, and is fully deployable using Render or Railway for a permanent URL.

⭐ Features

✔ Deep Learning Ensemble Model (EfficientNet + MobileNetV2)

✔ 95% accuracy on Mendeley Cervical Cancer 3-Class Dataset

✔ User-friendly web form (Name, Age, Gender, Phone, City, etc.)

✔ Image Upload & Real-Time Prediction

✔ Confidence Scores for all 3 classes

✔ PDF Report Generation (ReportLab)

✔ Deployment ready with Dockerfile and requirements.txt

✔ Runs locally in VS Code

📁 Project Structure
cervical-web-app/
│
├─ app.py                     # Flask backend
├─ requirements.txt           # All dependencies
├─ Dockerfile                 # Deployment container
├─ README.md                  
├─ models/
│   └─ ensemble_model.keras   # Final deployable model
│
├─ templates/
│   ├─ index.html             # Input form page
│   └─ result.html            # Result + PDF download page
│
└─ static/
    ├─ css/
    │   └─ styles.css         # Pink UI theme
    └─ uploads/               # Uploaded images + generated PDFs

🧠 Ensemble Model Details
1. EfficientNetB0

Pretrained on ImageNet

Good at identifying Normal cells

2. MobileNetV2 (Fine-Tuned)

Fine-tuned top layers

Best at detecting Malignant + Precancerous

3. Soft Voting Logic
final_prob = (efficientnet_prob + mobilenet_prob) / 2


The combined ensemble is saved as a single .keras model for easy deployment.

🧰 Technologies Used
Backend

Python

Flask

TensorFlow

NumPy

Pillow

ReportLab (PDF generation)

Frontend

HTML

CSS

Bootstrap

Deployment

Docker

Gunicorn

Render / Railway

🚀 How to Run This Project (VS Code)
1️⃣ Clone the project
git clone https://github.com/yourusername/cervical-web-app.git
cd cervical-web-app

2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

Linux / macOS
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Web App
python app.py


Open the browser at:

http://127.0.0.1:8501/


Upload an image → Get prediction → Download PDF.

🌍 Deployment (Render / Railway)

Push project to GitHub

Create New Web Service

Connect GitHub repo

Render auto-detects Dockerfile

Deploy → Get permanent public URL

Example:

https://cervical-cancer-detection.onrender.com

📄 PDF Report Includes:

Patient Details

Uploaded Image

Final Prediction

Confidence Scores

Date & Timestamp

Model used: EfficientNetB0 + MobileNetV2 Ensemble

🔮 Future Improvements

Add Grad-CAM heatmaps

Add patient history database

Add email PDF feature

Deploy inference on GPU

Add doctors login dashboard

✨ Acknowledgement

Special thanks to the Mendeley LBC dataset and open-source deep learning community.
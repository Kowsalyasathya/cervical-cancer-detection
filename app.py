# app.py
import os
import io
import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import gdown   # <-- Added for Google Drive model download

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

MODEL_PATH = "models/ensemble_model.keras"
FILE_ID = "1ERJKvrrgF3s8Sg8SnqPxs7Pahfxd3nTy"   # Your Google Drive file ID

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_a_random_secret"

# ─────────────────────────────────────────────
# DOWNLOAD MODEL IF MISSING
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("MODEL NOT FOUND. DOWNLOADING FROM GOOGLE DRIVE...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

CLASS_LABELS = ["Malignant", "Normal", "Precancerous"]

# ─────────────────────────────────────────────
# IMAGE HANDLING & PREDICTION
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def predict_from_path(img_path):
    x = preprocess_image(img_path)
    probs = model.predict(x)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_LABELS[pred_idx]
    confidences = {CLASS_LABELS[i]: float(probs[i]) for i in range(3)}
    return pred_label, confidences

# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def generate_pdf(user_info, image_path, prediction, confidences):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, "Cervical Cancer Detection Report")

    c.setFont("Helvetica", 11)
    y = height - 90
    for key, val in user_info.items():
        c.drawString(40, y, f"{key}: {val}")
        y -= 18

    y -= 6
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, f"Prediction: {prediction}")
    y -= 20

    c.setFont("Helvetica", 11)
    for k, v in confidences.items():
        c.drawString(40, y, f"{k}: {v*100:.2f}%")
        y -= 16

    try:
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        tmp = io.BytesIO()
        img.save(tmp, format="PNG")
        tmp.seek(0)
        x_img = width - 340
        y_img = height - 420
        c.drawImage(tmp, x_img, y_img, width=img.size[0], height=img.size[1])
    except Exception as e:
        print("Could not add image:", e)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 30, "Generated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    c.save()
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # User inputs
        user_info = {
            "Name": request.form.get("name", ""),
            "Date of Birth": request.form.get("dob", ""),
            "Age": request.form.get("age", ""),
            "Gender": request.form.get("gender", ""),
            "Date": request.form.get("date", ""),
            "Phone": request.form.get("phone", ""),
            "City": request.form.get("city", ""),
            "Pincode": request.form.get("pincode", "")
        }

        # File upload
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(save_path)

            prediction, confidences = predict_from_path(save_path)
            pdf_buffer = generate_pdf(user_info, save_path, prediction, confidences)

            pdf_name = fname.rsplit(".", 1)[0] + "_report.pdf"
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_name)

            with open(pdf_path, "wb") as f:
                f.write(pdf_buffer.getbuffer())

            return render_template(
                "result.html",
                image_url=url_for("static", filename=f"uploads/{fname}"),
                prediction=prediction,
                confidences=confidences,
                pdf_url=url_for("static", filename=f"uploads/{pdf_name}"),
                user_info=user_info
            )

        flash("Allowed types: png, jpg, jpeg")
        return redirect(request.url)

    return render_template("index.html")

# ─────────────────────────────────────────────
# LOCAL TEST MODE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)

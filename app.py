# app.py
import os
import io
import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Config
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MODEL_PATH = "models/ensemble_model.keras"   # ensure this file exists

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_a_random_secret"  # change for production

# Load ensemble model once
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")
CLASS_LABELS = ["Malignant", "Normal", "Precancerous"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_image(img_path, target_size=(224,224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def predict_from_path(img_path):
    x = preprocess_image(img_path)
    probs = model.predict(x)[0]    # shape (3,)
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_LABELS[pred_idx]
    confidences = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
    return pred_label, confidences

def generate_pdf(user_info, image_path, prediction, confidences):
    """
    Returns a BytesIO of the generated PDF.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, "Cervical Cancer Detection Report")

    # Meta
    c.setFont("Helvetica", 11)
    y = height - 90
    for key, val in user_info.items():
        c.drawString(40, y, f"{key}: {val}")
        y -= 18

    # Prediction
    y -= 6
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, f"Prediction: {prediction}")
    y -= 20
    c.setFont("Helvetica", 11)
    for k, v in confidences.items():
        c.drawString(40, y, f"{k}: {v*100:.2f}%")
        y -= 16

    # Add image thumbnail
    try:
        # keep aspect ratio
        img = Image.open(image_path)
        max_w, max_h = 300, 300
        img.thumbnail((max_w, max_h))
        tmp = io.BytesIO()
        img.save(tmp, format="PNG")
        tmp.seek(0)
        x_img = width - max_w - 40
        y_img = height - max_h - 120
        c.drawImage(tmp, x_img, y_img, width=img.size[0], height=img.size[1])
    except Exception as e:
        print("Could not add image to PDF:", e)

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 30, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()
    buffer.seek(0)
    return buffer

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect user info
        name = request.form.get("name", "")
        dob = request.form.get("dob", "")
        age = request.form.get("age", "")
        gender = request.form.get("gender", "")
        date = request.form.get("date", "")
        phone = request.form.get("phone", "")
        city = request.form.get("city", "")
        pincode = request.form.get("pincode", "")

        # file
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

            # predict
            pred_label, confidences = predict_from_path(save_path)

            # user info dict
            user_info = {
                "Name": name,
                "Date of Birth": dob,
                "Age": age,
                "Gender": gender,
                "Date": date,
                "Phone": phone,
                "City": city,
                "Pincode": pincode
            }

            # store info in session via redirect: we'll pass through query (or better: render result directly)
            # generate pdf buffer
            pdf_buffer = generate_pdf(user_info, save_path, pred_label, confidences)

            # Save PDF file to uploads (so user can download)
            pdf_name = fname.rsplit(".",1)[0] + "_report.pdf"
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_buffer.getbuffer())

            # Render result page
            return render_template("result.html",
                                   image_url=url_for("static", filename=f"uploads/{fname}"),
                                   prediction=pred_label,
                                   confidences=confidences,
                                   pdf_url=url_for("static", filename=f"uploads/{pdf_name}"),
                                   user_info=user_info)
        else:
            flash("Allowed image types: png, jpg, jpeg")
            return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    # For local testing; use gunicorn for production
    app.run(host="0.0.0.0", port=8501, debug=True)

# app.py
import os
import io
import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Paths
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/ensemble_model.keras"

ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_a_random_secret"

# ───────────────────────────────
# LOAD MODEL
# ───────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

CLASS_LABELS = ["Malignant", "Normal", "Precancerous"]

# ───────────────────────────────
# IMAGE PREDICTION
# ───────────────────────────────
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

# ───────────────────────────────
# ROUTES
# ───────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
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

        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected image")
            return redirect(request.url)

        if file and allowed_file(file.filename):

            fname = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(save_path)

            prediction, confidences = predict_from_path(save_path)

            return render_template(
                "result.html",
                image_url=url_for("static", filename=f"uploads/{fname}"),
                prediction=prediction,
                confidences=confidences,
                user_info=user_info
            )

        flash("Allowed types: png, jpg, jpeg")
        return redirect(request.url)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


# TEAM.NO.49:AI Based Multi Class Ensemble Learning System for Cervical Cancer Detection

## About

This project focuses on detecting cervical cancer using a deep learning ensemble architecture. The system classifies cervical cytology images into:

1.Normal

2.Precancerous

3.Malignant


The solution integrates EfficientNetB0 and MobileNetV2 (fine-tuned), combined using soft probability voting, achieving a final accuracy of 95%.
A Flask-based web app enables users to upload images and receive predictions, along with a downloadable medical-style PDF report.

## Features

📤 Upload cervical cell image

🤖 Real-time prediction using ensemble deep learning

📊 Confidence score for each class

📝 Enter patient details before prediction

📄 Downloadable PDF medical report

## Development Requirements

<img width="692" height="317" alt="Screenshot 2025-11-22 163047" src="https://github.com/user-attachments/assets/9c129954-58e6-44a2-b0e5-3fabb0288809" />

## System Architecture
![WhatsApp Image 2025-11-22 at 4 16 20 PM](https://github.com/user-attachments/assets/4c906d7c-5c5d-4836-81f4-829e67cf87ce)


📌 Ensemble Formula:
final_probabilities = (EfficientNetB0_output + MobileNetV2_output) / 2

## Methodology
### 1. Data Preprocessing

i) The images from the Mendeley Cervical Cytology Dataset were cleaned by removing corrupted or unreadable files.

ii) All images were resized to 224 × 224 px, normalized, and converted into a consistent RGB format suitable for CNN processing.

iii) Data augmentation techniques such as rotation, zoom, brightness shift, and horizontal flip were applied to improve generalization and reduce overfitting.

### 2. Model Training

i) Two deep learning models were used for feature extraction and classification:

1.EfficientNetB0 (Pretrained on ImageNet)

2.Fine-Tuned MobileNetV2

ii) The outputs of both models were combined using a soft probability voting ensemble technique, forming a final deployable model named:
ensemble_model.keras

iii) The model was trained in Google Colab using GPU acceleration with Adam optimizer, categorical cross-entropy loss, and early stopping to prevent overfitting.

### 3. Model Evaluation

Evaluation metrics included: accuracy, precision, recall, F1-score, and confusion matrix.

The ensemble model demonstrated improved performance across all classes compared to individual models.

The final deployed model achieved:

<img width="507" height="346" alt="Screenshot 2025-11-22 165254" src="https://github.com/user-attachments/assets/cc65fd52-1317-472a-b3d1-97722ddbdf0d" />


<img width="676" height="215" alt="Screenshot 2025-11-22 164428" src="https://github.com/user-attachments/assets/19dbfadd-41bb-41f2-9631-fe48edc8b746" />

### 4. Setup Instructions
#### Run the Flask Web App:
```
.\venv\Scripts\Activate
python app.py
```
#### Access Web Interface:
```
http://127.0.0.1:8000
http://172.20.10.5:8000
```
## Key Model Implementation Code
```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

# IMAGE SIZE & CLASS LABELS
img_size = (224, 224)
class_labels = ["Normal", "Precancerous", "Malignant"]

# MODEL 1: EfficientNetB0
base1 = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x1 = GlobalAveragePooling2D()(base1.output)
x1 = Dropout(0.4)(x1)
out1 = Dense(3, activation="softmax")(x1)
eff_model = Model(inputs=base1.input, outputs=out1)

# MODEL 2: Fine-Tuned MobileNetV2
base2 = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x2 = GlobalAveragePooling2D()(base2.output)
x2 = Dropout(0.4)(x2)
out2 = Dense(3, activation="softmax")(x2)
mobile_model = Model(inputs=base2.input, outputs=out2)

# Load trained models
eff_model = tf.keras.models.load_model("efficientnet_model.h5")
mobile_model = tf.keras.models.load_model("mobilenet_finetuned.h5")

# ENSEMBLE LAYER (Soft Voting)
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
eff_output = eff_model(input_layer)
mob_output = mobile_model(input_layer)

ensemble_output = tf.keras.layers.Average()([eff_output, mob_output])
ensemble_model = Model(inputs=input_layer, outputs=ensemble_output)

# Save Final Model
ensemble_model.save("ensemble_model.keras")
print("Final Ensemble Model Saved Successfully")

# IMAGE PREPROCESSING & PREDICTION
def preprocess_image(path):
    img = image.load_img(path, target_size=img_size)
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(img_path):
    img = preprocess_image(img_path)
    eff_pred = eff_model.predict(img)
    mob_pred = mobile_model.predict(img)
    final_prob = (eff_pred + mob_pred) / 2
    pred_class = np.argmax(final_prob)
    print("\nPredicted Class:", class_labels[pred_class])
    print("Confidence Score:", round(float(np.max(final_prob)) * 100, 2), "%")
```
## Project Structure

<img width="759" height="497" alt="Screenshot 2025-11-22 164737" src="https://github.com/user-attachments/assets/3d2d0ffd-f33b-4fe9-9944-711506c7e00a" />

## Results
The final ensemble model achieved an accuracy of 95%, providing strong performance in classifying cervical cytology images across all three classes (Normal, Precancerous, Malignant).

This system enables early-stage detection, which may support medical professionals and improve preventive healthcare outcomes.

## Output

#### Web-page asking for input from user
<img width="1920" height="1080" alt="Screenshot 2025-11-22 135355" src="https://github.com/user-attachments/assets/fae1ebc8-c470-45fa-9c80-dfef3310a6ac" />

<img width="1920" height="1080" alt="Screenshot 2025-11-22 140047" src="https://github.com/user-attachments/assets/206405b3-f58d-478b-820d-3f421755f16f" />

#### Web-page displays the result

<img width="1920" height="1080" alt="Screenshot 2025-11-22 140100" src="https://github.com/user-attachments/assets/feb13aaa-3d08-46db-a113-dae84dea3c68" />

## Future Enhancements

🔹 Store patient history using Firebase/MongoDB

🔹 Add batch image prediction

🔹 Deploy inference on GPU cloud

## References

Tan, M., & Le, Q. (2019) EfficientNet

Howard et al. (2017) MobileNet

Zhang, J., et al. (2020) Deep Learning Cervical Cancer Diagnosis

Zhou, K. & Chen (2024) Ensemble Learning for Medical Images

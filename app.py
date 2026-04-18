import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 Load SavedModel correctly
loaded = tf.saved_model.load("model_tf")
infer = loaded.signatures["serving_default"]

# -------------------------------
# Prediction Function
# -------------------------------
def predict_image(img_path):
    print("Image path:", img_path)

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    print("Running prediction...")

    # ✅ Correct prediction
    prediction = infer(tf.constant(img_array))
    prediction = list(prediction.values())[0].numpy()

    prob = float(prediction[0][0])

    if prob > 0.5:
        result = "🦴 Fracture Detected"
    else:
        result = "✅ Normal Bone"

    # -------------------------------
    # Simple image display (NO Grad-CAM for now)
    # -------------------------------
    heatmap_path = img_path  # just show same image

    return result, prob, heatmap_path

# -------------------------------
# Flask Route
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    image_path = ""
    prob = None
    heatmap_path = ""

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result, prob, heatmap_path = predict_image(filepath)
            image_path = filepath

    return render_template(
        'index.html',
        result=result,
        image_path=image_path,
        prob=prob,
        heatmap_path=heatmap_path
    )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

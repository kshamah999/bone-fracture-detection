import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Hide warnings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

print("🟢 Program started...")

# Load model
model = load_model("bone_fracture_model.keras", compile=False)
print("✅ Model loaded successfully!")

# Prediction function
def predict_image(img_path):
    try:
        print("\n📂 Checking:", img_path)
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        print("🔍 Predicting...")
        prediction = model.predict(img_array)

        print("📊 Prediction value:", prediction)

        # Result
        if prediction[0][0] > 0.5:
            print("🦴 Fracture Detected")
        else:
            print("✅ Normal Bone")

    except Exception as e:
        print("❌ Error:", e)

# -------------------------------
# Manual Input Loop
# -------------------------------

while True:
    img_path = input("👉 Enter image name (or type 'exit'):")

    if img_path.lower() == 'exit':
        print("👋 Exiting program...")
        break

    predict_image(img_path)

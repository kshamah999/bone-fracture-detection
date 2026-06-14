# 🦴 Bone Fracture Detection using Deep Learning

A Flask-based web application that detects bone fractures from X-ray images using a TensorFlow deep learning model.

## 🚀 Features

* Upload X-ray images through a simple web interface
* Detect whether a bone fracture is present
* Displays prediction confidence score
* Shows uploaded X-ray image
* Built with Flask and TensorFlow SavedModel

---

## 🛠️ Tech Stack

* Python
* Flask
* TensorFlow
* NumPy
* OpenCV
* HTML/CSS

---




### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bone-fracture-detection.git
cd bone-fracture-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📸 How It Works

1. Upload an X-ray image.
2. The image is resized to 224×224 pixels.
3. The trained TensorFlow model processes the image.
4. The system predicts:

   * 🦴 Fracture Detected
   * ✅ Normal Bone
5. Prediction confidence is displayed.

---

## 🧠 Model Information

* Framework: TensorFlow/Keras
* Input Size: 224 × 224 × 3
* Model Format: TensorFlow SavedModel
* Binary Classification:

  * Fracture
  * Normal

---

## 📈 Future Improvements

* Grad-CAM visualization for explainable AI
* Support for multiple bone types
* Better UI/UX design
* Deployment on Render/Streamlit/AWS
* PDF medical report generation

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit pull requests.


## 📄 License

This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.


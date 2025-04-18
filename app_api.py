from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fruit_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Danh sách nhãn
labels = ["apple", "banana", "chilli pepper", "corn", "cucumber",
          "grapes", "kiwi", "lemon", "mango", "orange",
          "pear", "pineapple", "pomegranate", "tomato", "watermelon"]

def preprocess_image(image):
    """Chuẩn hóa ảnh giống MobileNetV2"""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fruit prediction API (MobileNetV2-style preprocessing) is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file)
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": labels[predicted_class],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
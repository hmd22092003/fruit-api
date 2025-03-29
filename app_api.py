from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # Cho phép React/Vue truy cập API từ domain khác

# Load model đã lưu
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục chứa app_api.py
MODEL_PATH = os.path.join(BASE_DIR, "model", "fruit_model.h5")  # Đường dẫn tương đối

model = tf.keras.models.load_model(MODEL_PATH)  # Load model


# Danh sách nhãn trái cây
labels = ["apple", "banana", "chilli pepper", "corn", "cucumber",
          "grapes", "kiwi", "lemon", "mango", "orange",
          "pear", "pineapple", "pomegranate", "tomato", "watermelon"]

def preprocess_image(image):
    """Xử lý ảnh trước khi đưa vào model."""
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize ảnh về 224x224
    image = np.array(image) / 255.0   # Chuẩn hóa dữ liệu về [0,1]
    image = np.expand_dims(image, axis=0)  # Thêm batch dimension
    return image

@app.route("/", methods=["GET"])
def home():
    """Kiểm tra API có chạy hay không."""
    return jsonify({"message": "Fruit prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """Dự đoán loại trái cây từ ảnh."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    try:
        image = Image.open(file)
        image = preprocess_image(image)

        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))  # Xác suất dự đoán cao nhất

        return jsonify({
            "prediction": labels[predicted_class],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    PORT = int(os.environ.get("PORT", 5000))  # Lấy PORT từ biến môi trường
    app.run(host="0.0.0.0", port=PORT, debug=False)


from flask import Flask, request, send_file
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Tải model YOLOv8 từ file best.pt trong cùng thư mục
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

def read_image(file_storage):
    """Đọc ảnh từ file gửi lên và chuyển thành mảng numpy RGB"""
    image = Image.open(file_storage.stream).convert("RGB")
    return np.array(image)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image']
        img = read_image(file)

        # Nhận diện và vẽ bounding boxes
        results = model(img)[0]
        annotated = results.plot()  # Trả về ảnh đã vẽ box (ndarray BGR)

        # Chuyển ndarray ảnh BGR thành JPEG bytes
        _, jpeg = cv2.imencode('.jpg', annotated)

        # Trả ảnh dạng blob về client
        return send_file(
            BytesIO(jpeg.tobytes()),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='result.jpg'
        )
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/ping', methods=['GET'])
def ping():
    return "YOLOv8 backend is alive!", 200

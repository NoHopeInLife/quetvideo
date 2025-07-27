from flask import Flask, request, send_file
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

def read_image(file_storage):
    image = Image.open(file_storage.stream).convert("RGB")
    return np.array(image)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image']
        img = read_image(file)

        results = model(img)[0]
        annotated = results.plot()

        _, jpeg = cv2.imencode('.jpg', annotated)

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

# 👇 THÊM ĐOẠN NÀY
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import io
import base64

app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi từ bất kỳ domain nào

# Load YOLOv8n CPU
model = YOLO("yolov8n.pt")

def read_image_from_base64(b64_string):
    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return img

def draw_boxes(img, results):
    draw = ImageDraw.Draw(img)
    for r in results:
        for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), f"{int(cls)} {score:.2f}", fill="red")
    return img

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "Missing image"}), 400

    img = read_image_from_base64(data["image"])
    # inference giữ 320x320
    results = model.predict(np.array(img), imgsz=320, verbose=False)

    # Vẽ bounding boxes lên ảnh
    img_out = draw_boxes(img, results)

    # Chuyển ảnh thành base64 trả về
    buffer = io.BytesIO()
    img_out.save(buffer, format="JPEG")
    b64_out = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": b64_out})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

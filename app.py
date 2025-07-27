from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os
import base64

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

def read_image(file_storage):
    image = Image.open(file_storage.stream).convert("RGB")
    return np.array(image)

def convert_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image']
        img = read_image(file)

        # Run YOLO detection
        results = model(img)[0]
        
        # Get detections
        detections = []
        for box in results.boxes:
            if box.conf is not None and box.conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Convert to [x, y, width, height] format
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                detections.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "class": class_id,
                    "className": class_name
                })
        
        # Create annotated image
        annotated = results.plot()
        
        # Convert to base64
        processed_image_base64 = convert_to_base64(annotated)
        
        # Return JSON response
        response = {
            "detections": detections,
            "processed_image": processed_image_base64
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "YOLOv8 backend is alive!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000) 
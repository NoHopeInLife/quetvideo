from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os
import base64
import logging
from threading import Lock
from flask_cors import CORS

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

# Global model instance với thread lock
model = None
model_lock = Lock()

def load_model():
    """Load model một lần duy nhất"""
    global model
    if model is None:
        with model_lock:
            if model is None:
                logger.info("Loading YOLO model...")
                model = YOLO(MODEL_PATH)
                logger.info("Model loaded successfully")
    return model

def read_image(file_storage):
    image = Image.open(file_storage.stream).convert("RGB")
    return np.array(image)

def convert_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

@app.before_request
def log_request_info():
    logger.info('Request: %s %s', request.method, request.url)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Load model nếu chưa có
        model_instance = load_model()
        
        file = request.files['image']
        img = read_image(file)

        # Run YOLO detection với timeout
        logger.info("Processing image...")
        results = model_instance(img)[0]
        
        # Get detections
        detections = []
        for box in results.boxes:
            if box.conf is not None and box.conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model_instance.names[class_id]
                
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
        
        logger.info(f"Detection completed: {len(detections)} objects found")
        
        # Return JSON response
        response = {
            "detections": detections,
            "processed_image": processed_image_base64
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "YOLOv8 backend is alive!", 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        model_instance = load_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": model_instance is not None,
            "ultralytics_version": "8.3.170"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Load model khi khởi động
    load_model()
    app.run(host='0.0.0.0', port=10000, threaded=True)
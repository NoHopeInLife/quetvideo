from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
from threading import Lock
from flask_cors import CORS
import time
import torch
import gc
import psutil
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

# Global model instance với thread lock
model = None
model_lock = Lock()

# CPU optimization settings
CPU_CORES = psutil.cpu_count(logical=False)
TORCH_THREADS = min(4, CPU_CORES)
logger.info(f"CPU cores: {CPU_CORES}, Using {TORCH_THREADS} threads")

# Set torch threads
torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(1)

def load_model():
    """Load model với CPU optimizations"""
    global model
    if model is None:
        with model_lock:
            if model is None:
                logger.info("Loading YOLO model for CPU inference...")
                
                # CPU-specific optimizations
                torch.backends.cudnn.enabled = False
                torch.backends.mkldnn.enabled = True
                
                model = YOLO(MODEL_PATH)
                
                # Warm up với dummy image 320x320 (size từ frontend)
                logger.info("Warming up model...")
                dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
                model.predict(
                    dummy_img, 
                    imgsz=320, 
                    conf=0.5, 
                    device='cpu', 
                    verbose=False,
                    half=False
                )
                
                logger.info("Model loaded and optimized for CPU")
    return model

def read_image_from_frontend(file_storage):
    """Đọc ảnh đã được resize từ frontend (320x320)"""
    try:
        file_bytes = file_storage.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Cannot decode image")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Processing image of shape: {img.shape}")
        return img
        
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise

@app.route('/detect', methods=['POST'])
def detect():
    try:
        model_instance = load_model()

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # Đọc ảnh đã được resize từ frontend
        img = read_image_from_frontend(file)
        
        start = time.time()

        # YOLO inference với image size 320 (từ frontend)
        results = model_instance.predict(
            img,
            imgsz=320,
            conf=0.5,
            iou=0.5,
            device='cpu',
            verbose=False,
            stream=False,
            half=False,
            agnostic_nms=False,
            max_det=100
        )[0]

        inference_time = time.time() - start
        logger.info(f"CPU inference completed in {inference_time:.3f}s")

        # Parse results
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes_data = results.boxes
            confs = boxes_data.conf.cpu().numpy()
            xyxy = boxes_data.xyxy.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy().astype(int)
            
            # Filter by confidence
            valid_indices = confs > 0.5
            
            for i in np.where(valid_indices)[0]:
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                class_id = int(classes[i])
                class_name = model_instance.names[class_id]

                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                detections.append({
                    "bbox": bbox,
                    "confidence": round(conf, 3),
                    "class": class_id,
                    "className": class_name
                })

        # Memory cleanup
        del img
        gc.collect()

        return jsonify({
            "detections": detections,
            "inference_time": round(inference_time, 3),
            "num_detections": len(detections),
            "device": "cpu"
        })

    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "YOLOv8 CPU backend is alive!", 200

if __name__ == '__main__':
    # Load model trước
    load_model()
    
    # Get port from environment
    port = int(os.environ.get('PORT', 10000))
    
    app.run(
        host='0.0.0.0', 
        port=port,
        threaded=True,
        debug=False,
        processes=1
    )
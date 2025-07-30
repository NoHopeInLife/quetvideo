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
import time
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
import psutil

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
CPU_CORES = psutil.cpu_count(logical=False)  # Physical cores
TORCH_THREADS = min(4, CPU_CORES)  # Limit threads cho stability
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
                torch.backends.mkldnn.enabled = True  # Intel MKL-DNN optimization
                
                model = YOLO(MODEL_PATH)
                
                # Warm up với smaller dummy image cho CPU
                logger.info("Warming up model...")
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)  # Smaller for CPU
                model.predict(
                    dummy_img, 
                    imgsz=224, 
                    conf=0.5, 
                    device='cpu', 
                    verbose=False,
                    half=False  # CPU không support FP16
                )
                
                logger.info("Model loaded and optimized for CPU")
    return model

def read_image_cpu_optimized(file_storage):
    """Đọc ảnh tối ưu cho CPU processing"""
    try:
        # Đọc trực tiếp bytes
        file_bytes = file_storage.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Cannot decode image")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aggressive resizing cho CPU - nhỏ hơn để tăng tốc
        height, width = img.shape[:2]
        max_size = 640  # Smaller max size cho CPU
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            # Sử dụng INTER_LINEAR thay INTER_AREA (nhanh hơn trên CPU)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return img
        
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise

@app.before_request
def log_request_info():
    logger.info('Request: %s %s', request.method, request.url)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        model_instance = load_model()

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # CPU-optimized image reading
        img = read_image_cpu_optimized(file)
        logger.info(f"Processing image of shape: {img.shape}")
        
        start = time.time()

        # CPU-optimized YOLO inference
        results = model_instance.predict(
            img,
            imgsz=224,          # Smaller size cho CPU (224 thay 320)
            conf=0.5,           # Confidence threshold
            iou=0.5,            # Higher NMS threshold
            device='cpu',
            verbose=False,
            stream=False,
            half=False,         # CPU không support FP16
            agnostic_nms=False, # Faster NMS
            max_det=100         # Limit detections để tăng tốc
        )[0]

        inference_time = time.time() - start
        logger.info(f"CPU inference completed in {inference_time:.3f}s")

        # Fast result parsing
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            # Vectorized operations thay vòng for
            boxes_data = results.boxes
            confs = boxes_data.conf.cpu().numpy()
            xyxy = boxes_data.xyxy.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy().astype(int)
            
            # Filter by confidence in one go
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

        # Memory cleanup for CPU
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

@app.route('/detect_fast', methods=['POST'])
def detect_fast():
    """Ultra-fast mode với quality trade-off"""
    try:
        model_instance = load_model()

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        file = request.files['image']
        img = read_image_cpu_optimized(file)
        
        # Ultra aggressive resizing
        height, width = img.shape[:2]
        if max(height, width) > 320:
            scale = 320 / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)  # Fastest interpolation
        
        start = time.time()

        # Ultra-fast inference settings
        results = model_instance.predict(
            img,
            imgsz=160,          # Very small size
            conf=0.5,           # Higher confidence để ít post-processing
            iou=0.6,            # Higher IoU
            device='cpu',
            verbose=False,
            half=False,
            max_det=50,         # Limit detections
            agnostic_nms=True   # Faster NMS
        )[0]

        inference_time = time.time() - start
        
        # Minimal parsing
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                if box.conf > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "confidence": round(conf, 2),
                        "class": class_id,
                        "className": model_instance.names[class_id]
                    })

        del img
        gc.collect()

        return jsonify({
            "detections": detections,
            "inference_time": round(inference_time, 3),
            "mode": "ultra_fast"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "YOLOv8 CPU backend is alive!", 200

@app.route('/health', methods=['GET'])
def health():
    """Health check với CPU info"""
    try:
        model_instance = load_model()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            "status": "healthy",
            "model_loaded": model_instance is not None,
            "device": "cpu",
            "cpu_cores": CPU_CORES,
            "torch_threads": TORCH_THREADS,
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "available_memory": f"{memory.available / 1024**3:.1f}GB"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/benchmark', methods=['GET'])
def benchmark():
    """Benchmark CPU performance"""
    try:
        model_instance = load_model()
        
        # Create test image
        test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        times = []
        for i in range(5):  # 5 runs
            start = time.time()
            results = model_instance.predict(
                test_img,
                imgsz=224,
                conf=0.5,
                device='cpu',
                verbose=False
            )
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return jsonify({
            "avg_inference_time": round(avg_time, 3),
            "std_deviation": round(std_time, 3),
            "min_time": round(min(times), 3),
            "max_time": round(max(times), 3),
            "fps_estimate": round(1/avg_time, 1),
            "device": "cpu"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load model trước
    load_model()
    
    # CPU-optimized Flask settings
    app.run(
        host='0.0.0.0', 
        port=10000, 
        threaded=True,
        debug=False,
        processes=1  # Single process cho CPU optimization
    )
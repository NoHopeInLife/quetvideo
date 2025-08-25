from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
from threading import Lock
import os
import torch
import gc
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
model = None
model_lock = Lock()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

def load_model():
    """Load YOLOv8n pretrain FP16"""
    global model
    if model is None:
        with model_lock:
            if model is None:
                logger.info("Loading YOLOv8n FP16 model...")
                model = YOLO(MODEL_PATH)
                # Load FP16
                if DEVICE != "cpu":
                    model.model.half()  # float16 only on GPU
                # Warm-up
                dummy_img = np.zeros((320,320,3), dtype=np.uint8)
                model.predict(dummy_img, imgsz=320, device=DEVICE, verbose=False, half=(DEVICE!="cpu"))
                logger.info("Model loaded and warmed up")
    return model

def preprocess_image(file_storage):
    """Read image from POST request"""
    file_bytes = file_storage.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        model_instance = load_model()
        if 'image' not in request.files:
            return jsonify({"error":"No image file"}),400
        img = preprocess_image(request.files['image'])

        results = model_instance.predict(
            img,
            imgsz=320,
            device=DEVICE,
            conf=0.5,
            verbose=False,
            half=(DEVICE!="cpu"),
            agnostic_nms=True
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            xyxy = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            names = model_instance.names

            for i in range(len(confs)):
                class_name = names[classes[i]]
                if class_name.lower() != "person":
                    continue  # chỉ giữ class person
                x1, y1, x2, y2 = xyxy[i]
                bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                detections.append({
                    "bbox": bbox,
                    "confidence": float(confs[i]),
                    "class": int(classes[i]),
                    "className": class_name
                })

        gc.collect()
        return jsonify({"detections": detections, "num_detections": len(detections), "device": DEVICE})
    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/ping", methods=["GET"])
def ping():
    return "YOLOv8n FP16 CPU/GPU ready!",200

if __name__=="__main__":
    port = int(os.environ.get("PORT", 10000))
    load_model()
    app.run(host="0.0.0.0", port=port, debug=False)

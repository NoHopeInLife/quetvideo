from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
from threading import Lock
from flask_cors import CORS
import time
import gc
import psutil
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_int8_qdq.onnx")

# Global session instance với thread lock
session = None
input_name = None
model_lock = Lock()

# CPU optimization settings
CPU_CORES = psutil.cpu_count(logical=False)
logger.info(f"CPU cores: {CPU_CORES}")

def load_model():
    """Load mô hình ONNX INT8"""
    global session, input_name
    if session is None:
        with model_lock:
            if session is None:
                logger.info("Loading ONNX INT8 model for CPU inference...")
                session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
                input_name = session.get_inputs()[0].name
                logger.info("Model loaded and ready.")
    return session, input_name

def read_image_from_frontend(file_storage):
    try:
        file_bytes = file_storage.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
        logger.info(f"Processing image of shape: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise

def non_max_suppression(boxes, scores, conf_thres=0.5, iou_thres=0.4):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(), scores=scores.tolist(),
        score_threshold=conf_thres, nms_threshold=iou_thres
    )
    return indices.flatten() if len(indices) else []

@app.route('/detect', methods=['POST'])
def detect():
    try:
        session, input_name = load_model()

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        img = read_image_from_frontend(file)

        # Đảm bảo ảnh đã resize sẵn (ví dụ 320x320) từ frontend
        if img.shape[0] != 320 or img.shape[1] != 320:
            logger.warning(f"Unexpected image shape: {img.shape}. Expected (320, 320).")

        img_input = img.transpose(2, 0, 1) / 255.0
        img_input = img_input[np.newaxis, ...].astype(np.float32)

        start = time.time()
        outputs = session.run(None, {input_name: img_input})[0]  # (1, N, 6)
        preds = outputs[0]
        inference_time = time.time() - start
        logger.info(f"ONNX INT8 inference completed in {inference_time:.3f}s")

        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(int)

        # xywh to xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        results = []
        keep = non_max_suppression(xyxy, scores)
        for i in keep:
            x1, y1, x2, y2 = xyxy[i].astype(int)
            results.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "confidence": float(scores[i]),
                "class": int(class_ids[i]),
                "className": str(class_ids[i])
            })

        del img, img_input
        gc.collect()

        return jsonify({
            "detections": results,
            "inference_time": round(inference_time, 3),
            "num_detections": len(results),
            "device": "cpu"
        })

    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "YOLOv8 INT8 ONNX backend is alive!", 200

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False, processes=1)

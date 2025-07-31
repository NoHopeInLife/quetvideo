from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
from threading import Lock
from flask_cors import CORS
import time
import gc
import psutil
import os
import logging

# ==================== CẤU HÌNH ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

torch.set_num_threads(1)  # Giới hạn CPU threads

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.torchscript")
model = None
model_lock = Lock()

CPU_CORES = psutil.cpu_count(logical=False)
logger.info(f"CPU cores (physical): {CPU_CORES}")

# ==================== LOAD MÔ HÌNH ====================
def load_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                logger.info(f"Loading TorchScript model from {MODEL_PATH}...")
                model = torch.jit.load(MODEL_PATH, map_location="cpu")
                model.eval()
                logger.info("Model loaded successfully.")
    return model

# ==================== ĐỌC ẢNH ====================
def read_image_from_frontend(file_storage):
    start = time.time()
    file_bytes = file_storage.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    logger.info(f"Image read in {time.time() - start:.3f}s, shape: {img.shape}")
    return img

# ==================== NMS ====================
def non_max_suppression(boxes, scores, conf_thres=0.5, iou_thres=0.4):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(), scores=scores.tolist(),
        score_threshold=conf_thres, nms_threshold=iou_thres
    )
    return indices.flatten() if len(indices) else []

# ==================== API ====================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        overall_start = time.time()
        model = load_model()

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        img = read_image_from_frontend(file)
        if img.shape[:2] != (320, 320):
            logger.warning(f"Warning: Unexpected image size: {img.shape}, expected (320, 320)")

        step_start = time.time()
        img_input = img.transpose(2, 0, 1) / 255.0  # (3, H, W)
        img_input = img_input[np.newaxis, ...].astype(np.float32)
        img_tensor = torch.from_numpy(img_input)
        logger.info(f"Tensor preprocessing took {time.time() - step_start:.3f}s")

        step_start = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)[0]  # (1, 5, 8400)
        logger.info(f"Inference time: {time.time() - step_start:.3f}s")

        step_start = time.time()
        preds = outputs.squeeze(0).permute(1, 0).numpy()  # (8400, 5)
        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = np.zeros_like(scores, dtype=int)

        # Chuyển xywh → xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        keep = non_max_suppression(xyxy, scores, conf_thres=0.5, iou_thres=0.4)

        results = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i].astype(int)
            results.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "confidence": float(scores[i]),
                "class": int(class_ids[i]),
                "className": str(class_ids[i])
            })

        logger.info(f"Post-processing time: {time.time() - step_start:.3f}s")

        del img, img_input, img_tensor, outputs
        gc.collect()

        total = time.time() - overall_start
        logger.info(f"=== Total detection time: {total:.3f}s ===")

        return jsonify({
            "detections": results,
            "inference_time": round(total, 3),
            "num_detections": len(results),
            "device": "cpu"
        })

    except Exception as e:
        logger.exception("Error in detection")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "YOLOv8 TorchScript backend is alive!", 200

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=False, debug=False, processes=1)

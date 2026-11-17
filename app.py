import os
import io
import time
import base64
import threading
import webbrowser
import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

from evaluate import Detector
from gradcam import GradCAM

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global detector instances (Dual-Expert Ensemble: SD-v1.4 + ProGAN)
print("Initializing CO-SPY Dual-Expert Detectors...")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
detector_sd = None
detector_progan = None

def get_detectors():
    global detector_sd, detector_progan
    if detector_sd is None:
        print(f"Loading CO-SPY SD-v1.4 Detector (Diffusion Expert) on {DEVICE}...")
        use_ckpt = "ckpt_improved" if os.path.exists("ckpt_improved/sd-v1_4") else "ckpt"
        is_pretrain = not os.path.exists(os.path.join(use_ckpt, "sd-v1_4", "fusion_attention")) and not os.path.exists(os.path.join(use_ckpt, "sd-v1_4", "fusion-attention"))
        try:
            detector_sd = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="sd-v1_4",
                fusion_type="attention",
                ckpt=use_ckpt,
                pretrain=is_pretrain,
                use_freq=True,
                gradcam=True
            )
            print("SD-v1.4 Detector loaded successfully!")
        except Exception as e:
            print(f"Fallback loading pretrain for SD-v1.4: {e}")
            detector_sd = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="sd-v1_4",
                fusion_type="attention",
                pretrain=True,
                use_freq=True,
                gradcam=True
            )

    if detector_progan is None:
        print(f"Loading CO-SPY ProGAN Detector (GAN Expert) on {DEVICE}...")
        use_ckpt = "ckpt_improved" if os.path.exists("ckpt_improved/progan") else "ckpt"
        is_pretrain = not os.path.exists(os.path.join(use_ckpt, "progan", "fusion_attention")) and not os.path.exists(os.path.join(use_ckpt, "progan", "fusion-attention"))
        try:
            detector_progan = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="progan",
                fusion_type="attention",
                ckpt=use_ckpt,
                pretrain=is_pretrain,
                gradcam=True
            )
            print("ProGAN Detector loaded successfully!")
        except Exception as e:
            print(f"Fallback loading pretrain for ProGAN: {e}")
            detector_progan = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="progan",
                fusion_type="attention",
                pretrain=True,
                gradcam=True
            )

    return detector_sd, detector_progan

def image_to_base64(img_pil, format="JPEG", quality=90):
    buffered = io.BytesIO()
    if format.upper() == "JPEG" and img_pil.mode in ("RGBA", "P"):
        img_pil = img_pil.convert("RGB")
    img_pil.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples_dir = os.path.join(app.static_folder, "samples")
    samples = []
    if os.path.exists(samples_dir):
        catalog = [
            {
                "id": "sample_ai_portrait.jpg",
                "name": "AI Portrait (Hyper-Realistic)",
                "type": "AI-Generated",
                "tag": "Portrait / Diffusion",
                "file": "samples/sample_ai_portrait.jpg"
            },
            {
                "id": "sample_real_photo.jpg",
                "name": "Real Photograph (DSLR)",
                "type": "Real Camera",
                "tag": "Natural Photo",
                "file": "samples/sample_real_photo.jpg"
            },
            {
                "id": "sample_ai_scene.png",
                "name": "AI Interior Scene",
                "type": "AI-Generated",
                "tag": "Synthetic Scene",
                "file": "samples/sample_ai_scene.png"
            }
        ]
        for item in catalog:
            if os.path.exists(os.path.join(app.static_folder, item["file"])):
                samples.append(item)
    return jsonify({"samples": samples})

@app.route("/api/scan", methods=["POST"])
def scan_image():
    t_start = time.time()
    pil_img = None

    # Check uploaded file
    if "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            pil_img = Image.open(file.stream).convert("RGB")
    
    # Check sample selection
    if pil_img is None and request.is_json:
        data = request.get_json()
        sample_path = data.get("sample_path")
        if sample_path:
            full_path = os.path.join(app.static_folder, sample_path)
            if os.path.exists(full_path):
                pil_img = Image.open(full_path).convert("RGB")
    
    # Check base64 input
    if pil_img is None and request.is_json:
        data = request.get_json()
        b64_data = data.get("image_base64")
        if b64_data:
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if pil_img is None:
        return jsonify({"error": "No valid image provided"}), 400

    det_sd, det_progan = get_detectors()

    # Preprocessing for both detectors
    tensor_sd = det_sd.model.test_transform(pil_img).unsqueeze(0).to(det_sd.device)
    tensor_progan = det_progan.model.test_transform(pil_img).unsqueeze(0).to(det_progan.device)

    # 1. Predictions from both experts
    with torch.no_grad():
        score_sd = det_sd.model.predict(tensor_sd)[0]
        score_progan = det_progan.model.predict(tensor_progan)[0]

        # Branch scores for SD detector
        sem_sd = det_sd.model.sem(det_sd.model.sem_transform(tensor_sd)).sigmoid().item()
        art_sd = det_sd.model.art(det_sd.model.art_transform(tensor_sd)).sigmoid().item()

        # Branch scores for ProGAN detector
        sem_progan = det_progan.model.sem(det_progan.model.sem_transform(tensor_progan)).sigmoid().item()
        art_progan = det_progan.model.art(det_progan.model.art_transform(tensor_progan)).sigmoid().item()

    # 2. Dual-Expert MAX Ensemble
    overall_score = max(score_sd, score_progan)
    
    # Pick the winning expert for Grad-CAM explanation
    if score_sd >= score_progan:
        winning_det = det_sd
        winning_tensor = tensor_sd
        winning_expert = "Diffusion Specialist (SD-v1.4)"
        pred_sem = sem_sd
        pred_art = art_sd
    else:
        winning_det = det_progan
        winning_tensor = tensor_progan
        winning_expert = "GAN Specialist (ProGAN)"
        pred_sem = sem_progan
        pred_art = art_progan

    # 3. Grad-CAM from winning expert
    cam_gen = GradCAM(winning_det.model)
    cam_mask, _ = cam_gen.generate_cam(winning_tensor)

    # 4. Create Heatmap Overlays
    w, h = pil_img.size
    overlay_img = GradCAM.overlay_heatmap(pil_img, cam_mask, alpha=0.5)

    # Pure heatmap for separate inspection
    cam_resized = cv2.resize(cam_mask, (w, h))
    cam_smooth = cv2.GaussianBlur(cam_resized, (15, 15), 0)
    heatmap_pure_cv = cv2.applyColorMap(np.uint8(255 * cam_smooth), cv2.COLORMAP_JET)
    heatmap_pure_cv = cv2.cvtColor(heatmap_pure_cv, cv2.COLOR_BGR2RGB)
    heatmap_pure_pil = Image.fromarray(heatmap_pure_cv)

    t_end = time.time()
    inference_time = round((t_end - t_start) * 1000, 1)

    is_ai = overall_score > 0.5
    verdict = "AI-Generated" if is_ai else "Real Image"
    confidence = overall_score * 100 if is_ai else (1.0 - overall_score) * 100

    return jsonify({
        "success": True,
        "is_ai": is_ai,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "overall_score": round(overall_score, 4),
        "score_sd": round(score_sd * 100, 2),
        "score_progan": round(score_progan * 100, 2),
        "winning_expert": winning_expert,
        "sem_score": round(pred_sem * 100, 2),
        "art_score": round(pred_art * 100, 2),
        "inference_time_ms": inference_time,
        "image_size": f"{w} x {h}",
        "device": det_sd.device,
        "original_image": image_to_base64(pil_img, format="JPEG", quality=85),
        "overlay_image": image_to_base64(overlay_img, format="JPEG", quality=85),
        "heatmap_image": image_to_base64(heatmap_pure_pil, format="JPEG", quality=85),
    })

def start_server(port=5000, auto_open=True):
    # Pre-warm models in background
    threading.Thread(target=get_detectors, daemon=True).start()
    if auto_open:
        def open_browser():
            time.sleep(1.5)
            print(f"\nOpening Web App at http://127.0.0.1:{port} ...\n")
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("CO-SPY Web Application")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()
    start_server(port=args.port, auto_open=not args.no_browser)

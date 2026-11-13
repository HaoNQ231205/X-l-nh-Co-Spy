"""Web Demo Interface for CO-SPY Deepfake Detection using Gradio.

Plug-and-Play AI Image Detector:
- Tự động quét và liệt kê toàn bộ các Model trong Google Drive và máy cục bộ.
- Cho phép người dùng CHỈ VIỆC THẢ FILE MODEL (.pth) trực tiếp lên Web hoặc chọn từ danh sách.
- Dự đoán tức thì ảnh Thật vs AI kèm thanh phần trăm độ tự tin.
- Hiển thị trực quan bản đồ nhiệt Dual-Branch CAM (Semantic & Artifact).
- Tự động tạo link Public chia sẻ trên Google Colab (share=True).
"""

import os
import sys
import glob
import shutil
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
import torch
import numpy as np

# Ensure project root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cospy_infer.engine import InferenceEngine
from cospy_infer.jpeg_probe import probe_jpeg

# Global engine cache to avoid reloading models on every click
_ENGINE_CACHE: Dict[str, InferenceEngine] = {}


def scan_available_models() -> Dict[str, str]:
    """Tự động quét toàn bộ các thư mục để tìm model đã huấn luyện và model nền tảng."""
    models_dict = {}

    # 1. Model vừa train trên Google Drive (Writing)
    drive_writing_dir = "/content/drive/MyDrive/CO-SPY/trained_models/writing"
    if os.path.exists(drive_writing_dir):
        for fn in ["best_model.pth", "fusion_weights.pth", "train_checkpoint.pth"]:
            if os.path.exists(os.path.join(drive_writing_dir, fn)):
                models_dict["🎯 Model Writing (Đã train trên Google Drive)"] = drive_writing_dir
                break

    # 2. Quét các thư mục con khác trong trained_models trên Drive
    drive_parent = "/content/drive/MyDrive/CO-SPY/trained_models"
    if os.path.exists(drive_parent):
        for sub in os.listdir(drive_parent):
            sub_path = os.path.join(drive_parent, sub)
            if os.path.isdir(sub_path) and sub != "writing":
                for fn in ["best_model.pth", "fusion_weights.pth", "train_checkpoint.pth"]:
                    if os.path.exists(os.path.join(sub_path, fn)):
                        models_dict[f"📁 Drive Model: {sub}"] = sub_path
                        break

    # 3. Quét thư mục trained_models cục bộ
    local_trained = os.path.join(SCRIPT_DIR, "trained_models")
    if os.path.exists(local_trained):
        for root, dirs, files in os.walk(local_trained):
            for fn in ["best_model.pth", "fusion_weights.pth", "train_checkpoint.pth"]:
                if fn in files:
                    rel = os.path.relpath(root, SCRIPT_DIR)
                    models_dict[f"💻 Local Model: {rel}"] = root
                    break

    # 4. Thư mục models/ chuyên dụng (người dùng thả file vào đây)
    custom_models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(custom_models_dir, exist_ok=True)
    for f in os.listdir(custom_models_dir):
        fp = os.path.join(custom_models_dir, f)
        if f.endswith(".pth") or f.endswith(".pt"):
            models_dict[f"📦 File Model: {f}"] = fp
        elif os.path.isdir(fp):
            models_dict[f"📦 Thư mục Model: {f}"] = fp

    # 5. Model Pretrained mặc định gốc
    models_dict["⚡ Model Pretrained Gốc (sd-v1_4)"] = "DEFAULT_PRETRAINED"

    return models_dict


def get_engine_for_path(weights_path_or_dir: Optional[str]) -> InferenceEngine:
    """Nạp engine tối ưu với bộ nhớ đệm (cache)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = weights_path_or_dir if weights_path_or_dir else "DEFAULT_PRETRAINED"

    if cache_key not in _ENGINE_CACHE:
        print(f"🚀 [CO-SPY] Đang nạp mô hình từ: {cache_key} trên thiết bị [{device.upper()}]...")

        # Nếu là đường dẫn tới 1 file .pth đơn lẻ (chẳng hạn người dùng upload lên)
        if weights_path_or_dir and os.path.isfile(weights_path_or_dir):
            temp_dir = os.path.join(SCRIPT_DIR, "models", "temp_loaded")
            os.makedirs(temp_dir, exist_ok=True)
            # Copy file vào temp_dir với tên best_model.pth
            shutil.copyfile(weights_path_or_dir, os.path.join(temp_dir, "best_model.pth"))
            weights_dir = temp_dir
        elif weights_path_or_dir and weights_path_or_dir != "DEFAULT_PRETRAINED":
            weights_dir = weights_path_or_dir
        else:
            weights_dir = None

        engine = InferenceEngine(
            train_dataset="sd-v1_4",
            mode="fusion",
            pretrain=True,
            device=device,
            route="branch_native",
            views=1,
            weights_dir=weights_dir
        )
        _ENGINE_CACHE[cache_key] = engine

    return _ENGINE_CACHE[cache_key]


def predict_pipeline(
    input_image: Image.Image,
    model_selection: str,
    uploaded_model_file: Any,
    enable_cam: bool
):
    """Xử lý dự đoán ảnh cho giao diện Gradio."""
    if input_image is None:
        return "⚠️ Vui lòng tải một bức ảnh lên!", {}, "", None, None

    # 1. Xác định file model cần dùng
    models_dict = scan_available_models()
    model_source_info = ""

    # Ưu tiên 1: Người dùng kéo thả trực tiếp file .pth lên Web
    if uploaded_model_file is not None:
        target_path = uploaded_model_file.name if hasattr(uploaded_model_file, "name") else str(uploaded_model_file)
        weights_target = target_path
        model_source_info = f"📥 Đã nạp trực tiếp từ file tải lên Web: `{os.path.basename(target_path)}`"
    # Ưu tiên 2: Chọn từ danh sách quét tự động
    elif model_selection in models_dict:
        weights_target = models_dict[model_selection]
        model_source_info = f"📂 Đang dùng từ danh sách: `{model_selection}`"
    else:
        weights_target = None
        model_source_info = "⚡ Đang dùng Model Pretrained sd-v1_4 mặc định."

    try:
        engine = get_engine_for_path(weights_target)
    except Exception as e:
        return f"❌ Lỗi nạp mô hình: {e}", {}, "", None, None

    # 2. Dự đoán nhãn
    pred = engine.predict_image(input_image)

    prob_ai = pred["prob_fusion"]
    prob_real = 1.0 - prob_ai
    label = "🚨 ẢNH DO AI TẠO (AI-GENERATED)" if prob_ai > 0.5 else "📸 ẢNH CHỤP THẬT (REAL PHOTO)"

    confidences = {
        "AI-Generated (Giả mạo)": float(prob_ai),
        "Real Photo (Thật)": float(prob_real)
    }

    # 3. Bảng phân tích chi tiết
    breakdown_md = f"""
### 📊 THÔNG SỐ PHÂN TÍCH KỸ THUẬT:
* **Kết luận**: **{label}**
* **Xác suất Tổng hợp (Fusion)**: **{prob_ai * 100:.2f}% AI** | **{prob_real * 100:.2f}% Real**
* **Nhánh Semantic (Ngữ nghĩa SigLIP ViT)**: {pred['prob_sem'] * 100:.2f}% AI (Logit: `{pred['logit_sem']:.3f}`)
* **Nhánh Artifact (Vết nhiễu SD VAE)**: {pred['prob_art'] * 100:.2f}% AI (Logit: `{pred['logit_art']:.3f}`)
* **Đường xử lý hình học**: `{pred['route']}`
* **Model đang kích hoạt**: {model_source_info}
"""

    sem_cam_img = None
    art_cam_img = None

    # 4. Hiển thị Heatmap CAM nếu bật
    if enable_cam:
        try:
            from explain.gradcam_semantic import SemanticGradCAM
            from explain.cam_artifact import ArtifactCAM
            from explain.render import overlay_heatmap_on_image

            # Semantic CAM (soi dị tật ngữ nghĩa)
            sem_cam = SemanticGradCAM(engine.model.sem)
            sem_map = sem_cam.compute_cam(input_image)
            sem_cam_img = overlay_heatmap_on_image(input_image, sem_map, colormap="jet", alpha=0.5)

            # Artifact CAM (soi vết nhiễu latent AI)
            art_cam = ArtifactCAM(engine.model.art)
            art_map = art_cam.compute_cam(input_image)
            art_cam_img = overlay_heatmap_on_image(input_image, art_map, colormap="magma", alpha=0.5)
        except Exception as cam_err:
            breakdown_md += f"\n*(Lưu ý: Không tạo được ảnh nhiệt CAM: {cam_err})*"

    return label, confidences, breakdown_md, sem_cam_img, art_cam_img


def build_app():
    """Khởi tạo giao diện ứng dụng Gradio."""
    import gradio as gr

    # Quét danh sách model ban đầu
    initial_models = scan_available_models()
    model_choices = list(initial_models.keys())
    default_choice = model_choices[0] if model_choices else "⚡ Model Pretrained Gốc (sd-v1_4)"

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )

    with gr.Blocks(title="CO-SPY AI Image Detector", theme=theme) as demo:
        gr.Markdown(
            """
            # 🕵️‍♂️ CO-SPY: HỆ THỐNG PHÁT HIỆN ẢNH DO AI TẠO
            ### Plug-and-Play Detector: Tự động nạp Model đã train từ Google Drive hoặc Tải file `.pth` lên trực tiếp.
            """
        )

        with gr.Row():
            # CỘT TRÁI: ĐẦU VÀO & CẤU HÌNH MODEL
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📥 1. Tải ảnh cần kiểm tra lên (JPG, PNG, WEBP...)")

                with gr.Accordion("⚙️ Cấu hình Model (Tự động hoặc Tải file lên)", open=True):
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=default_choice,
                        label="🎯 Chọn Model có sẵn (Tự động quét từ Google Drive & Cục bộ)"
                    )
                    refresh_models_btn = gr.Button("🔄 Quét lại danh sách Model", size="sm")

                    uploaded_model_file = gr.File(
                        label="📤 HOẶC Kéo thả file model (.pth) trực tiếp vào đây",
                        file_types=[".pth", ".pt"]
                    )

                enable_cam = gr.Checkbox(
                    label="🔍 Bật ảnh nhiệt trực quan hóa (Dual-Branch CAM)",
                    value=False,
                    info="Hiển thị vùng bất thường ngữ nghĩa và vết nhiễu tần số cao của AI"
                )

                submit_btn = gr.Button("⚡ PHÂN TÍCH ẢNH NGAY", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Xóa làm lại", variant="secondary")

            # CỘT PHẢI: KẾT QUẢ DỰ ĐOÁN
            with gr.Column(scale=1):
                result_label = gr.Textbox(label="🎯 KẾT LUẬN CỦA CO-SPY", interactive=False, text_align="center")
                confidence_bar = gr.Label(label="📈 Tỷ lệ phần trăm dự đoán", num_top_classes=2)
                details_box = gr.Markdown(label="Chi tiết kỹ thuật")

        # KHU VỰC HIỂN THỊ CAM HEATMAP
        with gr.Row():
            with gr.Column():
                sem_cam_output = gr.Image(label="🧠 Semantic CAM (Soi dị tật Ngữ nghĩa - SigLIP ViT)", interactive=False)
            with gr.Column():
                art_cam_output = gr.Image(label="🔬 Artifact CAM (Soi vết nhiễu Latent - SD VAE ResNet)", interactive=False)

        # HÀNH ĐỘNG BẤM NÚT
        submit_btn.click(
            fn=predict_pipeline,
            inputs=[image_input, model_dropdown, uploaded_model_file, enable_cam],
            outputs=[result_label, confidence_bar, details_box, sem_cam_output, art_cam_output]
        )

        clear_btn.click(
            fn=lambda: (None, None, "", {}, "", None, None),
            inputs=[],
            outputs=[image_input, uploaded_model_file, result_label, confidence_bar, details_box, sem_cam_output, art_cam_output]
        )

        def update_models_list():
            scanned = scan_available_models()
            keys = list(scanned.keys())
            return gr.Dropdown(choices=keys, value=keys[0] if keys else None)

        refresh_models_btn.click(
            fn=update_models_list,
            inputs=[],
            outputs=[model_dropdown]
        )

        gr.Markdown(
            """
            ---
            *Hệ thống phát hiện AI Deepfake CO-SPY v2 - Hỗ trợ nạp model linh hoạt và giải thích trực quan.*
            """
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Tạo public link trên Colab (share=True)")
    parser.add_argument("--port", type=int, default=7860, help="Cổng chạy server")
    args = parser.parse_args()

    app = build_app()
    is_colab = os.path.exists("/content")
    share_mode = True if is_colab or args.share else False

    print(f"🌐 Đang khởi động Web App (Share mode: {share_mode})...")
    app.launch(share=share_mode, server_port=args.port, inbrowser=not is_colab)

# Cải Tiến CO-SPY: Nâng Cấp Kiến Trúc Phát Hiện Ảnh AI

## Bối cảnh
CO-SPY (CVPR'25) sử dụng 2 nhánh: **Semantic** (SigLIP CLIP) và **Artifact** (VAE Reconstruction Error + ResNet), kết hợp qua phương pháp đơn giản (concat + Linear). Bản cải tiến này nâng cấp **4 thành phần chính** mà không phá vỡ kiến trúc gốc.

> [!IMPORTANT]
> Tất cả cải tiến đều là **opt-in** thông qua tham số dòng lệnh mới. Mã gốc vẫn hoạt động bình thường khi không sử dụng các tham số mới.

## Tổng Quan 4 Cải Tiến

| # | Cải tiến | Mô tả | File ảnh hưởng |
|---|---------|-------|----------------|
| 1 | **Cross-Attention Fusion** | Thay thế `Linear(2,1)` bằng module Cross-Attention cho phép 2 nhánh tương tác qua lại | `attention_fusion.py` (NEW) |
| 2 | **Frequency Branch (FFT)** | Thêm nhánh phân tích tần số từ sai số tái tạo VAE, ghép vào Artifact | `artifact.py` (MODIFY) |
| 3 | **Focal Loss** | Tập trung vào mẫu khó (ảnh AI chất lượng cao), giảm trọng số mẫu dễ | `utils.py` (MODIFY) |
| 4 | **Grad-CAM Visualization** | Tạo heatmap giải thích vùng nào trong ảnh khiến mô hình phát hiện AI | `gradcam.py` (NEW) |

---

## Proposed Changes

### 1. Cross-Attention Fusion Module

Thay vì ghép đặc trưng đơn giản `torch.cat([pred_sem, pred_art]) → Linear(2,1)`, module mới sử dụng **Multi-Head Cross-Attention** để 2 nhánh "hỏi" lẫn nhau:
- Semantic features (Query) ↔ Artifact features (Key, Value)
- Artifact features (Query) ↔ Semantic features (Key, Value)
- Output: `Gated fusion → MLP classifier`

#### [NEW] [attention_fusion.py](file:///d:/Documents/XLA/Co-Spy/detectors/sd-v1_4/attention_fusion.py)
- `CrossAttentionFusionModule`: Module cross-attention 2 chiều + gating mechanism
- `CoSpyAttentionFusionDetector`: Detector sử dụng attention fusion, kế thừa `BaseDetector`
- Cấu trúc: `sem_feat (1152-d)` → project → cross-attn ← project ← `art_feat (512-d)` → gate → MLP(256→1)

#### [MODIFY] [__init__.py](file:///d:/Documents/XLA/Co-Spy/detectors/sd-v1_4/__init__.py)
- Thêm import `CoSpyAttentionFusionDetector`

---

### 2. Frequency Branch (FFT Features)

Bổ sung biến đổi FFT trên sai số tái tạo `x - VAE_recon(x)` để bắt các vết mờ tuần hoàn ở miền tần số.

#### [MODIFY] [artifact.py](file:///d:/Documents/XLA/Co-Spy/detectors/sd-v1_4/artifact.py)
- Thêm class `FrequencyBranch`: Tính FFT 2D của sai số → log-amplitude spectrum → Conv encoder nhỏ
- Cập nhật `VAEReconEncoder`: Thêm tùy chọn `use_freq=True` để ghép thêm đặc trưng tần số (concat)
- Output dim tăng từ `512` lên `512 + 128 = 640` khi bật `use_freq`
- Tham số `--use_freq` trong `main.py` để bật/tắt

---

### 3. Focal Loss

Focal Loss: $FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$ giúp tập trung vào mẫu khó phân loại.

#### [MODIFY] [utils.py](file:///d:/Documents/XLA/Co-Spy/utils.py)
- Thêm class `FocalBCEWithLogitsLoss(gamma=2.0, alpha=0.25)`

---

### 4. Grad-CAM Visualization

Module tạo heatmap giải thích từ nhánh Artifact (ResNet layers).

#### [NEW] [gradcam.py](file:///d:/Documents/XLA/Co-Spy/gradcam.py)
- Class `GradCAM`: Hook vào layer cuối của artifact encoder
- Function `generate_heatmap(model, image) → overlay_image`: Tạo ảnh chồng heatmap
- Tích hợp vào phase `test` qua tham số `--gradcam`

---

### 5. Tích hợp vào Pipeline

#### [MODIFY] [main.py](file:///d:/Documents/XLA/Co-Spy/main.py)
- Thêm các argument mới:
  - `--fusion_type` (`linear` | `attention`): Chọn kiểu fusion
  - `--use_freq`: Bật nhánh frequency trong artifact
  - `--loss_type` (`bce` | `focal`): Chọn hàm loss
  - `--focal_gamma`, `--focal_alpha`: Tham số Focal Loss
  - `--gradcam`: Bật Grad-CAM khi test

#### [MODIFY] [train.py](file:///d:/Documents/XLA/Co-Spy/train.py)
- Hỗ trợ `fusion_type="attention"` → load `CoSpyAttentionFusionDetector`
- Hỗ trợ `loss_type="focal"` → dùng `FocalBCEWithLogitsLoss`
- Truyền `use_freq` khi khởi tạo `ArtifactDetector`

#### [MODIFY] [evaluate.py](file:///d:/Documents/XLA/Co-Spy/evaluate.py)
- Hỗ trợ `fusion_type="attention"` khi load model
- Tích hợp `--gradcam` vào phase `test`

---

## Verification Plan

### Automated Tests
```bash
# Kiểm tra import không lỗi
python -c "from detectors.sd-v1_4 import CoSpyAttentionFusionDetector; print('OK')"

# Kiểm tra forward pass của attention fusion
python -c "
import torch
from detectors import sd_v1_4
model = sd_v1_4.CoSpyAttentionFusionDetector.__new__(sd_v1_4.CoSpyAttentionFusionDetector)
print('Import OK')
"

# Kiểm tra Focal Loss
python -c "
from utils import FocalBCEWithLogitsLoss
import torch
loss_fn = FocalBCEWithLogitsLoss()
pred = torch.randn(4, 1)
target = torch.randint(0, 2, (4, 1)).float()
loss = loss_fn(pred, target)
print(f'Focal Loss: {loss.item():.4f}')
"

# Kiểm tra Grad-CAM
python -c "from gradcam import GradCAM; print('GradCAM import OK')"
```

### Manual Verification
- Kiểm tra `python main.py --help` hiển thị đầy đủ tham số mới
- Chạy thử training với `--fusion_type attention --loss_type focal` (cần dataset)

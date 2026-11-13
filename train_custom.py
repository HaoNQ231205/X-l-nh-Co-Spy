"""Custom domain training & 3-stage fine-tuning tool for CO-SPY (train_custom.py).

Supports the 3 sequential stages of CO-SPY on custom datasets (e.g. Dataset/data/train/writing):
  1. mode='artifact' : Fine-tunes ArtifactDetector (ResNet + art.fc; keeps SD-VAE frozen).
                       Saves best weights to <save_dir>/artifact_weights.pth.
  2. mode='semantic' : Fine-tunes SemanticDetector (sem.fc; keeps SigLIP ViT frozen).
                       Saves best weights to <save_dir>/semantic_weights.pth.
  3. mode='fusion'   : Loads <save_dir>/artifact_weights.pth & <save_dir>/semantic_weights.pth
                       if present (otherwise pretrained), freezes both, and trains Linear(2, 1) fc.
                       Fully backward-compatible with existing train_checkpoint.pth!

Also supports mode='all_3_stages' (via train_three_stage_pipeline) to run Artifact -> Semantic -> Fusion.
"""

import os
import sys
import json
import shutil
import random
import tempfile
import argparse
import importlib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Ensure project root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import evaluate

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def is_image_file(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in VALID_EXTS


def extract_label_from_path(filepath: str) -> Optional[int]:
    """Detect ground truth label: 0=Real, 1=Fake/AI."""
    parts = filepath.lower().replace("\\", "/").split("/")
    for p in parts:
        if "0_real" in p or p in ("real", "mscoco", "mscoco2017") or p.endswith("real_image_examples"):
            return 0
        if (
            "1_fake" in p
            or p in ("fake", "synthetic", "ai", "generated")
            or "stable-diffusion" in p
            or "stable_diffusion" in p
        ):
            return 1
    fname = os.path.basename(filepath).lower()
    if "_real" in fname or fname.startswith("real_"):
        return 0
    if "_fake" in fname or fname.startswith("fake_"):
        return 1
    return None


class SingleBranchDataset(Dataset):
    """Dataset providing a single branch transform (faster for standalone artifact/semantic training)."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384), (128, 128, 128))
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return x, torch.tensor(label, dtype=torch.float32)


class DualBranchDataset(Dataset):
    """Custom dataset providing both semantic and artifact branch transforms."""

    def __init__(self, samples: List[Tuple[str, int]], sem_transform=None, art_transform=None):
        self.samples = samples
        self.sem_transform = sem_transform
        self.art_transform = art_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384), (128, 128, 128))

        sem_t = self.sem_transform(img) if self.sem_transform else transforms.ToTensor()(img)
        art_t = self.art_transform(img) if self.art_transform else transforms.ToTensor()(img)
        return sem_t, art_t, torch.tensor(label, dtype=torch.float32)


def scan_folder_for_samples(folder: str) -> List[Tuple[str, int]]:
    """Scan folder recursively and return list of (filepath, label)."""
    samples = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            if is_image_file(f):
                full_path = os.path.join(root, f)
                lbl = extract_label_from_path(full_path)
                if lbl is not None:
                    samples.append((full_path, lbl))
    return samples


def load_or_split_data(
    data_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Scan dataset directory, honoring train/val subdirectories if present, or auto-splitting 80/20."""
    train_sub = os.path.join(data_dir, "train")
    val_sub = os.path.join(data_dir, "val")

    if os.path.exists(train_sub) and os.path.exists(val_sub):
        train_samples = scan_folder_for_samples(train_sub)
        val_samples = scan_folder_for_samples(val_sub)
        if len(train_samples) > 0 and len(val_samples) > 0:
            return train_samples, val_samples

    all_samples = scan_folder_for_samples(data_dir)
    if not all_samples:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ (hoặc không nhận diện được nhãn real/fake) trong: {data_dir}")

    rng = random.Random(seed)
    real_samples = [s for s in all_samples if s[1] == 0]
    fake_samples = [s for s in all_samples if s[1] == 1]

    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    n_val_real = min(len(real_samples) - 1, max(1, int(len(real_samples) * val_split))) if len(real_samples) >= 2 else 0
    n_val_fake = min(len(fake_samples) - 1, max(1, int(len(fake_samples) * val_split))) if len(fake_samples) >= 2 else 0

    val_samples = real_samples[:n_val_real] + fake_samples[:n_val_fake]
    train_samples = real_samples[n_val_real:] + fake_samples[n_val_fake:]

    if not val_samples:
        val_samples = list(train_samples)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _stratified_subsample(samples: List[Tuple[str, int]], max_count: int, seed: int = 42) -> List[Tuple[str, int]]:
    """Sample up to max_count items while keeping Real/Fake balance."""
    if max_count <= 0 or not samples:
        return []
    if len(samples) <= max_count:
        return list(samples)
    rng = random.Random(seed)
    reals = [s for s in samples if s[1] == 0]
    fakes = [s for s in samples if s[1] == 1]
    rng.shuffle(reals)
    rng.shuffle(fakes)
    if reals and fakes:
        half = max_count // 2
        taken_real = reals[:half]
        taken_fake = fakes[: max_count - len(taken_real)]
        combined = taken_real + taken_fake
        if len(combined) < max_count:
            rem = reals[len(taken_real):] + fakes[len(taken_fake):]
            combined.extend(rem[: max_count - len(combined)])
        rng.shuffle(combined)
        return combined
    combined = list(samples)
    rng.shuffle(combined)
    return combined[:max_count]


def _prepare_train_val_samples(
    data_dir: str,
    replay_dir: Optional[str] = None,
    replay_ratio: float = 1.0,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], int, int]:
    """Load primary data and optionally mix balanced replay samples."""
    primary_train, primary_val = load_or_split_data(data_dir, val_split=0.2)
    train_samples = list(primary_train)
    val_samples = list(primary_val)

    replay_train_cnt = 0
    replay_val_cnt = 0
    if replay_dir and os.path.exists(replay_dir):
        r_train_all, r_val_all = load_or_split_data(replay_dir, val_split=0.2)
        r_train = _stratified_subsample(r_train_all, int(round(len(primary_train) * replay_ratio)), seed=42)
        r_val = _stratified_subsample(r_val_all, int(round(len(primary_val) * replay_ratio)), seed=43)
        train_samples.extend(r_train)
        val_samples.extend(r_val)
        replay_train_cnt = len(r_train)
        replay_val_cnt = len(r_val)
        random.Random(42).shuffle(train_samples)
        random.Random(43).shuffle(val_samples)
    elif replay_dir:
        print(f"⚠️ Thư mục replay_dir không tồn tại ({replay_dir}), bỏ qua trộn dữ liệu nền.")

    return train_samples, val_samples, replay_train_cnt, replay_val_cnt


def ensure_pretrained_weights(train_dataset: str = "sd-v1_4") -> Tuple[str, str, str]:
    """Find or auto-download pretrained weights. Checks local, Google Drive, or downloads from HF."""
    candidates = [
        os.path.join("pretrained", train_dataset),
        os.path.join("/content/drive/MyDrive/CO-SPY/pretrained", train_dataset),
        os.path.join(SCRIPT_DIR, "pretrained", train_dataset),
        os.path.join("/content/CO-SPY/pretrained", train_dataset),
    ]

    req_files = ["semantic_weights.pth", "artifact_weights.pth", "fusion_weights.pth"]

    for base in candidates:
        if not os.path.exists(base):
            continue
        sem_p = os.path.join(base, "semantic_weights.pth")
        art_p = os.path.join(base, "artifact_weights.pth")
        fus_p = os.path.join(base, "fusion_weights.pth")
        if os.path.exists(sem_p) and os.path.exists(art_p) and os.path.getsize(sem_p) > 0 and os.path.getsize(art_p) > 0:
            print(f"✅ Đã tìm thấy trọng số nền tảng tại: {base}")
            local_target = os.path.join("pretrained", train_dataset)
            if not os.path.exists(local_target):
                try:
                    os.makedirs(local_target, exist_ok=True)
                    for f_name in req_files:
                        src_f = os.path.join(base, f_name)
                        if os.path.exists(src_f):
                            shutil.copyfile(src_f, os.path.join(local_target, f_name))
                except Exception:
                    pass
            return sem_p, art_p, fus_p

    print(f"📥 Trọng số nền tảng '{train_dataset}' chưa có sẵn. Đang tự động tải từ HuggingFace về...")
    local_dir = os.path.join("pretrained", train_dataset)
    os.makedirs(local_dir, exist_ok=True)

    drive_pretrained = "/content/drive/MyDrive/CO-SPY/pretrained"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(os.path.join(drive_pretrained, train_dataset), exist_ok=True)

    url = f"https://huggingface.co/ruojiruoli/Co-Spy-Pretrained-Weights/resolve/main/{train_dataset}.zip"
    tmp_zip = f"/content/{train_dataset}.zip" if os.path.exists("/content") else f"{train_dataset}.zip"

    import urllib.request
    import zipfile
    urllib.request.urlretrieve(url, tmp_zip)

    with zipfile.ZipFile(tmp_zip, "r") as z:
        z.extractall("pretrained")
        if os.path.exists("/content/drive/MyDrive"):
            try:
                z.extractall(drive_pretrained)
            except Exception:
                pass

    if os.path.exists(tmp_zip):
        os.remove(tmp_zip)

    sem_p = os.path.join(local_dir, "semantic_weights.pth")
    art_p = os.path.join(local_dir, "artifact_weights.pth")
    fus_p = os.path.join(local_dir, "fusion_weights.pth")
    print(f"🎉 Tải và đồng bộ thành công trọng số '{train_dataset}'!")
    return sem_p, art_p, fus_p


def _register_torch_safe_globals() -> None:
    try:
        import numpy as np
        if hasattr(torch.serialization, "add_safe_globals"):
            for sub in ["_core", "core"]:
                mod = getattr(np, sub, None)
                if mod and hasattr(mod, "multiarray") and hasattr(mod.multiarray, "scalar"):
                    torch.serialization.add_safe_globals([mod.multiarray.scalar])
    except Exception:
        pass


def _torch_load_compat(path: str, map_location: str = "cpu") -> Any:
    _register_torch_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_google_drive_path(path: str) -> bool:
    norm = os.path.abspath(path).replace("\\", "/")
    return norm.startswith("/content/drive/")


def _save_torch_atomic(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if _is_google_drive_path(path):
        # Serialize to local SSD first, then overwrite in-place on Drive FUSE to reuse the same Drive file ID
        # and avoid creating duplicate same-named files via rename/os.replace.
        fd, tmp_local = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        try:
            torch.save(obj, tmp_local)
            # Clean up any leftover .tmp file on Drive from older versions
            stale_tmp = path + ".tmp"
            if os.path.exists(stale_tmp):
                try:
                    os.remove(stale_tmp)
                except OSError:
                    pass
            write_mode = "r+b" if os.path.exists(path) else "wb"
            try:
                dst_ctx = open(path, write_mode)
            except OSError:
                write_mode = "wb"
                dst_ctx = open(path, write_mode)
            with open(tmp_local, "rb") as src, dst_ctx as dst:
                if write_mode == "r+b":
                    dst.seek(0)
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                if write_mode == "r+b":
                    dst.truncate()
                dst.flush()
                try:
                    os.fsync(dst.fileno())
                except OSError:
                    pass
        finally:
            if os.path.exists(tmp_local):
                try:
                    os.remove(tmp_local)
                except OSError:
                    pass
        return

    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _extract_non_frozen_state(model: nn.Module, branch: str) -> Dict[str, torch.Tensor]:
    """Exclude heavy frozen CLIP/SigLIP or SD-VAE parameters from checkpoints & saved branch weights."""
    if branch == "semantic":
        return {
            "fc.weight": model.fc.weight.detach().cpu(),
            "fc.bias": model.fc.bias.detach().cpu(),
        }
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("artifact_encoder.vae.")
    }


def _compute_pos_weight(samples: List[Tuple[str, int]], device: str, enabled: bool = True) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    n_real = sum(1 for s in samples if s[1] == 0)
    n_fake = sum(1 for s in samples if s[1] == 1)
    if n_real == 0 or n_fake == 0:
        return None
    ratio = max(0.25, min(10.0, n_real / float(n_fake)))
    if abs(ratio - 1.0) < 0.05:
        return None
    return torch.tensor([ratio], dtype=torch.float32, device=device)


def _build_epoch_loader(
    dataset: Dataset,
    batch_size: int,
    epoch: int,
    start_batch_idx: int = 0,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, int]:
    """Build deterministic per-epoch DataLoader starting from start_batch_idx for exact mid-epoch resume."""
    n = len(dataset)
    perm = list(range(n))
    random.Random(1000 + epoch).shuffle(perm)
    total_batches = (n // batch_size) if drop_last else ((n + batch_size - 1) // batch_size)
    remaining = perm[start_batch_idx * batch_size :]
    loader = DataLoader(
        Subset(dataset, remaining),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader, total_batches


def _restore_checkpoint(
    ckpt_file: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epochs: int,
    device: str,
    label: str,
    num_train_samples: int,
    meta_fallback_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], int, float, float, int, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Restore completed-epoch or mid-epoch checkpoint cleanly."""
    start_epoch = 1
    best_acc = 0.0
    best_ap = 0.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    resume_mid: Optional[Dict[str, Any]] = None

    if not os.path.exists(ckpt_file):
        return None, start_epoch, best_acc, best_ap, best_epoch, history, resume_mid

    try:
        ckpt_data = _torch_load_compat(ckpt_file, map_location=device)
        last_epoch = ckpt_data.get("epoch", 0)
        mid = ckpt_data.get("mid_epoch")

        if last_epoch >= epochs and ckpt_data.get("is_finished", False) and not mid:
            print("=" * 80)
            print(f"🎉 [{label}] ĐÃ HOÀN TẤT ĐỦ {epochs} EPOCHS TỪ TRƯỚC!")
            print(f"   • Best Val Accuracy : {ckpt_data.get('best_acc', 0.0) * 100:.2f}% (Epoch {ckpt_data.get('best_epoch')})")
            print(f"   • Best Val AP       : {ckpt_data.get('best_ap', 0.0) * 100:.2f}%")
            print("=" * 80)
            if meta_fallback_path and os.path.exists(meta_fallback_path):
                with open(meta_fallback_path, "r", encoding="utf-8") as mf:
                    return json.load(mf), start_epoch, best_acc, best_ap, best_epoch, history, resume_mid
            return ckpt_data, start_epoch, best_acc, best_ap, best_epoch, history, resume_mid

        has_mid = (
            isinstance(mid, dict)
            and mid.get("epoch") == last_epoch + 1
            and mid.get("num_train_samples") == num_train_samples
            and 0 < mid.get("next_batch_idx", 0) < mid.get("num_batches", 0)
        )
        if last_epoch > 0 or has_mid:
            model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_data:
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
            best_acc = ckpt_data.get("best_acc", 0.0)
            best_ap = ckpt_data.get("best_ap", 0.0)
            best_epoch = ckpt_data.get("best_epoch", 0)
            history = ckpt_data.get("history", [])

            if has_mid:
                start_epoch = mid["epoch"]
                resume_mid = mid
                print("=" * 80)
                print(f"🔄 RESUME GIỮA EPOCH [{label}]:")
                print(f"   • Nạp từ        : {ckpt_file}")
                print(f"   • Epoch đang dở : {start_epoch}/{epochs} (tiếp tục từ batch {mid['next_batch_idx']}/{mid['num_batches']})")
                print(f"   • Best cũ       : {best_acc * 100:.2f}% (Epoch {best_epoch})")
                print("=" * 80)
            else:
                start_epoch = last_epoch + 1
                print("=" * 80)
                print(f"🔄 RESUME [{label}]: Tiếp tục từ Epoch {start_epoch}/{epochs} (Best cũ: {best_acc*100:.2f}%)")
                print("=" * 80)
    except Exception as err:
        print(f"⚠️ Không nạp được checkpoint cũ ({err}), bắt đầu lại từ Epoch 1.")

    return None, start_epoch, best_acc, best_ap, best_epoch, history, resume_mid


def _compute_branch_signature(save_dir: str) -> Optional[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]]:
    """Return file signatures for Stage 1/2 custom branch weights when Stage 1/2 checkpoints exist."""
    art_ckpt = os.path.join(save_dir, "ckpt_artifact.pth")
    sem_ckpt = os.path.join(save_dir, "ckpt_semantic.pth")
    if not os.path.exists(art_ckpt) and not os.path.exists(sem_ckpt):
        return None

    def _sig(path: str, marker_exists: bool) -> Optional[Tuple[int, int]]:
        if marker_exists and os.path.isfile(path):
            st = os.stat(path)
            return (int(st.st_size), int(st.st_mtime_ns))
        return None

    return (
        _sig(os.path.join(save_dir, "artifact_weights.pth"), os.path.exists(art_ckpt)),
        _sig(os.path.join(save_dir, "semantic_weights.pth"), os.path.exists(sem_ckpt)),
    )


def _train_single_branch(
    branch: str,
    data_dir: str,
    save_dir: str,
    train_dataset: str,
    epochs: int,
    batch_size: int,
    lr: Optional[float],
    device: str,
    resume: bool,
    replay_dir: Optional[str],
    replay_ratio: float,
    auto_pos_weight: bool,
    save_every_batches: int = 100,
) -> Dict[str, Any]:
    """Train/fine-tune standalone 'artifact' or 'semantic' branch with epoch & mid-epoch checkpointing."""
    assert branch in ("artifact", "semantic")
    os.makedirs(save_dir, exist_ok=True)

    if lr is None:
        lr = 1e-4 if branch == "artifact" else 5e-4

    ckpt_file = os.path.join(save_dir, f"ckpt_{branch}.pth")
    out_weights_file = os.path.join(save_dir, f"{branch}_weights.pth")

    train_samples, val_samples, replay_train_cnt, replay_val_cnt = _prepare_train_val_samples(
        data_dir, replay_dir=replay_dir, replay_ratio=replay_ratio
    )
    train_real = sum(1 for s in train_samples if s[1] == 0)
    train_fake = sum(1 for s in train_samples if s[1] == 1)
    val_real = sum(1 for s in val_samples if s[1] == 0)
    val_fake = sum(1 for s in val_samples if s[1] == 1)

    pos_w = _compute_pos_weight(train_samples, device=device, enabled=auto_pos_weight)

    print("=" * 80)
    print(f"🔥 BẮT ĐẦU TRAIN NHÁNH [{branch.upper()}] ({train_dataset}):")
    print(f"   • Thư mục dữ liệu chính : {data_dir}")
    if replay_dir:
        print(f"   • Dữ liệu Replay        : {replay_dir} (tỷ lệ {replay_ratio}x)")
    print(f"   • Thư mục lưu trên Drive: {save_dir}")
    print(f"   • Checkpoint resume     : {os.path.basename(ckpt_file)} -> Xuất: {os.path.basename(out_weights_file)}")
    print(f"   • Dữ liệu Train / Val   : {len(train_samples)} ({train_real} Real, {train_fake} Fake, {replay_train_cnt} Replay) / {len(val_samples)} ({val_real} Real, {val_fake} Fake)")
    if pos_w is not None:
        print(f"   • Cân bằng lệch nhãn    : pos_weight = {pos_w.item():.2f} (bù cho lớp Fake)")
    print(f"   • Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Device: {device.upper()}")
    print("=" * 80)

    det_module = importlib.import_module(f"detectors.{train_dataset}")
    sem_pre, art_pre, _ = ensure_pretrained_weights(train_dataset)

    if branch == "artifact":
        ArtifactDetector = getattr(det_module, "ArtifactDetector")
        model = ArtifactDetector()
        if os.path.exists(art_pre):
            model.load_weights(art_pre)
        for name, p in model.named_parameters():
            p.requires_grad = not name.startswith("artifact_encoder.vae.")
    else:
        SemanticDetector = getattr(det_module, "SemanticDetector")
        model = SemanticDetector()
        if os.path.exists(sem_pre):
            model.load_weights(sem_pre)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model.to(device)
    print(f"   • Tham số đang học ({branch}): {sum(p.numel() for p in trainable_params):,}")

    train_ds = SingleBranchDataset(train_samples, transform=model.train_transform)
    val_ds = SingleBranchDataset(val_samples, transform=model.val_transform)

    num_workers = 2 if os.name != "nt" else 0
    drop_last = (branch == "artifact" and len(train_ds) > batch_size and len(train_ds) % batch_size == 1)
    _, steps_per_epoch = _build_epoch_loader(
        train_ds, batch_size=batch_size, epoch=1, start_batch_idx=0,
        drop_last=drop_last, num_workers=num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)

    start_epoch = 1
    best_acc = 0.0
    best_ap = 0.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    resume_mid: Optional[Dict[str, Any]] = None

    if resume:
        done_report, start_epoch, best_acc, best_ap, best_epoch, history, resume_mid = _restore_checkpoint(
            ckpt_file=ckpt_file,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            label=branch.upper(),
            num_train_samples=len(train_samples),
        )
        if done_report is not None:
            return done_report

    def save_branch_ckpt(epoch_idx: int, is_finished: bool = False, mid_epoch: Optional[Dict[str, Any]] = None) -> None:
        _save_torch_atomic({
            "epoch": epoch_idx,
            "branch": branch,
            "is_finished": is_finished,
            "mid_epoch": mid_epoch,
            "model_state_dict": _extract_non_frozen_state(model, branch),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "best_ap": best_ap,
            "best_epoch": best_epoch,
            "history": history,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, ckpt_file)

    save_interval = max(0, int(save_every_batches))

    for epoch in range(start_epoch, epochs + 1):
        if branch == "artifact":
            model.train()
            if hasattr(model, "artifact_encoder") and hasattr(model.artifact_encoder, "vae"):
                model.artifact_encoder.vae.eval()
        else:
            model.eval()  # SigLIP ViT has no BN; keep frozen clip eval()

        if resume_mid is not None and resume_mid.get("epoch") == epoch:
            start_batch_idx = int(resume_mid["next_batch_idx"])
            train_loss = float(resume_mid.get("train_loss_sum", 0.0))
            train_preds = list(resume_mid.get("train_preds", []))
            train_trues = list(resume_mid.get("train_trues", []))
            resume_mid = None
        else:
            start_batch_idx = 0
            train_loss = 0.0
            train_preds, train_trues = [], []

        epoch_loader, total_batches = _build_epoch_loader(
            train_ds, batch_size=batch_size, epoch=epoch, start_batch_idx=start_batch_idx,
            drop_last=drop_last, num_workers=num_workers, pin_memory=(device == "cuda")
        )

        pbar = tqdm(epoch_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [{branch}]", initial=start_batch_idx, total=total_batches)
        cur_next_batch = start_batch_idx
        try:
            for step_idx, (x_t, labels) in enumerate(pbar):
                x_t = x_t.to(device)
                labels_t = labels.to(device).unsqueeze(1)
                optimizer.zero_grad()

                if branch == "artifact" and x_t.size(0) == 1:
                    model.eval()
                logits = model(x_t)
                loss = criterion(logits, labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_preds.extend(logits.sigmoid().flatten().detach().cpu().tolist())
                train_trues.extend(labels.cpu().tolist())
                cur_next_batch = start_batch_idx + step_idx + 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if save_interval > 0 and (cur_next_batch % save_interval == 0) and (cur_next_batch < total_batches):
                    save_branch_ckpt(
                        epoch_idx=epoch - 1,
                        is_finished=False,
                        mid_epoch={
                            "epoch": epoch,
                            "next_batch_idx": cur_next_batch,
                            "num_batches": total_batches,
                            "num_train_samples": len(train_samples),
                            "train_loss_sum": train_loss,
                            "train_preds": train_preds,
                            "train_trues": train_trues,
                        },
                    )
        except KeyboardInterrupt:
            if cur_next_batch > start_batch_idx and cur_next_batch < total_batches:
                save_branch_ckpt(
                    epoch_idx=epoch - 1,
                    is_finished=False,
                    mid_epoch={
                        "epoch": epoch,
                        "next_batch_idx": cur_next_batch,
                        "num_batches": total_batches,
                        "num_train_samples": len(train_samples),
                        "train_loss_sum": train_loss,
                        "train_preds": train_preds,
                        "train_trues": train_trues,
                    },
                )
                print(f"\n💾 Đã lưu khẩn cấp tiến trình giữa Epoch {epoch} tại batch {cur_next_batch}/{total_batches} trước khi dừng!")
            raise

        _, train_acc = evaluate(train_preds, train_trues)

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x_t, labels in val_loader:
                logits = model(x_t.to(device))
                val_preds.extend(logits.sigmoid().flatten().cpu().tolist())
                val_trues.extend(labels.tolist())

        val_ap, val_acc = evaluate(val_preds, val_trues)
        avg_loss = train_loss / max(1, total_batches)
        print(
            f" [{branch.upper()}] Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | Val AP: {val_ap*100:5.2f}%"
        )

        history.append({
            "epoch": epoch,
            "branch": branch,
            "train_loss": avg_loss,
            "train_acc": round(train_acc * 100, 2),
            "val_acc": round(val_acc * 100, 2),
            "val_ap": round(val_ap * 100, 2),
        })

        if (val_acc > best_acc) or (val_acc == best_acc and val_ap >= best_ap):
            best_acc = val_acc
            best_ap = val_ap
            best_epoch = epoch
            _save_torch_atomic(_extract_non_frozen_state(model, branch), out_weights_file)

        save_branch_ckpt(epoch_idx=epoch, is_finished=(epoch == epochs), mid_epoch=None)

    if best_epoch == 0:
        _save_torch_atomic(_extract_non_frozen_state(model, branch), out_weights_file)

    report = {
        "stage": branch,
        "save_dir": save_dir,
        "weights_file": out_weights_file,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_acc * 100, 2),
        "best_val_ap": round(best_ap * 100, 2),
        "history": history,
    }
    print(f"✅ HOÀN TẤT NHÁNH [{branch.upper()}] -> Đã lưu {out_weights_file} (Best Val Acc: {best_acc*100:.2f}% | AP: {best_ap*100:.2f}%)\n")
    return report


def train_custom_model(
    data_dir: str = "Dataset/data/train/writing",
    save_dir: str = "/content/drive/MyDrive/CO-SPY/trained_models/writing",
    train_dataset: str = "sd-v1_4",
    mode: str = "fusion",
    epochs: int = 10,
    batch_size: int = 16,
    lr: Optional[float] = None,
    device: Optional[str] = None,
    resume: bool = True,
    replay_dir: Optional[str] = None,
    replay_ratio: float = 1.0,
    auto_pos_weight: bool = True,
    save_every_batches: int = 100,
) -> Dict[str, Any]:
    """Train/Fine-tune CO-SPY stage ('artifact', 'semantic', 'fusion', or 'all_3_stages')."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "all_3_stages":
        return train_three_stage_pipeline(
            data_dir=data_dir,
            save_dir=save_dir,
            train_dataset=train_dataset,
            artifact_epochs=epochs,
            semantic_epochs=epochs,
            fusion_epochs=max(1, min(3, epochs)),
            batch_size=batch_size,
            device=device,
            resume=resume,
            replay_dir=replay_dir,
            replay_ratio=replay_ratio,
            auto_pos_weight=auto_pos_weight,
            save_every_batches=save_every_batches,
        )

    if mode in ("artifact", "semantic"):
        return _train_single_branch(
            branch=mode,
            data_dir=data_dir,
            save_dir=save_dir,
            train_dataset=train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            resume=resume,
            replay_dir=replay_dir,
            replay_ratio=replay_ratio,
            auto_pos_weight=auto_pos_weight,
            save_every_batches=save_every_batches,
        )

    if mode != "fusion":
        raise ValueError(f"Unsupported mode '{mode}'. Choose from 'artifact', 'semantic', 'fusion', 'all_3_stages'.")

    if lr is None:
        lr = 1e-2

    os.makedirs(save_dir, exist_ok=True)
    ckpt_file = os.path.join(save_dir, "train_checkpoint.pth")

    # Ưu tiên nạp 2 nhánh Artifact/Semantic đã train riêng trong save_dir nếu có!
    custom_sem_file = os.path.join(save_dir, "semantic_weights.pth")
    custom_art_file = os.path.join(save_dir, "artifact_weights.pth")
    sem_pre, art_pre, fus_pre = ensure_pretrained_weights(train_dataset)
    using_custom_sem = os.path.isfile(custom_sem_file) and os.path.getsize(custom_sem_file) > 0
    using_custom_art = os.path.isfile(custom_art_file) and os.path.getsize(custom_art_file) > 0

    print("=" * 80)
    print("🔥 BẮT ĐẦU HUẤN LUYỆN FUSION (HỖ TRỢ CHẠY TIẾP DỞ DANG):")
    print(f"   • Thư mục dữ liệu train : {data_dir}")
    if replay_dir:
        print(f"   • Dữ liệu Replay        : {replay_dir} (tỷ lệ {replay_ratio}x)")
    print(f"   • Thư mục lưu trên Drive: {save_dir}")
    print(f"   • Nhánh Semantic dùng   : {'CUSTOM trong save_dir' if using_custom_sem else f'Pretrained {train_dataset}'}")
    print(f"   • Nhánh Artifact dùng   : {'CUSTOM trong save_dir' if using_custom_art else f'Pretrained {train_dataset}'}")
    print(f"   • Chế độ chạy tiếp dở   : {'BẬT (Auto-Resume)' if resume else 'TẮT (Chạy lại từ đầu)'}")
    print(f"   • Tổng số Epochs        : {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"   • Thiết bị tính toán    : {device.upper()}")
    print("=" * 80)

    train_samples, val_samples, replay_train_cnt, replay_val_cnt = _prepare_train_val_samples(
        data_dir, replay_dir=replay_dir, replay_ratio=replay_ratio
    )
    train_real = sum(1 for s in train_samples if s[1] == 0)
    train_fake = sum(1 for s in train_samples if s[1] == 1)
    val_real = sum(1 for s in val_samples if s[1] == 0)
    val_fake = sum(1 for s in val_samples if s[1] == 1)

    print("📊 DỮ LIỆU:")
    print(f"   • Tập Train : {len(train_samples)} ảnh (Real: {train_real} | Fake: {train_fake} | Replay: {replay_train_cnt})")
    print(f"   • Tập Val   : {len(val_samples)} ảnh (Real: {val_real} | Fake: {val_fake} | Replay: {replay_val_cnt})")

    det_module = importlib.import_module(f"detectors.{train_dataset}")
    CoSpyFusionDetector = getattr(det_module, "CoSpyFusionDetector")

    sem_to_load = custom_sem_file if using_custom_sem else sem_pre
    model = CoSpyFusionDetector(
        semantic_weights_path=sem_to_load,
        artifact_weights_path=art_pre,
    )
    if using_custom_art:
        model.art.load_weights(custom_art_file)

    if os.path.exists(fus_pre):
        try:
            model.load_weights(fus_pre)
        except Exception:
            pass

    for p in model.sem.parameters():
        p.requires_grad = False
    for p in model.art.parameters():
        p.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    model.to(device)

    train_dataset_obj = DualBranchDataset(
        train_samples,
        sem_transform=model.sem.train_transform,
        art_transform=model.art.train_transform,
    )
    val_dataset_obj = DualBranchDataset(
        val_samples,
        sem_transform=model.sem.val_transform,
        art_transform=model.art.val_transform,
    )

    num_workers = 2 if os.name != "nt" else 0
    _, steps_per_epoch = _build_epoch_loader(
        train_dataset_obj, batch_size=batch_size, epoch=1, start_batch_idx=0,
        drop_last=False, num_workers=num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(val_dataset_obj, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)

    bias_floor = 2.0 if train_dataset == "sd-v1_4" else 0.0

    start_epoch = 1
    best_acc = 0.0
    best_ap = 0.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    resume_mid: Optional[Dict[str, Any]] = None

    expected_branch_sig = _compute_branch_signature(save_dir)
    resume_fusion = resume
    if resume and expected_branch_sig is not None and os.path.exists(ckpt_file):
        try:
            prev_ckpt = _torch_load_compat(ckpt_file, map_location="cpu")
            if prev_ckpt.get("branch_signature") != expected_branch_sig:
                print(
                    "🔄 Phát hiện nhánh Artifact/Semantic mới được cập nhật trong thư mục writing — "
                    "bắt đầu train lại lớp Fusion từ Epoch 1 (bỏ qua train_checkpoint.pth cũ)."
                )
                resume_fusion = False
        except Exception:
            pass

    if resume_fusion:
        done_report, start_epoch, best_acc, best_ap, best_epoch, history, resume_mid = _restore_checkpoint(
            ckpt_file=ckpt_file,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            label="FUSION",
            num_train_samples=len(train_samples),
            meta_fallback_path=os.path.join(save_dir, "model_metadata.json"),
        )
        if done_report is not None:
            return done_report

    def save_train_checkpoint_atomic(epoch_idx: int, is_finished: bool = False, mid_epoch: Optional[Dict[str, Any]] = None) -> None:
        state = {
            "epoch": epoch_idx,
            "is_finished": is_finished,
            "mid_epoch": mid_epoch,
            "branch_signature": expected_branch_sig,
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
                if not k.startswith("sem.clip.") and not k.startswith("art.artifact_encoder.vae.")
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "best_ap": best_ap,
            "best_epoch": best_epoch,
            "history": history,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_torch_atomic(state, ckpt_file)

    cur_w = model.fc.weight.data.cpu().numpy().flatten().tolist()
    cur_b = model.fc.bias.data.cpu().item()
    save_interval = max(0, int(save_every_batches))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        model.sem.eval()
        model.art.eval()

        if resume_mid is not None and resume_mid.get("epoch") == epoch:
            start_batch_idx = int(resume_mid["next_batch_idx"])
            train_loss = float(resume_mid.get("train_loss_sum", 0.0))
            train_preds = list(resume_mid.get("train_preds", []))
            train_trues = list(resume_mid.get("train_trues", []))
            resume_mid = None
        else:
            start_batch_idx = 0
            train_loss = 0.0
            train_preds, train_trues = [], []

        epoch_loader, total_batches = _build_epoch_loader(
            train_dataset_obj, batch_size=batch_size, epoch=epoch, start_batch_idx=start_batch_idx,
            drop_last=False, num_workers=num_workers, pin_memory=(device == "cuda")
        )

        pbar = tqdm(epoch_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", initial=start_batch_idx, total=total_batches)
        cur_next_batch = start_batch_idx
        try:
            for step_idx, (sem_t, art_t, labels) in enumerate(pbar):
                sem_t = sem_t.to(device)
                art_t = art_t.to(device)
                labels_tensor = labels.to(device).unsqueeze(1)

                optimizer.zero_grad()

                with torch.no_grad():
                    pred_sem = model.sem(sem_t)
                    pred_art = model.art(art_t)
                    feats = torch.cat([pred_sem, pred_art], dim=1)

                logits = model.fc(feats)
                loss = criterion(logits, labels_tensor)

                b = model.fc.bias[0]
                if bias_floor > 0:
                    loss = loss + torch.relu(bias_floor - b)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                probs = logits.sigmoid().flatten().detach().cpu()
                train_preds.extend(probs.tolist())
                train_trues.extend(labels.cpu().tolist())
                cur_next_batch = start_batch_idx + step_idx + 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if save_interval > 0 and (cur_next_batch % save_interval == 0) and (cur_next_batch < total_batches):
                    save_train_checkpoint_atomic(
                        epoch_idx=epoch - 1,
                        is_finished=False,
                        mid_epoch={
                            "epoch": epoch,
                            "next_batch_idx": cur_next_batch,
                            "num_batches": total_batches,
                            "num_train_samples": len(train_samples),
                            "train_loss_sum": train_loss,
                            "train_preds": train_preds,
                            "train_trues": train_trues,
                        },
                    )
        except KeyboardInterrupt:
            if cur_next_batch > start_batch_idx and cur_next_batch < total_batches:
                save_train_checkpoint_atomic(
                    epoch_idx=epoch - 1,
                    is_finished=False,
                    mid_epoch={
                        "epoch": epoch,
                        "next_batch_idx": cur_next_batch,
                        "num_batches": total_batches,
                        "num_train_samples": len(train_samples),
                        "train_loss_sum": train_loss,
                        "train_preds": train_preds,
                        "train_trues": train_trues,
                    },
                )
                print(f"\n💾 Đã lưu khẩn cấp tiến trình Fusion giữa Epoch {epoch} tại batch {cur_next_batch}/{total_batches} trước khi dừng!")
            raise

        _, train_acc = evaluate(train_preds, train_trues)

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for sem_t, art_t, labels in val_loader:
                pred_sem = model.sem(sem_t.to(device))
                pred_art = model.art(art_t.to(device))
                feats = torch.cat([pred_sem, pred_art], dim=1)
                logits = model.fc(feats)
                val_preds.extend(logits.sigmoid().flatten().cpu().tolist())
                val_trues.extend(labels.tolist())

        val_ap, val_acc = evaluate(val_preds, val_trues)
        cur_w = model.fc.weight.data.cpu().numpy().flatten().tolist()
        cur_b = model.fc.bias.data.cpu().item()

        avg_loss = train_loss / max(1, total_batches)
        print(
            f" Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | "
            f"Val AP: {val_ap*100:5.2f}% | Fusion [w_sem={cur_w[0]:.3f}, w_art={cur_w[1]:.3f}, b={cur_b:.3f}]"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_acc": round(train_acc * 100, 2),
            "val_acc": round(val_acc * 100, 2),
            "val_ap": round(val_ap * 100, 2),
            "w_sem": cur_w[0],
            "w_art": cur_w[1],
            "bias": cur_b,
        })

        if val_acc >= best_acc:
            best_acc = val_acc
            best_ap = val_ap
            best_epoch = epoch
            model.save_weights(os.path.join(save_dir, "best_model.pth"))

        save_train_checkpoint_atomic(epoch, is_finished=(epoch == epochs), mid_epoch=None)

    fusion_out = os.path.join(save_dir, "fusion_weights.pth")
    _save_torch_atomic({k: v.detach().cpu() for k, v in model.fc.state_dict().items()}, fusion_out)

    # Chỉ copy pretrained vào save_dir nếu chưa có file custom từ Bước 1 (artifact) / Bước 2 (semantic)
    if not os.path.exists(custom_sem_file) and os.path.exists(sem_pre):
        shutil.copyfile(sem_pre, custom_sem_file)
    if not os.path.exists(custom_art_file) and os.path.exists(art_pre):
        shutil.copyfile(art_pre, custom_art_file)

    save_train_checkpoint_atomic(epochs, is_finished=True)

    meta_path = os.path.join(save_dir, "model_metadata.json")
    meta_info = {
        "model_name": f"CO-SPY-{train_dataset}-writing",
        "mode": "fusion",
        "used_custom_artifact": using_custom_art,
        "used_custom_semantic": using_custom_sem,
        "dataset_path": data_dir,
        "replay_dir": replay_dir,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_acc * 100, 2),
        "best_val_ap": round(best_ap * 100, 2),
        "total_train_images": len(train_samples),
        "total_val_images": len(val_samples),
        "resumed_from_epoch": start_epoch if start_epoch > 1 else None,
        "final_fusion_weight": cur_w,
        "final_fusion_bias": cur_b,
        "history": history,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("🎉 HUẤN LUYỆN FUSION HOÀN TẤT!")
    print(f"   • Best Val Accuracy : {best_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"   • Best Val AP       : {best_ap*100:.2f}%")
    print(f"   • Danh sách model đã lưu trên Drive ({save_dir}):")
    print("     ├── best_model.pth        (Fusion Linear(2, 1) tốt nhất)")
    print("     ├── fusion_weights.pth    (Lớp Fusion Linear(2, 1) epoch cuối)")
    print("     ├── semantic_weights.pth  (Nhánh Semantic)")
    print("     ├── artifact_weights.pth  (Nhánh Artifact)")
    print("     ├── train_checkpoint.pth  (Checkpoint Fusion)")
    print("     └── model_metadata.json   (Thông số & lịch sử)")
    print("=" * 80)
    return meta_info


def train_three_stage_pipeline(
    data_dir: str = "Dataset/data/train/writing",
    save_dir: str = "/content/drive/MyDrive/CO-SPY/trained_models/writing",
    train_dataset: str = "sd-v1_4",
    artifact_epochs: int = 5,
    semantic_epochs: int = 5,
    fusion_epochs: int = 2,
    batch_size: int = 16,
    artifact_lr: float = 1e-4,
    semantic_lr: float = 5e-4,
    fusion_lr: float = 1e-2,
    device: Optional[str] = None,
    resume: bool = True,
    replay_dir: Optional[str] = None,
    replay_ratio: float = 1.0,
    auto_pos_weight: bool = True,
    save_every_batches: int = 100,
) -> Dict[str, Any]:
    """Run the sequential 3-stage CO-SPY training pipeline: Artifact -> Semantic -> Fusion."""
    print("=" * 80)
    print("🚀 QUY TRÌNH TRAIN TUẦN TỰ 3 BƯỚC: ARTIFACT -> SEMANTIC -> FUSION")
    print(f"   • Dữ liệu  : {data_dir}")
    print(f"   • Lưu tại  : {save_dir}")
    print("=" * 80)

    art_rep = _train_single_branch(
        branch="artifact",
        data_dir=data_dir,
        save_dir=save_dir,
        train_dataset=train_dataset,
        epochs=artifact_epochs,
        batch_size=batch_size,
        lr=artifact_lr,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        resume=resume,
        replay_dir=replay_dir,
        replay_ratio=replay_ratio,
        auto_pos_weight=auto_pos_weight,
        save_every_batches=save_every_batches,
    )

    sem_rep = _train_single_branch(
        branch="semantic",
        data_dir=data_dir,
        save_dir=save_dir,
        train_dataset=train_dataset,
        epochs=semantic_epochs,
        batch_size=batch_size,
        lr=semantic_lr,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        resume=resume,
        replay_dir=replay_dir,
        replay_ratio=replay_ratio,
        auto_pos_weight=auto_pos_weight,
        save_every_batches=save_every_batches,
    )

    fus_rep = train_custom_model(
        data_dir=data_dir,
        save_dir=save_dir,
        train_dataset=train_dataset,
        mode="fusion",
        epochs=fusion_epochs,
        batch_size=batch_size,
        lr=fusion_lr,
        device=device,
        resume=resume,
        replay_dir=replay_dir,
        replay_ratio=replay_ratio,
        auto_pos_weight=auto_pos_weight,
        save_every_batches=save_every_batches,
    )

    return {"artifact": art_rep, "semantic": sem_rep, "fusion": fus_rep}


def main():
    parser = argparse.ArgumentParser(description="CO-SPY Custom 3-Stage Training with Resume Support")
    parser.add_argument("--data_dir", type=str, default="Dataset/data/train/writing")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/CO-SPY/trained_models/writing")
    parser.add_argument("--train_dataset", type=str, default="sd-v1_4", choices=["sd-v1_4", "progan"])
    parser.add_argument("--mode", type=str, default="fusion", choices=["artifact", "semantic", "fusion", "all_3_stages"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--artifact_epochs", type=int, default=5)
    parser.add_argument("--semantic_epochs", type=int, default=5)
    parser.add_argument("--fusion_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--replay_dir", type=str, default=None)
    parser.add_argument("--replay_ratio", type=float, default=1.0)
    parser.add_argument("--save_every_batches", type=int, default=100)
    parser.add_argument("--no_pos_weight", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    if args.mode == "all_3_stages":
        train_three_stage_pipeline(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            train_dataset=args.train_dataset,
            artifact_epochs=args.artifact_epochs,
            semantic_epochs=args.semantic_epochs,
            fusion_epochs=args.fusion_epochs,
            batch_size=args.batch_size,
            device=args.device,
            resume=not args.no_resume,
            replay_dir=args.replay_dir,
            replay_ratio=args.replay_ratio,
            auto_pos_weight=not args.no_pos_weight,
            save_every_batches=args.save_every_batches,
        )
    else:
        train_custom_model(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            train_dataset=args.train_dataset,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            resume=not args.no_resume,
            replay_dir=args.replay_dir,
            replay_ratio=args.replay_ratio,
            auto_pos_weight=not args.no_pos_weight,
            save_every_batches=args.save_every_batches,
        )


if __name__ == "__main__":
    main()

"""Google Drive integration and evaluation tool for CO-SPY (drive_tool.py).

Features:
1. Pretrained Weights Sync & Resume:
   - Checks if weights already exist on Drive before downloading/uploading.
2. Robust Test Dataset Download from Drive Link via gdown.
3. Fault-Tolerant & Resumable Image Testing across 3 dedicated test splits:
   - aigc       : Dataset/data/test/AIGCDetectionBenchMark/test
   - cospy_real : Dataset/data/test/Co-Spy-Benchreal_image_examples
   - synthetic  : Dataset/data/test/synthetic
   Each split uses its own checkpoint & result JSON files so progress never collides.
"""

import os
import sys
import csv
import json
import shutil
import argparse
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from tqdm import tqdm

# Ensure project root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if TYPE_CHECKING:
    from cospy_infer.engine import InferenceEngine

from cospy_infer.jpeg_probe import probe_jpeg

REQUIRED_WEIGHTS = {
    "sd-v1_4": ["artifact_weights.pth", "semantic_weights.pth", "fusion_weights.pth"],
    "progan": ["artifact_weights.pth", "semantic_weights.pth", "fusion_weights.pth"]
}

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/CO-SPY"
DEFAULT_WRITING_MODEL_DIR = "/content/drive/MyDrive/CO-SPY/trained_models/writing"

TEST_SPLITS: Dict[str, Dict[str, Any]] = {
    "aigc": {
        "title": "AIGCDetectionBenchMark/test",
        "drive_dir": "/content/drive/MyDrive/Dataset/data/test/AIGCDetectionBenchMark/test",
        "local_dir": "Dataset/data/test/AIGCDetectionBenchMark/test",
        "fixed_label": None,
        "checkpoint_file": "test_checkpoint_aigc.json",
        "backup_file": "test_checkpoint_aigc_backup.json",
        "results_file": "test_results_aigc.json",
    },
    "cospy_real": {
        "title": "Co-Spy-Benchreal_image_examples",
        "drive_dir": "/content/drive/MyDrive/Dataset/data/test/Co-Spy-Benchreal_image_examples",
        "local_dir": "Dataset/data/test/Co-Spy-Benchreal_image_examples",
        "fixed_label": 0,
        "checkpoint_file": "test_checkpoint_cospy_real.json",
        "backup_file": "test_checkpoint_cospy_real_backup.json",
        "results_file": "test_results_cospy_real.json",
    },
    "synthetic": {
        "title": "synthetic",
        "drive_dir": "/content/drive/MyDrive/Dataset/data/test/synthetic",
        "local_dir": "Dataset/data/test/synthetic",
        "fixed_label": 1,
        "checkpoint_file": "test_checkpoint_synthetic.json",
        "backup_file": "test_checkpoint_synthetic_backup.json",
        "results_file": "test_results_synthetic.json",
    },
}

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def check_model_complete(model_name: str, target_dir: str) -> bool:
    """Check if all required weights for model exist in target_dir."""
    model_folder = os.path.join(target_dir, model_name)
    if not os.path.exists(model_folder):
        return False
    for req in REQUIRED_WEIGHTS.get(model_name, []):
        fp = os.path.join(model_folder, req)
        if not os.path.exists(fp) or os.path.getsize(fp) == 0:
            return False
    return True


def upload_via_local_drive_path(dest_path: str) -> List[str]:
    """Copy pretrained weights to local Google Drive folder, skipping existing ones."""
    src_dir = os.path.join(SCRIPT_DIR, "pretrained")
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source pretrained folder not found: {src_dir}")

    dest_pretrained = os.path.join(dest_path, "CO-SPY-pretrained")
    os.makedirs(dest_pretrained, exist_ok=True)

    copied = []
    for model_name in ["sd-v1_4", "progan"]:
        if check_model_complete(model_name, dest_pretrained):
            print(f"✅ Weights for '{model_name}' already complete on Drive. Skipping.")
            continue

        model_src = os.path.join(src_dir, model_name)
        if not os.path.exists(model_src):
            continue

        model_dest = os.path.join(dest_pretrained, model_name)
        os.makedirs(model_dest, exist_ok=True)

        for f in os.listdir(model_src):
            if f.endswith(".pth"):
                s_file = os.path.join(model_src, f)
                d_file = os.path.join(model_dest, f)
                print(f"Copying to Drive: {model_name}/{f} ({os.path.getsize(s_file)/(1024*1024):.1f} MB)...")
                shutil.copy2(s_file, d_file)
                copied.append(d_file)

    print(f"Sync complete to: {dest_pretrained}")
    return copied


def download_test_dataset_from_drive(drive_url: str, output_dir: str = "Dataset/data/test") -> str:
    """Download folder or file from Google Drive link using gdown."""
    import gdown
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nDownloading test dataset from Google Drive...")
    print(f"  URL        : {drive_url}")
    print(f"  Destination: {output_dir}")

    if "folders" in drive_url or "id=" in drive_url:
        try:
            gdown.download_folder(url=drive_url, output=output_dir, quiet=False, use_cookies=False)
        except Exception:
            gdown.download(url=drive_url, output=output_dir, quiet=False, fuzzy=True)
    else:
        zip_out = os.path.join(output_dir, "test_images.zip")
        gdown.download(url=drive_url, output=zip_out, quiet=False, fuzzy=True)
        if os.path.exists(zip_out):
            import zipfile
            with zipfile.ZipFile(zip_out, "r") as z:
                z.extractall(output_dir)
            os.remove(zip_out)

    print("Download finished!")
    return output_dir


def normalize_rel_key(rel_str: str) -> str:
    """Normalize relative image keys across OSes so resume matches reliably."""
    k = rel_str.replace("\\", "/")
    while k.startswith("./"):
        k = k[2:]
    return k


def _infer_fixed_label_from_dir(test_dir: str) -> Optional[int]:
    """Infer dataset-wide ground-truth label when pointing straight at a single-class split."""
    norm = test_dir.rstrip("/\\").lower().replace("\\", "/")
    tail = norm.split("/")[-1] if norm else ""
    if tail.endswith("real_image_examples"):
        return 0
    if tail == "synthetic":
        return 1
    return None


def extract_ground_truth_label(filepath: str, fixed_label: Optional[int] = None) -> Optional[int]:
    """Detect ground truth label from fixed split label or path structure (0=Real, 1=Fake)."""
    if fixed_label is not None:
        return fixed_label
    norm = filepath.lower().replace("\\", "/")
    parts = norm.split("/")
    for p in parts:
        if "0_real" in p or p == "real" or p.endswith("real_image_examples"):
            return 0
        if "1_fake" in p or p in ("fake", "synthetic"):
            return 1
    fname = os.path.basename(norm)
    if "_real" in fname or fname.startswith("real_"):
        return 0
    if "_fake" in fname or fname.startswith("fake_"):
        return 1
    return None


def resolve_split_test_dir(split_key: str, custom_dir: Optional[str] = None) -> str:
    """Resolve the directory path for one of the 3 test splits (Drive first, local fallback)."""
    if split_key not in TEST_SPLITS:
        raise ValueError(f"Unknown test split '{split_key}'. Choose from: {list(TEST_SPLITS.keys())}")
    if custom_dir:
        return custom_dir
    cfg = TEST_SPLITS[split_key]
    if os.path.exists(cfg["drive_dir"]):
        return cfg["drive_dir"]
    if os.path.exists(cfg["local_dir"]):
        return cfg["local_dir"]
    return cfg["drive_dir"]


def has_valid_weights_dir(weights_dir: Optional[str]) -> bool:
    """Check whether weights_dir exists and contains a recognized fine-tuned checkpoint."""
    if not weights_dir or not os.path.isdir(weights_dir):
        return False
    for fn in ("best_model.pth", "fusion_weights.pth", "train_checkpoint.pth"):
        fp = os.path.join(weights_dir, fn)
        if os.path.isfile(fp) and os.path.getsize(fp) > 0:
            return True
    return False


def _sync_pretrained_from_drive(train_dataset: str) -> None:
    """Ensure local pretrained/<train_dataset> exists by copying from Drive cache if missing."""
    required = REQUIRED_WEIGHTS.get(train_dataset, [])
    local_dir = os.path.join(SCRIPT_DIR, "pretrained", train_dataset)
    if required and all(os.path.isfile(os.path.join(local_dir, fn)) and os.path.getsize(os.path.join(local_dir, fn)) > 0 for fn in required):
        return

    for drive_base in (os.path.join(DEFAULT_DRIVE_DIR, "pretrained"), os.path.join(DEFAULT_DRIVE_DIR, "CO-SPY-pretrained")):
        cand_dir = os.path.join(drive_base, train_dataset)
        if os.path.isdir(cand_dir):
            os.makedirs(local_dir, exist_ok=True)
            for fn in os.listdir(cand_dir):
                if fn.endswith(".pth"):
                    src_f = os.path.join(cand_dir, fn)
                    dst_f = os.path.join(local_dir, fn)
                    if not os.path.isfile(dst_f) or os.path.getsize(dst_f) == 0:
                        shutil.copy2(src_f, dst_f)
            break


def load_writing_engine(
    weights_dir: str = DEFAULT_WRITING_MODEL_DIR,
    train_dataset: str = "sd-v1_4",
    device: Optional[str] = None,
    require_weights: bool = True,
) -> "InferenceEngine":
    """Load the fixed fine-tuned 'writing' model on top of pretrained sd-v1_4 backbones."""
    from cospy_infer.engine import InferenceEngine

    _sync_pretrained_from_drive(train_dataset)

    if has_valid_weights_dir(weights_dir):
        loaded_file = next(
            fn for fn in ("best_model.pth", "fusion_weights.pth", "train_checkpoint.pth")
            if os.path.isfile(os.path.join(weights_dir, fn))
        )
        has_art = os.path.isfile(os.path.join(weights_dir, "artifact_weights.pth"))
        has_sem = os.path.isfile(os.path.join(weights_dir, "semantic_weights.pth"))
        print("=" * 80)
        print("🎯 ĐANG NẠP MODEL 'writing' ĐÃ FINE-TUNE TRÊN GOOGLE DRIVE:")
        print(f"   • Thư mục model : {weights_dir}")
        print(f"   • Artifact      : {'artifact_weights.pth (Custom)' if has_art else f'Pretrained ({train_dataset})'}")
        print(f"   • Semantic      : {'semantic_weights.pth (Custom)' if has_sem else f'Pretrained ({train_dataset})'}")
        print(f"   • Fusion Head   : {loaded_file}")
        print("=" * 80)
        resolved_weights = weights_dir
    elif require_weights:
        has_stage12 = os.path.isdir(weights_dir) and any(
            os.path.isfile(os.path.join(weights_dir, fn))
            for fn in ("ckpt_artifact.pth", "ckpt_semantic.pth", "artifact_weights.pth", "semantic_weights.pth")
        )
        hint = (
            "Đã thấy file nhánh Stage 1/2 nhưng chưa có 'best_model.pth' của Stage 3 — hãy chạy CO_SPY_Train_Stage3_Fusion.ipynb trước khi test!"
            if has_stage12
            else "Hãy chạy train model 'writing' lên Drive trước khi test."
        )
        raise FileNotFoundError(f"Không tìm thấy model 'writing' hoàn chỉnh tại '{weights_dir}'. {hint}")
    else:
        print("=" * 80)
        print(f"⚠️ Chưa tìm thấy model tại {weights_dir}, dùng tạm pretrained '{train_dataset}'.")
        print("=" * 80)
        resolved_weights = None

    return InferenceEngine(
        train_dataset=train_dataset,
        mode="fusion",
        pretrain=True,
        device=device,
        route="branch_native",
        views=1,
        weights_dir=resolved_weights,
    )


def _is_google_drive_path(path: str) -> bool:
    norm = os.path.abspath(path).replace("\\", "/")
    return norm.startswith("/content/drive/")


def _save_json_atomic(path: Optional[str], data: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        if _is_google_drive_path(path):
            # Overwrite in-place on Drive FUSE so Google Drive reuses the same file ID without duplicates.
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
            return
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        pass


def _compute_classification_metrics(results_dict: Dict[str, Any], compute_ap: bool = False) -> Dict[str, Any]:
    """Compute confusion counts, per-class accuracies, and optional AP from prediction entries."""
    tp = tn = fp = fn = 0
    y_true: List[int] = []
    y_score: List[float] = []

    for entry in results_dict.values():
        if "error" in entry:
            continue
        gt_name = entry.get("ground_truth")
        is_ai = entry.get("is_ai_generated")
        if gt_name not in ("Real", "AI-Generated") or is_ai is None:
            continue
        gt_val = 1 if gt_name == "AI-Generated" else 0
        pred_val = 1 if bool(is_ai) else 0
        if gt_val == 1 and pred_val == 1:
            tp += 1
        elif gt_val == 0 and pred_val == 0:
            tn += 1
        elif gt_val == 0 and pred_val == 1:
            fp += 1
        else:
            fn += 1

        if compute_ap and entry.get("prob_fusion") is not None:
            y_true.append(gt_val)
            y_score.append(float(entry["prob_fusion"]))

    labeled = tp + tn + fp + fn
    correct = tp + tn
    real_labeled = tn + fp
    ai_labeled = tp + fn

    ap_val: Optional[float] = None
    if compute_ap and len(y_true) > 0 and 0 in y_true and 1 in y_true:
        try:
            from sklearn.metrics import average_precision_score
            ap_val = round(float(average_precision_score(y_true, y_score)) * 100, 2)
        except Exception:
            ap_val = None

    return {
        "labeled_samples": labeled,
        "correct_samples": correct,
        "accuracy": round(correct / labeled * 100, 2) if labeled > 0 else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "real_labeled": real_labeled,
        "real_accuracy": round(tn / real_labeled * 100, 2) if real_labeled > 0 else None,
        "ai_labeled": ai_labeled,
        "ai_accuracy": round(tp / ai_labeled * 100, 2) if ai_labeled > 0 else None,
        "ap": ap_val,
    }


def _build_summary(
    results_dict: Dict[str, Any],
    folder_summaries: Dict[str, Any],
    total_done: int,
    ai_cnt: int,
    real_cnt: int,
    labeled_cnt: int,
    correct_cnt: int,
    error_cnt: int,
    split_name: Optional[str] = None,
    test_dir: Optional[str] = None,
    weights_dir: Optional[str] = None,
    compute_ap: bool = False,
) -> Dict[str, Any]:
    cls_metrics = _compute_classification_metrics(results_dict, compute_ap=compute_ap)
    m_labeled = cls_metrics["labeled_samples"] or labeled_cnt
    m_correct = cls_metrics["correct_samples"] if cls_metrics["labeled_samples"] > 0 else correct_cnt
    running_acc = cls_metrics["accuracy"] if cls_metrics["labeled_samples"] > 0 else (
        round((correct_cnt / labeled_cnt * 100), 2) if labeled_cnt > 0 else None
    )
    valid_preds = ai_cnt + real_cnt
    return {
        "split": split_name,
        "test_dir": test_dir,
        "weights_dir": weights_dir,
        "completed_folders": sum(1 for v in folder_summaries.values() if v.get("is_finished") is True),
        "completed_images": total_done,
        "error_count": error_cnt,
        "ai_count": ai_cnt,
        "ai_pct": round(ai_cnt / max(1, valid_preds) * 100, 2),
        "real_count": real_cnt,
        "real_pct": round(real_cnt / max(1, valid_preds) * 100, 2),
        "labeled_samples": m_labeled,
        "correct_samples": m_correct,
        "running_accuracy": running_acc,
        "real_accuracy": cls_metrics["real_accuracy"],
        "ai_accuracy": cls_metrics["ai_accuracy"],
        "ap": cls_metrics["ap"],
        "tp": cls_metrics["tp"],
        "tn": cls_metrics["tn"],
        "fp": cls_metrics["fp"],
        "fn": cls_metrics["fn"],
        "folder_summary": folder_summaries,
        "results": results_dict,
    }


def test_images_in_directory(
    test_dir: str = "Dataset/data/test",
    train_dataset: str = "sd-v1_4",
    device: Optional[str] = None,
    output_json: Optional[str] = None,
    checkpoint_json: Optional[str] = None,
    backup_json: Optional[str] = None,
    weights_dir: Optional[str] = None,
    fixed_label: Optional[int] = None,
    engine: Optional["InferenceEngine"] = None,
    split_name: Optional[str] = None,
    save_every: int = 10,
) -> Dict[str, Any]:
    """Streaming leaf-folder test run with 100% robust resume for partially finished runs."""
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu test: {test_dir}")

    if fixed_label is None:
        fixed_label = _infer_fixed_label_from_dir(test_dir)

    ckpt_basename = os.path.basename(checkpoint_json) if checkpoint_json else "test_checkpoint.json"
    local_ckpt = os.path.join(test_dir, ckpt_basename)
    write_local_copy = (checkpoint_json is None)
    ckpt_path = checkpoint_json or local_ckpt
    ckpt_backup = backup_json or os.path.join(
        os.path.dirname(os.path.abspath(ckpt_path)),
        ckpt_basename.replace(".json", "_backup.json"),
    )
    if output_json is None:
        output_json = os.path.join(test_dir, "test_results.json")

    # 1. Nạp và khôi phục tiến trình dự đoán dở dang (Resume Check)
    results_dict: Dict[str, Any] = {}
    folder_summaries: Dict[str, Any] = {}
    ckpt_source = None

    candidates = []
    for p in (ckpt_path, local_ckpt, ckpt_backup, output_json):
        if p and p not in candidates:
            candidates.append(p)

    best_data = None
    best_count = -1
    for cp in candidates:
        if os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                cnt = len(loaded.get("results", {}))
                if cnt > best_count:
                    best_count = cnt
                    best_data = loaded
                    ckpt_source = cp
            except Exception:
                pass

    if best_data and best_count > 0:
        for k, v in best_data.get("results", {}).items():
            results_dict[normalize_rel_key(k)] = v
        folder_summaries = best_data.get("folder_summary", {})

        prev_correct = sum(1 for v in results_dict.values() if v.get("is_correct") is True)
        prev_labeled = sum(1 for v in results_dict.values() if v.get("is_correct") is not None)
        finished_f_count = sum(1 for v in folder_summaries.values() if v.get("is_finished") is True)
        prev_acc = round((prev_correct / prev_labeled * 100), 2) if prev_labeled > 0 else "N/A"

        print("=" * 80)
        print(f"🔄 RESUME [{split_name or test_dir}]:")
        print(f"   • Nạp từ file          : {ckpt_source}")
        print(f"   • Số ảnh đã test trước : {len(results_dict)} ảnh")
        print(f"   • Số thư mục đã xong   : {finished_f_count} thư mục")
        print(f"   • Accuracy tích lũy cũ : {prev_acc}% ({prev_correct}/{prev_labeled} ảnh)")
        print("   👉 Tự động bỏ qua ảnh đã xong và chạy tiếp ngay tại ảnh dở dang!")
        print("=" * 80)
    else:
        print(f"ℹ️ Bắt đầu phiên kiểm thử mới [{split_name or test_dir}].")

    # 2. Khởi tạo Engine nếu chưa truyền sẵn từ ngoài
    if engine is None:
        if weights_dir is not None:
            engine = load_writing_engine(
                weights_dir=weights_dir,
                train_dataset=train_dataset,
                device=device,
                require_weights=True,
            )
        else:
            from cospy_infer.engine import InferenceEngine
            engine = InferenceEngine(
                train_dataset=train_dataset,
                mode="fusion",
                pretrain=True,
                device=device,
                route="branch_native",
                views=1,
            )

    def save_checkpoint_atomic(data: Dict[str, Any], include_backup: bool = False) -> None:
        _save_json_atomic(ckpt_path, data)
        if write_local_copy and local_ckpt != ckpt_path:
            _save_json_atomic(local_ckpt, data)
        if include_backup:
            _save_json_atomic(ckpt_backup, data)

    def is_image(fname: str) -> bool:
        return os.path.splitext(fname)[1].lower() in VALID_IMAGE_EXTS

    total_done = len(results_dict)
    ai_cnt = sum(1 for v in results_dict.values() if v.get("is_ai_generated") is True)
    real_cnt = sum(1 for v in results_dict.values() if v.get("is_ai_generated") is False)
    labeled_cnt = sum(1 for v in results_dict.values() if v.get("is_correct") is not None)
    correct_cnt = sum(1 for v in results_dict.values() if v.get("is_correct") is True)
    error_cnt = sum(1 for v in results_dict.values() if "error" in v)

    summary = _build_summary(
        results_dict, folder_summaries,
        total_done, ai_cnt, real_cnt, labeled_cnt, correct_cnt, error_cnt,
        split_name=split_name, test_dir=test_dir, weights_dir=weights_dir,
    )

    folder_counter = 0
    save_interval = max(1, int(save_every))

    try:
        for root, dirs, files in os.walk(test_dir, topdown=True):
            dirs.sort()
            img_files = [f for f in sorted(files) if is_image(f)]
            if not img_files:
                continue

            folder_counter += 1
            rel_folder = normalize_rel_key(os.path.relpath(root, test_dir))

            # Fast Skip cho thư mục đã hoàn tất
            f_info = folder_summaries.get(rel_folder, {})
            if f_info.get("is_finished") is True:
                acc_str = f" | Acc: {f_info.get('accuracy')}%" if f_info.get("accuracy") is not None else ""
                print(f"[{folder_counter}] ⏩ Đã xong từ trước: [{rel_folder}] ({f_info.get('total_images')} ảnh{acc_str}) -> Bỏ qua!")
                continue

            done_in_folder: List[str] = []
            pending_in_folder: List[str] = []
            for f in img_files:
                rel_name = normalize_rel_key(os.path.join(rel_folder, f))
                if rel_name in results_dict:
                    done_in_folder.append(f)
                else:
                    pending_in_folder.append(f)

            f_done_cnt = len(done_in_folder)
            f_ai = 0
            f_real = 0
            f_labeled = 0
            f_correct = 0
            for fn in done_in_folder:
                entry = results_dict.get(normalize_rel_key(os.path.join(rel_folder, fn)), {})
                if entry.get("is_ai_generated") is True:
                    f_ai += 1
                elif entry.get("is_ai_generated") is False:
                    f_real += 1
                if entry.get("is_correct") is not None:
                    f_labeled += 1
                    if entry.get("is_correct") is True:
                        f_correct += 1

            if not pending_in_folder:
                f_acc = round(f_correct / f_labeled * 100, 2) if f_labeled > 0 else None
                folder_summaries[rel_folder] = {
                    "folder": rel_folder,
                    "total_images": len(img_files),
                    "completed_images": f_done_cnt,
                    "is_finished": True,
                    "ai_count": f_ai,
                    "real_count": f_real,
                    "labeled_samples": f_labeled,
                    "correct_samples": f_correct,
                    "accuracy": f_acc,
                }
                summary = _build_summary(
                    results_dict, folder_summaries,
                    total_done, ai_cnt, real_cnt, labeled_cnt, correct_cnt, error_cnt,
                    split_name=split_name, test_dir=test_dir, weights_dir=weights_dir,
                )
                save_checkpoint_atomic(summary)
                continue

            print("\n" + "-" * 80)
            if done_in_folder:
                print(f"📁 [{folder_counter}] CHẠY TIẾP DỞ DANG: {rel_folder}")
                print(f"   • Đã xong {len(done_in_folder)}/{len(img_files)} ảnh -> Chạy tiếp từ ảnh thứ {len(done_in_folder) + 1}!")
            else:
                print(f"📁 [{folder_counter}] BẮT ĐẦU THƯ MỤC MỚI: {rel_folder} ({len(img_files)} ảnh)")
            print("-" * 80)

            pbar = tqdm(
                pending_in_folder,
                desc=f"[{folder_counter}] {rel_folder[-25:]}",
                initial=len(done_in_folder),
                total=len(img_files),
            )
            for idx_in_pending, f in enumerate(pbar, start=1):
                img_p = os.path.join(root, f)
                rel_name = normalize_rel_key(os.path.join(rel_folder, f))
                gt_label = extract_ground_truth_label(rel_name, fixed_label=fixed_label)

                try:
                    probe = probe_jpeg(img_p, train_dataset=train_dataset)
                    pred_res = engine.predict_image(img_p, allow_truncated_legacy=False)

                    is_ai = (pred_res["prediction"] == 1)
                    is_correct = (pred_res["prediction"] == gt_label) if gt_label is not None else None

                    results_dict[rel_name] = {
                        "file": rel_name,
                        "folder": rel_folder,
                        "prediction": pred_res["label_name"],
                        "is_ai_generated": is_ai,
                        "ground_truth": "Real" if gt_label == 0 else ("AI-Generated" if gt_label == 1 else "Unknown"),
                        "is_correct": is_correct,
                        "prob_fusion": pred_res["prob_fusion"],
                        "prob_semantic": pred_res["prob_sem"],
                        "prob_artifact": pred_res["prob_art"],
                        "logit_fusion": pred_res["logit_fusion"],
                        "logit_sem": pred_res["logit_sem"],
                        "logit_art": pred_res["logit_art"],
                        "jpeg_probe": {
                            "format": probe["format"],
                            "quality": probe["estimated_jpeg_quality"],
                            "is_ood": probe["is_ood"],
                        },
                    }
                    if is_ai:
                        f_ai += 1
                        ai_cnt += 1
                    else:
                        f_real += 1
                        real_cnt += 1
                    if is_correct is not None:
                        f_labeled += 1
                        labeled_cnt += 1
                        if is_correct:
                            f_correct += 1
                            correct_cnt += 1
                except Exception as err:
                    results_dict[rel_name] = {"file": rel_name, "folder": rel_folder, "error": str(err)}
                    error_cnt += 1

                f_done_cnt += 1
                total_done += 1
                f_acc = round(f_correct / f_labeled * 100, 2) if f_labeled > 0 else None
                running_acc = round((correct_cnt / labeled_cnt * 100), 2) if labeled_cnt > 0 else None

                folder_summaries[rel_folder] = {
                    "folder": rel_folder,
                    "total_images": len(img_files),
                    "completed_images": f_done_cnt,
                    "is_finished": f_done_cnt == len(img_files),
                    "ai_count": f_ai,
                    "real_count": f_real,
                    "labeled_samples": f_labeled,
                    "correct_samples": f_correct,
                    "accuracy": f_acc,
                }

                if idx_in_pending % save_interval == 0 or idx_in_pending == len(pending_in_folder):
                    summary = _build_summary(
                        results_dict, folder_summaries,
                        total_done, ai_cnt, real_cnt, labeled_cnt, correct_cnt, error_cnt,
                        split_name=split_name, test_dir=test_dir, weights_dir=weights_dir,
                    )
                    save_checkpoint_atomic(summary, include_backup=(total_done % 50 == 0))

                acc_info = f"Acc: {f_acc}%" if f_acc is not None else "Done"
                cum_acc_info = f"CumAcc: {running_acc}%" if running_acc is not None else ""
                pbar.set_postfix_str(f"{acc_info} | {cum_acc_info}")

            print(f"✅ Hoàn tất thư mục [{folder_counter}]: {rel_folder}")
            f_info = folder_summaries[rel_folder]
            if f_info["accuracy"] is not None:
                print(f"   📊 Accuracy thư mục này: {f_info['accuracy']}% ({f_info['correct_samples']}/{f_info['labeled_samples']} ảnh)")
            if summary.get("running_accuracy") is not None:
                print(f"   📈 Accuracy tích lũy toàn bộ phần này: {summary['running_accuracy']}% ({correct_cnt}/{labeled_cnt} ảnh)")

    finally:
        summary = _build_summary(
            results_dict, folder_summaries,
            total_done, ai_cnt, real_cnt, labeled_cnt, correct_cnt, error_cnt,
            split_name=split_name, test_dir=test_dir, weights_dir=weights_dir,
            compute_ap=True,
        )
        save_checkpoint_atomic(summary, include_backup=True)

    _save_json_atomic(output_json, summary)
    if write_local_copy:
        local_output = os.path.join(test_dir, os.path.basename(output_json))
        if local_output != output_json:
            _save_json_atomic(local_output, summary)

    running_acc = summary.get("running_accuracy")
    real_acc = summary.get("real_accuracy")
    ai_acc = summary.get("ai_accuracy")
    ap_val = summary.get("ap")
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT KIỂM THỬ [{split_name or test_dir}]: {total_done} ảnh")
    print(f"   • Dự đoán AI         : {ai_cnt} ({summary.get('ai_pct', 0)}%)")
    print(f"   • Dự đoán Real       : {real_cnt} ({summary.get('real_pct', 0)}%)")
    if error_cnt > 0:
        print(f"   • Ảnh lỗi bỏ qua     : {error_cnt}")
    if running_acc is not None:
        print(f"   • Accuracy phần này  : {running_acc}% ({summary.get('correct_samples')}/{summary.get('labeled_samples')} ảnh có nhãn)")
    if real_acc is not None:
        print(f"   • Real Accuracy (TN) : {real_acc}% ({summary.get('tn')}/{summary.get('tn', 0) + summary.get('fp', 0)})")
    if ai_acc is not None:
        print(f"   • AI Accuracy   (TP) : {ai_acc}% ({summary.get('tp')}/{summary.get('tp', 0) + summary.get('fn', 0)})")
    if ap_val is not None:
        print(f"   • Average Prec (AP)  : {ap_val}%")
    print(f"💾 File kết quả đã lưu tại: {output_json}")
    print("=" * 80)
    return summary


def run_test_split(
    split_key: str,
    test_dir: Optional[str] = None,
    drive_dir: str = DEFAULT_DRIVE_DIR,
    weights_dir: str = DEFAULT_WRITING_MODEL_DIR,
    train_dataset: str = "sd-v1_4",
    device: Optional[str] = None,
    engine: Optional["InferenceEngine"] = None,
    save_every: int = 10,
    require_weights: bool = True,
) -> Dict[str, Any]:
    """Run one of the 3 isolated test splits ('aigc', 'cospy_real', 'synthetic')."""
    if split_key not in TEST_SPLITS:
        raise ValueError(f"Unknown split '{split_key}'. Choose from: {list(TEST_SPLITS.keys())}")

    cfg = TEST_SPLITS[split_key]
    resolved_dir = resolve_split_test_dir(split_key, custom_dir=test_dir)
    os.makedirs(drive_dir, exist_ok=True)

    ckpt_path = os.path.join(drive_dir, cfg["checkpoint_file"])
    backup_path = os.path.join(drive_dir, cfg["backup_file"])
    out_path = os.path.join(drive_dir, cfg["results_file"])

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU TEST PHẦN [{split_key.upper()}]: {cfg['title']}")
    print(f"   • Thư mục ảnh   : {resolved_dir}")
    print(f"   • Model         : {weights_dir}")
    print(f"   • Checkpoint    : {ckpt_path}")
    print(f"   • File kết quả  : {out_path}")
    print("=" * 80)

    if engine is None:
        engine = load_writing_engine(
            weights_dir=weights_dir,
            train_dataset=train_dataset,
            device=device,
            require_weights=require_weights,
        )

    return test_images_in_directory(
        test_dir=resolved_dir,
        train_dataset=train_dataset,
        device=device,
        output_json=out_path,
        checkpoint_json=ckpt_path,
        backup_json=backup_path,
        weights_dir=weights_dir,
        fixed_label=cfg["fixed_label"],
        engine=engine,
        split_name=split_key,
        save_every=save_every,
    )


def summarize_all_splits(
    drive_dir: str = DEFAULT_DRIVE_DIR,
    save_summary: bool = True,
    weights_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Read individual results from the 3 splits on Drive and build a combined overview."""
    splits_overview: Dict[str, Any] = {}
    merged_results: Dict[str, Any] = {}
    grand_images = 0
    grand_ai = 0
    grand_real = 0
    grand_errors = 0
    grand_labeled = 0
    grand_correct = 0
    detected_weights_dir = weights_dir
    split_accs: List[float] = []

    print("=" * 92)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ 3 PHẦN DATA TEST ĐỘC LẬP (MODEL 'writing')")
    print("=" * 92)

    for idx, (split_key, cfg) in enumerate(TEST_SPLITS.items(), start=1):
        res_path = os.path.join(drive_dir, cfg["results_file"])
        ckpt_path = os.path.join(drive_dir, cfg["checkpoint_file"])
        source_path = res_path if os.path.exists(res_path) else (ckpt_path if os.path.exists(ckpt_path) else None)
        status = "completed" if os.path.exists(res_path) else ("in_progress" if source_path else "not_started")

        if source_path is None:
            print(f"   • Phần {idx} [{split_key:<10}] ({cfg['title']}): ⏳ Chưa chạy")
            splits_overview[split_key] = {"title": cfg["title"], "status": "not_started"}
            continue

        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not detected_weights_dir and data.get("weights_dir"):
            detected_weights_dir = data.get("weights_dir")

        split_res = data.get("results", {})
        for r_k, r_v in split_res.items():
            merged_results[f"{split_key}/{r_k}"] = r_v

        cls_m = _compute_classification_metrics(split_res, compute_ap=True)
        n_img = data.get("completed_images", len(split_res))
        n_ai = data.get("ai_count", 0)
        n_real = data.get("real_count", 0)
        n_err = data.get("error_count", 0)
        n_lbl = cls_m["labeled_samples"] if split_res else data.get("labeled_samples", 0)
        n_cor = cls_m["correct_samples"] if split_res else data.get("correct_samples", 0)
        acc = cls_m["accuracy"] if cls_m["accuracy"] is not None else data.get("running_accuracy")

        grand_images += n_img
        grand_ai += n_ai
        grand_real += n_real
        grand_errors += n_err
        grand_labeled += n_lbl
        grand_correct += n_cor
        if acc is not None:
            split_accs.append(acc)

        acc_str = f"{acc}%" if acc is not None else "N/A"
        tag = "✅" if status == "completed" else "🔄"
        extra_parts = []
        if cls_m["real_accuracy"] is not None:
            extra_parts.append(f"Real:{cls_m['real_accuracy']}%")
        if cls_m["ai_accuracy"] is not None:
            extra_parts.append(f"AI:{cls_m['ai_accuracy']}%")
        if cls_m["ap"] is not None:
            extra_parts.append(f"AP:{cls_m['ap']}%")
        extra_str = f" | {' '.join(extra_parts)}" if extra_parts else ""
        print(
            f"   • {tag} Phần {idx} [{split_key:<10}] {cfg['title']:<32} | "
            f"Ảnh: {n_img:>6} | AI: {n_ai:>6} | Real: {n_real:>6} | "
            f"Acc: {acc_str:>7} ({n_cor}/{n_lbl}){extra_str}"
        )

        splits_overview[split_key] = {
            "title": cfg["title"],
            "status": status,
            "source_file": source_path,
            "completed_images": n_img,
            "error_count": n_err,
            "ai_count": n_ai,
            "real_count": n_real,
            "labeled_samples": n_lbl,
            "correct_samples": n_cor,
            "accuracy": acc,
            "real_accuracy": cls_m["real_accuracy"],
            "ai_accuracy": cls_m["ai_accuracy"],
            "ap": cls_m["ap"],
            "tp": cls_m["tp"],
            "tn": cls_m["tn"],
            "fp": cls_m["fp"],
            "fn": cls_m["fn"],
        }

    overall_acc = round(grand_correct / grand_labeled * 100, 2) if grand_labeled > 0 else None
    macro_acc = round(sum(split_accs) / len(split_accs), 2) if split_accs else None
    global_m = _compute_classification_metrics(merged_results, compute_ap=True)

    print("-" * 92)
    print(
        f"   🏆 ACC TỔNG (Gộp 3 phần) : {overall_acc if overall_acc is not None else 'N/A'}% "
        f"({grand_correct}/{grand_labeled} ảnh có nhãn) | Tổng ảnh: {grand_images} (AI: {grand_ai} | Real: {grand_real})"
    )
    if macro_acc is not None:
        print(f"   📐 Trung bình đều {len(split_accs)} phần : {macro_acc}% (Macro Avg)")
    if global_m["real_accuracy"] is not None or global_m["ai_accuracy"] is not None:
        r_s = f"{global_m['real_accuracy']}%" if global_m["real_accuracy"] is not None else "N/A"
        a_s = f"{global_m['ai_accuracy']}%" if global_m["ai_accuracy"] is not None else "N/A"
        ap_s = f" | AP tổng: {global_m['ap']}%" if global_m["ap"] is not None else ""
        print(
            f"   ⚖️  Chi tiết toàn cục    : Real Acc={r_s} | AI Acc={a_s}{ap_s} | "
            f"TP={global_m['tp']} TN={global_m['tn']} FP={global_m['fp']} FN={global_m['fn']}"
        )
    print("=" * 92)

    combined = {
        "weights_dir": detected_weights_dir or DEFAULT_WRITING_MODEL_DIR,
        "total_images": grand_images,
        "total_errors": grand_errors,
        "total_ai": grand_ai,
        "total_real": grand_real,
        "total_labeled": grand_labeled,
        "total_correct": grand_correct,
        "overall_accuracy": overall_acc,
        "macro_accuracy_3_splits": macro_acc,
        "overall_real_accuracy": global_m["real_accuracy"],
        "overall_ai_accuracy": global_m["ai_accuracy"],
        "overall_ap": global_m["ap"],
        "tp": global_m["tp"],
        "tn": global_m["tn"],
        "fp": global_m["fp"],
        "fn": global_m["fn"],
        "splits": splits_overview,
    }
    if save_summary:
        out_summary = os.path.join(drive_dir, "test_results_summary.json")
        _save_json_atomic(out_summary, combined)
        out_csv = os.path.join(drive_dir, "test_results_summary.csv")
        try:
            os.makedirs(drive_dir, exist_ok=True)
            write_target = out_csv if _is_google_drive_path(out_csv) else (out_csv + ".tmp")
            with open(write_target, "w", newline="", encoding="utf-8-sig") as cf:
                writer = csv.writer(cf)
                writer.writerow(["split", "title", "status", "images", "ai_pred", "real_pred", "labeled", "correct", "accuracy_pct", "real_acc_pct", "ai_acc_pct", "ap_pct"])
                for sk, sv in splits_overview.items():
                    writer.writerow([
                        sk, sv.get("title", ""), sv.get("status", ""),
                        sv.get("completed_images", 0), sv.get("ai_count", 0), sv.get("real_count", 0),
                        sv.get("labeled_samples", 0), sv.get("correct_samples", 0),
                        sv.get("accuracy", ""), sv.get("real_accuracy", ""), sv.get("ai_accuracy", ""), sv.get("ap", ""),
                    ])
                writer.writerow([
                    "OVERALL", "Combined 3 Splits", "summary",
                    grand_images, grand_ai, grand_real, grand_labeled, grand_correct,
                    overall_acc if overall_acc is not None else "",
                    global_m["real_accuracy"] if global_m["real_accuracy"] is not None else "",
                    global_m["ai_accuracy"] if global_m["ai_accuracy"] is not None else "",
                    global_m["ap"] if global_m["ap"] is not None else "",
                ])
            if write_target != out_csv:
                os.replace(write_target, out_csv)
            print(f"💾 Đã lưu bảng tổng hợp tại:\n   • JSON: {out_summary}\n   • CSV : {out_csv}")
        except Exception:
            print(f"💾 Đã lưu bảng tổng hợp tại: {out_summary}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="CO-SPY Google Drive Sync and Testing Tool")
    subparsers = parser.add_subparsers(dest="command")

    up_parser = subparsers.add_parser("upload")
    up_parser.add_argument("--dest_path", type=str, required=True, help="Google Drive path")

    dl_parser = subparsers.add_parser("download")
    dl_parser.add_argument("--drive_url", type=str, required=True)
    dl_parser.add_argument("--output_dir", type=str, default="Dataset/data/test")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--drive_url", type=str, default=None)
    test_parser.add_argument("--split", type=str, default=None, choices=["aigc", "cospy_real", "synthetic", "all"])
    test_parser.add_argument("--test_dir", type=str, default=None)
    test_parser.add_argument("--drive_dir", type=str, default=DEFAULT_DRIVE_DIR)
    test_parser.add_argument("--weights_dir", type=str, default=DEFAULT_WRITING_MODEL_DIR)
    test_parser.add_argument("--train_dataset", type=str, default="sd-v1_4", choices=["sd-v1_4", "progan"])
    test_parser.add_argument("--device", type=str, default=None)
    test_parser.add_argument("--output_json", type=str, default=None)
    test_parser.add_argument("--save_every", type=int, default=10)

    sum_parser = subparsers.add_parser("summary")
    sum_parser.add_argument("--drive_dir", type=str, default=DEFAULT_DRIVE_DIR)

    args = parser.parse_args()
    if args.command == "upload":
        upload_via_local_drive_path(args.dest_path)
    elif args.command == "download":
        download_test_dataset_from_drive(args.drive_url, args.output_dir)
    elif args.command == "summary":
        summarize_all_splits(drive_dir=args.drive_dir)
    elif args.command == "test":
        if args.split:
            engine = load_writing_engine(
                weights_dir=args.weights_dir,
                train_dataset=args.train_dataset,
                device=args.device,
                require_weights=True,
            )
            keys = ["aigc", "cospy_real", "synthetic"] if args.split == "all" else [args.split]
            for k in keys:
                run_test_split(
                    split_key=k,
                    test_dir=args.test_dir if len(keys) == 1 else None,
                    drive_dir=args.drive_dir,
                    weights_dir=args.weights_dir,
                    train_dataset=args.train_dataset,
                    device=args.device,
                    engine=engine,
                    save_every=args.save_every,
                )
            if args.split == "all":
                summarize_all_splits(drive_dir=args.drive_dir, weights_dir=args.weights_dir)
        else:
            t_dir = args.test_dir or "Dataset/data/test"
            if args.drive_url:
                download_test_dataset_from_drive(args.drive_url, t_dir)
            test_images_in_directory(
                test_dir=t_dir,
                train_dataset=args.train_dataset,
                device=args.device,
                output_json=args.output_json,
                weights_dir=args.weights_dir if has_valid_weights_dir(args.weights_dir) else None,
                save_every=args.save_every,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

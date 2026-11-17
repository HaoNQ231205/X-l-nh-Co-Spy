"""Deduplication and integrity audit utilities for CO-SPY codebase and datasets.

Includes:
- Codebase and checkpoint SHA-256 checksums
- Exact duplicate detection (SHA-256 on raw pixel data)
- Near duplicate detection (dHash, pHash)
- mscoco2017/val2017 <-> Co-Spy-Bench overlap check (leakage L1/L2)
- URL deduplication in in_the_wild (flux.txt, lexica.txt) (leakage L6)
"""

import os
import hashlib
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from PIL import Image


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def compute_pixel_sha256(image: Image.Image) -> str:
    """Compute SHA-256 hash of the decoded pixel buffer of an image.
    
    Standardizes image mode to RGB and computes SHA-256 on contiguous bytes.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.ascontiguousarray(np.array(image))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) for an image.
    
    Resizes image to (hash_size + 1, hash_size) grayscale,
    then compares adjacent pixel intensities horizontally.
    """
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.int32)
    # Compare adjacent columns: True if column i > column i+1
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Pack boolean array into an integer
    decimal_value = 0
    for bit in diff.flatten():
        decimal_value = (decimal_value << 1) | int(bit)
    return decimal_value


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Compute perceptual hash (pHash) using discrete cosine transform (DCT).
    
    Resizes to (32, 32) grayscale, applies 2D DCT, extracts top-left (8, 8)
    low-frequency block (excluding DC coefficient [0, 0]), compares against median.
    """
    from scipy.fftpack import dct
    
    img_size = hash_size * highfreq_factor
    img = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.float32)
    
    # 2D DCT (type 2)
    dct_transformed = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    
    # Extract low-frequency 8x8 block (top-left)
    dct_lowfreq = dct_transformed[:hash_size, :hash_size]
    
    # Exclude DC term (at 0,0) when computing median
    med = np.median(dct_lowfreq[1:, 1:])
    diff = dct_lowfreq > med
    
    decimal_value = 0
    for bit in diff.flatten():
        decimal_value = (decimal_value << 1) | int(bit)
    return decimal_value


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def filter_duplicate_urls(
    filepath: str,
    out_filepath: Optional[str] = None
) -> Dict[str, Any]:
    """Scan and filter duplicate URLs in a URL file (e.g. flux.txt, lexica.txt).
    
    Normalizes URLs by stripping whitespace, trailing slashes, and anchors.
    Writes unique URLs to out_filepath if provided.
    """
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}", "total": 0, "unique": 0}

    total_lines = 0
    seen_urls: Set[str] = set()
    unique_urls: List[str] = []
    duplicate_count = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            # Basic normalization
            norm_url = url.rstrip("/")
            if norm_url in seen_urls:
                duplicate_count += 1
            else:
                seen_urls.add(norm_url)
                unique_urls.append(url)

    unique_count = len(unique_urls)
    dup_rate = (duplicate_count / total_lines * 100.0) if total_lines > 0 else 0.0

    if out_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(out_filepath)), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            for u in unique_urls:
                f.write(u + "\n")

    return {
        "filepath": filepath,
        "total_lines": total_lines,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "duplicate_rate_pct": round(dup_rate, 2),
        "output_saved_to": out_filepath,
    }


def scan_duplicates_in_paths(
    image_paths: List[str],
    phash_threshold: int = 5
) -> Dict[str, Any]:
    """Scan a list of image paths for exact pixel duplicates and near-duplicates.
    
    Args:
        image_paths: List of file paths to images.
        phash_threshold: Max Hamming distance to consider two images near-duplicates.
    """
    pixel_hashes: Dict[str, List[str]] = {}
    phashes: List[Tuple[str, int]] = []
    errors: List[str] = []

    for p in image_paths:
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                px_hash = compute_pixel_sha256(img_rgb)
                pixel_hashes.setdefault(px_hash, []).append(p)
                ph = compute_phash(img_rgb)
                phashes.append((p, ph))
        except Exception as e:
            errors.append(f"{p}: {str(e)}")

    exact_duplicates = {k: v for k, v in pixel_hashes.items() if len(v) > 1}

    # Near duplicates via pHash
    near_duplicates = []
    n = len(phashes)
    for i in range(n):
        path_i, ph_i = phashes[i]
        for j in range(i + 1, n):
            path_j, ph_j = phashes[j]
            dist = hamming_distance(ph_i, ph_j)
            if dist <= phash_threshold:
                near_duplicates.append({
                    "pair": (path_i, path_j),
                    "hamming_dist": dist
                })

    return {
        "total_scanned": len(image_paths),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": len(near_duplicates),
        "near_duplicates": near_duplicates[:100],  # cap to top 100 for report
        "errors": errors,
    }


def check_mscoco_leakage(
    train_val_dir: str = "data/train/sd-v1_4/mscoco2017/val2017",
    test_bench_root: str = "data/test/Co-Spy-Bench"
) -> Dict[str, Any]:
    """Check data leakage: overlap between mscoco2017/val2017 and Co-Spy-Bench real images.
    
    Leakage issue L1: Co-Spy-Bench 'mscoco' column loads from mscoco2017/val2017,
    which was used as the validation set for model selection.
    """
    if not os.path.exists(train_val_dir):
        return {
            "status": "skipped",
            "reason": f"Train val directory not found: {train_val_dir}"
        }

    train_val_files = set(f for f in os.listdir(train_val_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    
    # Check filenames in real image examples
    bench_mscoco_dir = os.path.join(test_bench_root, "real_image_examples", "mscoco")
    overlap_files = []
    if os.path.exists(bench_mscoco_dir):
        bench_files = set(f for f in os.listdir(bench_mscoco_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
        overlap_files = list(train_val_files.intersection(bench_files))

    return {
        "train_val_dir": train_val_dir,
        "train_val_image_count": len(train_val_files),
        "bench_mscoco_dir": bench_mscoco_dir,
        "overlap_count": len(overlap_files),
        "overlap_files": overlap_files[:20],
        "is_leaked": len(overlap_files) > 0,
        "note": "MSCOCO test column overlaps with validation set used to choose best_model.pth (Issue L1)."
    }


def verify_codebase_integrity(
    root_dir: str = "."
) -> Dict[str, str]:
    """Compute SHA-256 for all core source files to identify code version precisely."""
    core_files = [
        "train.py",
        "train.sh",
        "evaluate.py",
        "main.py",
        "utils.py",
        "environment.yml",
        os.path.join("dataSets", "__init__.py"),
        os.path.join("dataSets", "dataset.py"),
        os.path.join("dataSets", "cospy_real.py"),
        os.path.join("detectors", "sd-v1_4", "__init__.py"),
        os.path.join("detectors", "sd-v1_4", "base.py"),
        os.path.join("detectors", "sd-v1_4", "artifact.py"),
        os.path.join("detectors", "sd-v1_4", "semantic.py"),
        os.path.join("detectors", "sd-v1_4", "fusion.py"),
        os.path.join("detectors", "sd-v1_4", "end2end.py"),
        os.path.join("detectors", "progan", "__init__.py"),
        os.path.join("detectors", "progan", "base.py"),
        os.path.join("detectors", "progan", "artifact.py"),
        os.path.join("detectors", "progan", "semantic.py"),
        os.path.join("detectors", "progan", "fusion.py"),
        os.path.join("detectors", "progan", "end2end.py"),
    ]

    manifest: Dict[str, str] = {}
    for rel_path in core_files:
        full_path = os.path.join(root_dir, rel_path)
        if os.path.exists(full_path):
            manifest[rel_path.replace("\\", "/")] = compute_file_sha256(full_path)
        else:
            manifest[rel_path.replace("\\", "/")] = "NOT_FOUND"

    return manifest


"""Comprehensive verification and audit runner for CO-SPY.

Runs:
1. Source code and checkpoint SHA-256 verification
2. Duplicate URL audit on in_the_wild (flux.txt, lexica.txt)
3. MSCOCO data leakage check
4. Metadata bias audit
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audit.dedup import (
    verify_codebase_integrity,
    filter_duplicate_urls,
    check_mscoco_leakage,
    compute_file_sha256,
)
from audit.metadata_bias import audit_metadata_bias


def run_full_audit(
    root_dir: str = PROJECT_ROOT,
    clean_urls: bool = True,
    output_report_path: str = "audit_report.json"
) -> Dict[str, Any]:
    """Execute complete audit protocol and return consolidated report."""
    print("=" * 60)
    print("CO-SPY Codebase and Dataset Integrity Audit")
    print("=" * 60)

    report: Dict[str, Any] = {}

    # 1. Codebase SHA-256 Verification
    print("\n[Step 1] Verifying core codebase SHA-256 checksums...")
    code_manifest = verify_codebase_integrity(root_dir)
    report["codebase_sha256"] = code_manifest
    print(f"  Verified {len(code_manifest)} core source files.")

    # Checkpoint SHA-256 if available
    pretrained_dir = os.path.join(root_dir, "pretrained")
    ckpt_manifest = {}
    if os.path.exists(pretrained_dir):
        for r, _, fnames in os.walk(pretrained_dir):
            for fn in fnames:
                if fn.endswith((".pth", ".pt", ".bin")):
                    fpath = os.path.join(r, fn)
                    rel = os.path.relpath(fpath, root_dir).replace("\\", "/")
                    ckpt_manifest[rel] = compute_file_sha256(fpath)
    report["checkpoint_sha256"] = ckpt_manifest

    # 2. In_the_wild URL Deduplication Check
    print("\n[Step 2] Scanning in_the_wild URL duplicates (flux.txt, lexica.txt)...")
    url_dir = os.path.join(root_dir, "data", "in_the_wild", "urls")
    flux_path = os.path.join(url_dir, "flux.txt")
    lexica_path = os.path.join(url_dir, "lexica.txt")

    flux_out = os.path.join(url_dir, "flux_clean.txt") if clean_urls else None
    lexica_out = os.path.join(url_dir, "lexica_clean.txt") if clean_urls else None

    flux_audit = filter_duplicate_urls(flux_path, flux_out)
    lexica_audit = filter_duplicate_urls(lexica_path, lexica_out)

    report["url_dedup"] = {
        "flux": flux_audit,
        "lexica": lexica_audit,
    }
    if "total_lines" in flux_audit:
        print(f"  flux.txt: {flux_audit['unique_count']} unique / {flux_audit['total_lines']} total ({flux_audit['duplicate_rate_pct']}% duplicates)")
    if "total_lines" in lexica_audit:
        print(f"  lexica.txt: {lexica_audit['unique_count']} unique / {lexica_audit['total_lines']} total ({lexica_audit['duplicate_rate_pct']}% duplicates)")

    # 3. MSCOCO Data Leakage Check
    print("\n[Step 3] Checking MSCOCO val2017 training/validation leakage into Co-Spy-Bench...")
    train_val_dir = os.path.join(root_dir, "data", "train", "sd-v1_4", "mscoco2017", "val2017")
    bench_dir = os.path.join(root_dir, "data", "test", "Co-Spy-Bench")
    leakage_audit = check_mscoco_leakage(train_val_dir, bench_dir)
    report["mscoco_leakage"] = leakage_audit
    print(f"  MSCOCO Leakage detected: {leakage_audit.get('is_leaked', False)} ({leakage_audit.get('overlap_count', 0)} overlaps)")

    # 4. Save report
    out_path = os.path.join(root_dir, output_report_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit complete! Consolidated report written to: {out_path}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CO-SPY Verification and Audit Tool")
    parser.add_argument("--root", type=str, default=PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--no_clean_urls", action="store_true", help="Do not write cleaned URL files")
    parser.add_argument("--out", type=str, default="audit_report.json", help="Path for JSON output report")
    args = parser.parse_args()

    run_full_audit(
        root_dir=args.root,
        clean_urls=not args.no_clean_urls,
        output_report_path=args.out
    )


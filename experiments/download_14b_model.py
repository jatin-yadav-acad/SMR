#!/usr/bin/env python
"""
download_14b_model.py
=====================
Downloads the pre-quantized 14B model (unsloth/Qwen2.5-14B-bnb-4bit, 9.27 GB)
for scaling SMR-LoRA to the 14B parameter foundation model tier.
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

def main():
    print("=" * 80)
    print("  DOWNLOADING QWEN2.5-14B (4-BIT / 8-BIT BNB QUANTIZED, 9.27 GB)")
    print("=" * 80)
    start = time.time()
    try:
        path = snapshot_download(
            repo_id="unsloth/Qwen2.5-14B-bnb-4bit",
            resume_download=True,
            max_workers=4
        )
        elapsed = time.time() - start
        print(f"\n[OK] Qwen2.5-14B successfully downloaded in {elapsed:.1f}s ({elapsed/60:.1f}m)!")
        print(f"Path: {path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

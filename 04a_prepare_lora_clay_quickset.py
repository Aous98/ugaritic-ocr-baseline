#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a small LoRA training dataset from clay style crops.
This is a quick smoke-test dataset (≈300 images).
"""

import os
import random
from pathlib import Path
from PIL import Image

PROJECT_ROOT = "/home/aous/Desktop/research/2025/synth"

SRC_DIR = os.path.join(PROJECT_ROOT, "data/02_style_refs/clay")
OUT_IMG = os.path.join(PROJECT_ROOT, "data/04_lora/clay/images")
OUT_CAP = os.path.join(PROJECT_ROOT, "data/04_lora/clay/captions")

N_SAMPLES = 300
TARGET_SIZE = 512
SEED = 123

CAPTION = (
    "ugaritic_clay_style, ancient clay tablet texture, "
    "carved inscription, archaeological photograph"
)

def main():
    random.seed(SEED)
    Path(OUT_IMG).mkdir(parents=True, exist_ok=True)
    Path(OUT_CAP).mkdir(parents=True, exist_ok=True)

    files = [
        f for f in os.listdir(SRC_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not files:
        raise RuntimeError(f"No images found in {SRC_DIR}")

    random.shuffle(files)
    files = files[:min(N_SAMPLES, len(files))]

    # clear previous outputs
    for d in (OUT_IMG, OUT_CAP):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    for i, fname in enumerate(files):
        img = Image.open(os.path.join(SRC_DIR, fname)).convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)

        out_name = f"clay_{i:05d}.png"
        img.save(os.path.join(OUT_IMG, out_name), optimize=True)

        with open(
            os.path.join(OUT_CAP, out_name.replace(".png", ".txt")),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(CAPTION + "\n")

    print("[OK] LoRA quick dataset prepared")
    print("Images:", len(files))
    print("Path:", OUT_IMG)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np


# =========================
# PROJECT PATHS
# =========================
PROJECT_ROOT = "/home/aous/Desktop/research/2025/synth"

IN_DIR = os.path.join(PROJECT_ROOT, "real imgs")

OUT_ROOT = os.path.join(PROJECT_ROOT, "data/02_style_refs")
OUT_CLAY = os.path.join(OUT_ROOT, "clay")
OUT_HAND = os.path.join(OUT_ROOT, "hand")
OUT_META = os.path.join(OUT_ROOT, "_meta")
# =========================


# =========================
# CROP SETTINGS
# =========================
CROP_SIZE = 512

# How many crops per source image
CROPS_PER_CLAY_IMG = 800
CROPS_PER_HAND_IMG = 600

# Clay: how to split into texture vs context
CLAY_TEXTURE_RATIO = 0.70
CLAY_CONTEXT_RATIO = 0.30

# Clay edge thresholds (Sobel magnitude mean)
# If edge_mean <= TEXTURE_MAX -> "texture"
# If CONTEXT_MIN <= edge_mean <= CONTEXT_MAX -> "context"
CLAY_EDGE_TEXTURE_MAX = 60.0
CLAY_EDGE_CONTEXT_MIN = 60.0
CLAY_EDGE_CONTEXT_MAX = 220.0

# Hand/facsimile acceptance thresholds:
# - nonblack_frac: fraction of pixels with gray >= 25 (not near-black)
# - stroke_frac: fraction of pixels in Otsu-thresholded binary (bw > 0)
HAND_MIN_NONBLACK_FRAC = 0.0010   # 0.10% pixels not near-black
HAND_MIN_STROKE_FRAC   = 0.0008   # 0.08% pixels classified as strokes

# Resize policy: if image is too small for crop, upscale so min(H,W) >= CROP_SIZE
UPSCALE_IF_NEEDED = True

SEED = 42
MAX_TRIES_PER_CROP = 120
# =========================


# =========================
# FILE SELECTION / STYLES
# =========================
# We classify by filename prefix; adjust only if you rename files.
# - KTU* -> hand (facsimile)
# - everything else in the folder -> clay
# We explicitly ignore any file containing "ax rs" (you deleted it anyway).
IGNORE_PATTERNS = [
    r"ax\s*rs",   # ignore ax style if present
]
# =========================


def ensure_dirs():
    for d in [OUT_CLAY, OUT_HAND, OUT_META]:
        Path(d).mkdir(parents=True, exist_ok=True)


def sanitize_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.replace("\n", "_")
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "", stem)
    return stem[:80] if stem else "img"


def should_ignore(path: str) -> bool:
    base = os.path.basename(path).lower()
    for pat in IGNORE_PATTERNS:
        if re.search(pat, base):
            return True
    return False


def classify_style(path: str) -> str:
    base = os.path.basename(path).lower()
    if base.startswith("ktu"):
        return "hand"
    return "clay"


def edge_mean(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag))


def nonblack_fraction(gray: np.ndarray, thr: int = 25) -> float:
    # pixels not near-black
    return float((gray >= thr).mean())


def stroke_fraction_otsu(gray: np.ndarray) -> float:
    # Otsu threshold robustly separates strokes/background even if strokes are not pure white
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((bw > 0).mean())


def maybe_upscale(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= crop_size:
        return img
    if not UPSCALE_IF_NEEDED:
        return img
    scale = crop_size / float(min(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def random_crop(img: np.ndarray, crop_size: int):
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        return None, None
    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    crop = img[y:y + crop_size, x:x + crop_size].copy()
    return crop, (x, y, x + crop_size, y + crop_size)


def accept_hand_crop(crop_rgb: np.ndarray) -> tuple[bool, dict]:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2GRAY)

    nbf = nonblack_fraction(gray, thr=25)
    sf = stroke_fraction_otsu(gray)

    info = {
        "nonblack_frac": nbf,
        "stroke_frac": sf,
    }

    ok = (nbf >= HAND_MIN_NONBLACK_FRAC) and (sf >= HAND_MIN_STROKE_FRAC)
    return ok, info


def accept_clay_crop(crop_rgb: np.ndarray) -> tuple[bool, dict]:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2GRAY)
    em = edge_mean(gray)
    info = {"edge_mean": em}
    return True, info


def main():
    random.seed(SEED)
    ensure_dirs()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(IN_DIR, e)))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found in: {IN_DIR}")

    manifest = []
    total_written = 0

    for p in paths:
        if should_ignore(p):
            print(f"[SKIP] ignored by pattern: {os.path.basename(p)}")
            continue

        style = classify_style(p)
        stem = sanitize_stem(p)

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cannot read: {p}")
            continue

        img = maybe_upscale(img, CROP_SIZE)

        if style == "clay":
            n_total = CROPS_PER_CLAY_IMG
            n_texture = int(round(n_total * CLAY_TEXTURE_RATIO))
            n_context = n_total - n_texture
            targets = [("texture", n_texture), ("context", n_context)]
        else:
            n_total = CROPS_PER_HAND_IMG
            targets = [("any", n_total)]

        for mode, n_need in targets:
            written = 0
            attempts = 0

            while written < n_need and attempts < n_need * MAX_TRIES_PER_CROP:
                attempts += 1

                crop, xyxy = random_crop(img, CROP_SIZE)
                if crop is None:
                    break

                if style == "hand":
                    ok, info = accept_hand_crop(crop)
                    if not ok:
                        continue

                    out_dir = OUT_HAND
                    out_name = f"{stem}_{mode}_{written:05d}.png"
                    out_path = os.path.join(out_dir, out_name)
                    cv2.imwrite(out_path, crop)

                else:
                    ok, info = accept_clay_crop(crop)
                    em = info["edge_mean"]

                    if mode == "texture":
                        ok = (em <= CLAY_EDGE_TEXTURE_MAX)
                    elif mode == "context":
                        ok = (CLAY_EDGE_CONTEXT_MIN <= em <= CLAY_EDGE_CONTEXT_MAX)
                    else:
                        ok = True

                    if not ok:
                        continue

                    out_dir = OUT_CLAY
                    out_name = f"{stem}_{mode}_{written:05d}.png"
                    out_path = os.path.join(out_dir, out_name)
                    cv2.imwrite(out_path, crop)

                manifest.append({
                    "out_path": out_path,
                    "source_path": p,
                    "style": style,
                    "mode": mode,
                    "crop_xyxy": [int(v) for v in xyxy],
                    **info
                })

                written += 1
                total_written += 1

            print(f"[OK] {os.path.basename(p)} | style={style} mode={mode} -> {written}/{n_need} crops")

    meta_path = os.path.join(OUT_META, "crops_manifest.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Input dir: {IN_DIR}")
    print(f"Output: {OUT_ROOT}")
    print(f"Manifest: {meta_path}")
    print(f"Total crops written: {total_written}")


if __name__ == "__main__":
    main()


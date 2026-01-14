#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_binarize(crop: np.ndarray) -> np.ndarray:
    if not HAS_CV2:
        return (crop < 128).astype(np.uint8) * 255
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def to_square_and_resize(crop: np.ndarray, size: int) -> Image.Image:
    # crop must be a non-empty numpy array
    if crop is None or not hasattr(crop, "shape") or crop.size == 0:
        raise ValueError("Empty/invalid crop")

    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        raise ValueError("Too-small crop")

    s = max(h, w)
    canvas = np.ones((s, s), dtype=np.uint8) * 255
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    pil = Image.fromarray(canvas)
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return pil


def build_model(num_classes: int, arch: str):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
    elif arch == "resnet34":
        m = models.resnet34(weights=None)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError(arch)

    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best_model.pt")
    ap.add_argument("--image", required=True, help="page_XXXXXX.png")
    ap.add_argument("--json", required=True, help="page_XXXXXX.json")
    ap.add_argument("--out", required=True, help="Output visualization PNG")
    ap.add_argument("--font", default="/usr/share/fonts/truetype/noto/NotoSansUgaritic-Regular.ttf")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    char2id = ckpt["char2id"]
    id2char = {i:c for c,i in char2id.items()}
    cfg = ckpt["cfg"]
    crop_size = int(cfg.get("crop_size", 96))
    pad_px = int(cfg.get("pad_px", 6))
    binarize = bool(cfg.get("binarize", True))
    arch = cfg.get("arch", "resnet34")

    model = build_model(num_classes=len(char2id), arch=arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_pil = Image.open(args.image).convert("RGB")
    img_gray = np.array(Image.open(args.image).convert("L"))
    data = read_json(args.json)

    tf = transforms.ToTensor()
    draw = ImageDraw.Draw(img_pil)

    # font for predicted labels
    try:
        font = ImageFont.truetype(args.font, 18)
    except Exception:
        font = ImageFont.load_default()

    preds_lines = []

    for line in data["lines"]:
        pred_chars = []
        for ch in line["chars"]:
            x1,y1,x2,y2 = map(int, ch["bbox_xyxy"])
            # pad
            x1p = max(0, x1 - pad_px); y1p = max(0, y1 - pad_px)
            x2p = min(img_gray.shape[1], x2 + pad_px); y2p = min(img_gray.shape[0], y2 + pad_px)

            H, W = img_gray.shape[:2]

# Clip bbox to image bounds
            x1, y1, x2, y2 = map(int, ch["bbox_xyxy"])
            x1p = max(0, x1 - pad_px); y1p = max(0, y1 - pad_px)
            x2p = min(W, x2 + pad_px); y2p = min(H, y2 + pad_px)

# Skip invalid boxes
            if x2p <= x1p or y2p <= y1p:
                continue

            crop = img_gray[y1p:y2p, x1p:x2p]

# Skip empty crops (can happen if bbox is bad)
            if crop is None or crop.size == 0:
                continue

# Optional: skip tiny crops (garbage)
            if crop.shape[0] < 3 or crop.shape[1] < 3:
                continue

            if binarize:
                crop = maybe_binarize(crop)

            try:
                crop_pil = to_square_and_resize(crop, crop_size).convert("L")
            except ValueError:
                continue

            x = tf(crop_pil).unsqueeze(0).to(device)  # 1x1xHxW
            with torch.no_grad():
                pid = int(torch.argmax(model(x), dim=1).item())
            pred = id2char[pid]
            pred_chars.append(pred)

            # draw bbox and predicted label
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            draw.text((x1, max(0, y1-18)), pred, fill=(255,0,0), font=font)

        preds_lines.append("".join(pred_chars))

    # Save visualization
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img_pil.save(args.out)

    # Also print first 2 lines to terminal for sanity
    print("[PRED line0]", preds_lines[0] if preds_lines else "")
    print("[GT   line0]", data["lines"][0]["text"] if data.get("lines") else "")


if __name__ == "__main__":
    main()


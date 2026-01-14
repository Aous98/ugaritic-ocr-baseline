#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast OCR baseline on TEMPLATE images using provided JSON bboxes.

Pipeline:
- Read template PNG + JSON (with per-char bbox + label)
- Crop glyphs using bbox (perfect segmentation)
- Train a small classifier (ResNet-18) on glyph crops
- Decode in JSON order
- Compute metrics: char accuracy + CER (Levenshtein)

This script SPLITS pages into train/val/test (by PNG+JSON pairs):
- train_frac (default 0.8)
- val_frac   (default 0.1)
- test_frac  (remaining)

Your dataset:
- images: /home/aous/Desktop/research/2025/synth/data/01_templates/images/page_*.png
- labels: /home/aous/Desktop/research/2025/synth/data/01_templates/labels/page_*.json
- page size: 1024x1024 (OK)
"""

import os
import json
import glob
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

# Optional: OpenCV for better Otsu thresholding
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    images_dir: str
    json_dir: str
    out_dir: str

    seed: int = 42
    epochs: int = 8
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    pad_px: int = 6
    crop_size: int = 96
    binarize: bool = True

    train_frac: float = 0.8
    val_frac: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, gt) / len(gt)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_gray(path: str) -> np.ndarray:
    # 1024x1024 is fine; we crop using bbox
    img = Image.open(path).convert("L")
    return np.array(img)


def list_pages(images_dir: str, json_dir: str) -> List[Tuple[str, str]]:
    """
    Match page_*.png with page_*.json by basename.
    """
    pngs = sorted(glob.glob(os.path.join(images_dir, "page_*.png")))
    pairs = []
    for p in pngs:
        base = os.path.splitext(os.path.basename(p))[0]  # e.g. page_000000
        j = os.path.join(json_dir, base + ".json")
        if os.path.exists(j):
            pairs.append((p, j))
    return pairs


def crop_with_padding(img: np.ndarray, bbox_xyxy: List[int], pad: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    return img[y1:y2, x1:x2]


def maybe_binarize(crop: np.ndarray) -> np.ndarray:
    # Template domain: black glyph on white background
    if not HAS_CV2:
        return (crop < 128).astype(np.uint8) * 255
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def to_square_and_resize(crop: np.ndarray, size: int) -> Image.Image:
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.ones((s, s), dtype=np.uint8) * 255
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    pil = Image.fromarray(canvas)
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return pil


# -------------------------
# Dataset objects
# -------------------------
@dataclass
class Sample:
    page_id: str
    line_idx: int
    char_idx: int
    gt_char: str
    crop_pil: Image.Image


def extract_samples(png_path: str, json_path: str, cfg: CFG) -> Tuple[List[Sample], Dict]:
    page_id = os.path.splitext(os.path.basename(png_path))[0]
    img = load_image_gray(png_path)
    data = read_json(json_path)

    samples: List[Sample] = []
    for line in data["lines"]:
        li = int(line["line_idx"])
        for ch in line["chars"]:
            ci = int(ch["char_idx"])
            gt = ch["char"]
            bbox = ch["bbox_xyxy"]
            crop = crop_with_padding(img, bbox, cfg.pad_px)
            if cfg.binarize:
                crop = maybe_binarize(crop)
            crop_pil = to_square_and_resize(crop, cfg.crop_size)
            samples.append(Sample(page_id, li, ci, gt, crop_pil))
    return samples, data


class GlyphDataset(Dataset):
    def __init__(self, samples: List[Sample], char2id: Dict[str, int], is_train: bool, cfg: CFG):
        self.samples = samples
        self.char2id = char2id
        self.cfg = cfg

        if is_train:
            self.tf = transforms.Compose([
                transforms.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=255),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = s.crop_pil.convert("L")
        x = self.tf(x)  # 1xHxW
        y = self.char2id[s.gt_char]
        return x, y


# -------------------------
# Model
# -------------------------
def build_model(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # adapt first conv to grayscale (1 channel)
    w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        m.conv1.weight.copy_(w.mean(dim=1, keepdim=True))

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def decode_page(model: nn.Module, cfg: CFG, page_png: str, page_json: str,
                char2id: Dict[str, int], id2char: Dict[int, str]) -> Dict:
    model.eval()
    samples, data = extract_samples(page_png, page_json, cfg)

    tf = transforms.ToTensor()
    X = torch.stack([tf(s.crop_pil.convert("L")) for s in samples], dim=0).to(cfg.device)

    preds = []
    bs = 512
    for i in range(0, X.shape[0], bs):
        logits = model(X[i:i + bs])
        yhat = torch.argmax(logits, dim=1)
        preds.extend(yhat.cpu().tolist())

    by_line: Dict[int, List[Tuple[int, str]]] = {}
    for s, pid in zip(samples, preds):
        by_line.setdefault(s.line_idx, [])
        by_line[s.line_idx].append((s.char_idx, id2char[pid]))

    pred_lines = []
    gt_lines = []
    line_metrics = []

    for line in sorted(data["lines"], key=lambda x: int(x["line_idx"])):
        li = int(line["line_idx"])
        gt = line["text"]
        items = sorted(by_line.get(li, []), key=lambda t: t[0])
        pred = "".join([p for _, p in items])

        pred_lines.append(pred)
        gt_lines.append(gt)

        L = min(len(pred), len(gt))
        correct = sum(1 for k in range(L) if pred[k] == gt[k])
        acc = correct / len(gt) if len(gt) else 1.0

        line_metrics.append({
            "line_idx": li,
            "gt_len": len(gt),
            "pred_len": len(pred),
            "char_acc": acc,
            "cer": cer(pred, gt),
        })

    # Page-level metric on concatenated text (remove newlines)
    full_pred = "".join(pred_lines)
    full_gt = data.get("full_text", "".join(gt_lines))

    L = min(len(full_pred), len(full_gt))
    correct = sum(1 for k in range(L) if full_pred[k] == full_gt[k])
    page_char_acc = correct / len(full_gt) if len(full_gt) else 1.0

    return {
        "page_id": os.path.splitext(os.path.basename(page_png))[0],
        "page_char_acc": page_char_acc,
        "page_cer": cer(full_pred, full_gt),
        "pred_lines": pred_lines,
        "gt_lines": gt_lines,
        "line_metrics": line_metrics,
        "n_chars": int(data.get("n_chars", len(full_gt))),
        "n_lines": int(data.get("n_lines", len(pred_lines))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, default="/home/aous/Desktop/research/2025/synth/data/01_templates/images",
                    help="Directory with page_*.png")
    ap.add_argument("--json_dir", type=str, default="/home/aous/Desktop/research/2025/synth/data/01_templates/labels",
                    help="Directory with page_*.json")
    ap.add_argument("--out_dir", type=str, default="/home/aous/Desktop/research/2025/synth/runs/ocr_template_resnet18",
                    help="Output directory")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop_size", type=int, default=96)
    ap.add_argument("--pad_px", type=int, default=6)
    ap.add_argument("--no_binarize", action="store_true", help="Disable Otsu binarization")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    cfg = CFG(
        images_dir=args.images_dir,
        json_dir=args.json_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pad_px=args.pad_px,
        crop_size=args.crop_size,
        binarize=(not args.no_binarize),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        num_workers=args.num_workers,
    )

    if cfg.train_frac + cfg.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 (test is the remainder).")

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    pairs = list_pages(cfg.images_dir, cfg.json_dir)
    if len(pairs) == 0:
        raise RuntimeError(
            f"No (png,json) pairs found.\nimages_dir={cfg.images_dir}\njson_dir={cfg.json_dir}\n"
            f"Expected: page_*.png and page_*.json with same basenames."
        )

    # Reproducible shuffle + split by PAGES
    rnd = random.Random(cfg.seed)
    rnd.shuffle(pairs)

    n = len(pairs)
    n_train = int(cfg.train_frac * n)
    n_val = int(cfg.val_frac * n)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"[INFO] total pages={n} | train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"[INFO] device={cfg.device} | binarize={cfg.binarize} | crop_size={cfg.crop_size} | pad_px={cfg.pad_px}")
    if not HAS_CV2 and cfg.binarize:
        print("[WARN] cv2 not found; using simple threshold fallback for binarization.")

    # Extract train/val samples + build vocab
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    vocab_set = set()

    # Optional: quick sanity on first page
    s0, d0 = extract_samples(train_pairs[0][0], train_pairs[0][1], cfg)
    print(f"[SANITY] first train page: {os.path.basename(train_pairs[0][0])} | n_glyphs={len(s0)} | n_lines={d0.get('n_lines')}")

    for p, j in train_pairs:
        s, _ = extract_samples(p, j, cfg)
        train_samples.extend(s)
        for x in s:
            vocab_set.add(x.gt_char)

    for p, j in val_pairs:
        s, _ = extract_samples(p, j, cfg)
        val_samples.extend(s)
        for x in s:
            vocab_set.add(x.gt_char)

    vocab = sorted(list(vocab_set))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for c, i in char2id.items()}
    num_classes = len(vocab)

    with open(os.path.join(cfg.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] glyph samples: train={len(train_samples)} val={len(val_samples)} | num_classes={num_classes}")

    train_ds = GlyphDataset(train_samples, char2id, is_train=True, cfg=cfg)
    val_ds   = GlyphDataset(val_samples,   char2id, is_train=False, cfg=cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = os.path.join(cfg.out_dir, "best_model.pt")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                v_correct += int((pred == y).sum().item())
                v_total += int(y.numel())
        val_acc = v_correct / max(1, v_total)

        print(f"[E{ep:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "char2id": char2id,
                "cfg": cfg.__dict__,
                "best_val_acc": best_val_acc,
            }, best_path)

    print(f"[INFO] best_val_acc (glyph) = {best_val_acc:.4f}")
    print(f"[INFO] saved best model: {best_path}")

    # Load best model
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # End-to-end decoding evaluation on TEST pages
    page_accs = []
    page_cers = []
    page_outputs = []

    for p, j in test_pairs:
        r = decode_page(model, cfg, p, j, char2id, id2char)
        page_outputs.append(r)
        page_accs.append(r["page_char_acc"])
        page_cers.append(r["page_cer"])

    summary = {
        "n_pages_total": n,
        "n_pages_train": len(train_pairs),
        "n_pages_val": len(val_pairs),
        "n_pages_test": len(test_pairs),
        "best_val_acc_glyph": float(best_val_acc),
        "avg_test_page_char_acc": float(np.mean(page_accs)) if page_accs else None,
        "avg_test_page_cer": float(np.mean(page_cers)) if page_cers else None,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "crop_size": cfg.crop_size,
        "pad_px": cfg.pad_px,
        "binarize": cfg.binarize,
    }

    with open(os.path.join(cfg.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pages": page_outputs}, f, ensure_ascii=False, indent=2)

    print("[RESULT]", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


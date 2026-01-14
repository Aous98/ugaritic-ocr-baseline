#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, random, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


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

    arch: str = "resnet34"   # resnet18/resnet34/resnet50
    pretrained: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def list_pages(images_dir: str, json_dir: str) -> List[Tuple[str, str]]:
    pngs = sorted(glob.glob(os.path.join(images_dir, "page_*.png")))
    pairs = []
    for p in pngs:
        base = os.path.splitext(os.path.basename(p))[0]
        j = os.path.join(json_dir, base + ".json")
        if os.path.exists(j):
            pairs.append((p, j))
    return pairs


def crop_with_padding(img: np.ndarray, bbox_xyxy, pad: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2]


def maybe_binarize(crop: np.ndarray) -> np.ndarray:
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
    canvas[y0:y0+h, x0:x0+w] = crop
    return Image.fromarray(canvas).resize((size, size), resample=Image.BILINEAR)


@dataclass
class Sample:
    gt_char: str
    crop_pil: Image.Image


def extract_samples(png_path: str, json_path: str, cfg: CFG) -> List[Sample]:
    img = load_image_gray(png_path)
    data = read_json(json_path)
    out: List[Sample] = []
    for line in data["lines"]:
        for ch in line["chars"]:
            gt = ch["char"]
            crop = crop_with_padding(img, ch["bbox_xyxy"], cfg.pad_px)
            if cfg.binarize:
                crop = maybe_binarize(crop)
            crop_pil = to_square_and_resize(crop, cfg.crop_size)
            out.append(Sample(gt, crop_pil))
    return out


class GlyphDataset(Dataset):
    def __init__(self, samples: List[Sample], char2id: Dict[str,int], is_train: bool):
        self.samples = samples
        self.char2id = char2id
        if is_train:
            self.tf = transforms.Compose([
                transforms.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=255),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([transforms.ToTensor()])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = self.tf(s.crop_pil.convert("L"))  # 1xHxW
        y = self.char2id[s.gt_char]
        return x, y


def build_model(num_classes: int, arch: str, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet34":
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # Make grayscale conv1
    if pretrained:
        w = base.conv1.weight.data
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
    else:
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    base.fc = nn.Linear(base.fc.in_features, num_classes)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="/home/aous/Desktop/research/2025/synth/data/01_templates/images")
    ap.add_argument("--json_dir",   default="/home/aous/Desktop/research/2025/synth/data/01_templates/labels")
    ap.add_argument("--out_dir",    default="/home/aous/Desktop/research/2025/synth/runs/ocr_template_resnet34")
    ap.add_argument("--arch",       default="resnet34", choices=["resnet18","resnet34","resnet50"])
    ap.add_argument("--epochs",     type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--crop_size",  type=int, default=96)
    ap.add_argument("--pad_px",     type=int, default=6)
    ap.add_argument("--no_binarize", action="store_true")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true")
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
        arch=args.arch,
        pretrained=(not args.no_pretrained),
    )

    if cfg.train_frac + cfg.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    pairs = list_pages(cfg.images_dir, cfg.json_dir)
    if not pairs:
        raise RuntimeError("No page_*.png/json pairs found.")

    rnd = random.Random(cfg.seed)
    rnd.shuffle(pairs)

    n = len(pairs)
    n_train = int(cfg.train_frac * n)
    n_val = int(cfg.val_frac * n)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]

    print(f"[INFO] total pages={n} train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"[INFO] arch={cfg.arch} pretrained={cfg.pretrained} device={cfg.device} binarize={cfg.binarize}")

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    vocab_set = set()

    for p, j in train_pairs:
        s = extract_samples(p, j, cfg)
        train_samples.extend(s)
        for x in s: vocab_set.add(x.gt_char)

    for p, j in val_pairs:
        s = extract_samples(p, j, cfg)
        val_samples.extend(s)
        for x in s: vocab_set.add(x.gt_char)

    vocab = sorted(list(vocab_set))
    char2id = {c:i for i,c in enumerate(vocab)}
    id2char = {i:c for c,i in char2id.items()}

    with open(os.path.join(cfg.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f, ensure_ascii=False, indent=2)

    num_classes = len(vocab)
    print(f"[INFO] glyph samples: train={len(train_samples)} val={len(val_samples)} num_classes={num_classes}")

    train_ds = GlyphDataset(train_samples, char2id, is_train=True)
    val_ds = GlyphDataset(val_samples, char2id, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(num_classes, cfg.arch, cfg.pretrained).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = os.path.join(cfg.out_dir, "best_model.pt")

    for ep in range(1, cfg.epochs+1):
        model.train()
        tot = 0; corr = 0; loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device); y = y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            corr += int((pred == y).sum().item())
            tot += int(y.numel())

        train_loss = loss_sum / max(1, tot)
        train_acc = corr / max(1, tot)

        model.eval()
        vtot = 0; vcorr = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg.device); y = y.to(cfg.device)
                pred = torch.argmax(model(x), dim=1)
                vcorr += int((pred == y).sum().item())
                vtot += int(y.numel())
        val_acc = vcorr / max(1, vtot)

        print(f"[E{ep:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "char2id": char2id,
                "cfg": cfg.__dict__,
                "best_val_acc": best_val,
                "test_pairs": test_pairs,  # store split for reproducibility
            }, best_path)

    print(f"[INFO] best_val_acc={best_val:.6f}")
    print(f"[INFO] saved: {best_path}")


if __name__ == "__main__":
    main()


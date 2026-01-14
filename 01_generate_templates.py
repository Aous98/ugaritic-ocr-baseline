#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# =========================
# CONFIG (YOUR PATHS)
# =========================
PROJECT_ROOT = "/home/aous/Desktop/research/2025/synth"

CSV_PATH  = os.path.join(PROJECT_ROOT, "data/00_raw/map_dic.csv")
OUT_IMG   = os.path.join(PROJECT_ROOT, "data/01_templates/images")
OUT_LBL   = os.path.join(PROJECT_ROOT, "data/01_templates/labels")

# Font (install Noto Sans Ugaritic if needed)
FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansUgaritic-Regular.ttf"

# We will read the alphabet from this column in map_dic.csv:
ALPHABET_COLUMN = "Letter"

# Output page settings
IMG_W, IMG_H = 1024, 1024
MARGIN_X, MARGIN_Y = 60, 60

# Text layout settings
LINES_MIN, LINES_MAX = 8, 15
CHARS_MIN, CHARS_MAX = 10, 15

# Typography
FONT_SIZE = 56      # reduce to 48 if overflow
LINE_GAP  = 18
CHAR_GAP  = 10

# How many pages to generate
N_IMAGES = 2000

# Randomness
SEED = 42
# =========================


def ensure_dirs():
    Path(OUT_IMG).mkdir(parents=True, exist_ok=True)
    Path(OUT_LBL).mkdir(parents=True, exist_ok=True)


def load_font():
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(
            f"Font not found: {FONT_PATH}\n"
            f"Install fonts-noto-extra (or Noto Sans Ugaritic), or set FONT_PATH correctly."
        )
    return ImageFont.truetype(FONT_PATH, FONT_SIZE)


def build_alphabet_from_map_dic(df: pd.DataFrame) -> list[str]:
    if ALPHABET_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{ALPHABET_COLUMN}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    alphabet = []
    for v in df[ALPHABET_COLUMN].dropna().astype(str).tolist():
        v = v.strip()
        if not v:
            continue
        # Some CSVs may contain multiple glyphs; take each non-space char.
        for ch in v:
            if not ch.isspace():
                alphabet.append(ch)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for ch in alphabet:
        if ch not in seen:
            uniq.append(ch)
            seen.add(ch)

    if not uniq:
        raise ValueError("Alphabet is empty after parsing. Check CSV content.")

    return uniq


def sample_page(alphabet: list[str]) -> list[list[str]]:
    n_lines = random.randint(LINES_MIN, LINES_MAX)
    lines = []
    for _ in range(n_lines):
        n_chars = random.randint(CHARS_MIN, CHARS_MAX)
        lines.append([random.choice(alphabet) for _ in range(n_chars)])
    return lines


def render_page(lines: list[list[str]], font: ImageFont.FreeTypeFont):
    """
    Returns:
      img: PIL image
      label: dict with full text, per-line text, and per-char boxes
    """
    img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = MARGIN_Y
    label_lines = []
    all_chars = []

    for line_idx, chars in enumerate(lines):
        x = MARGIN_X

        # Estimate line height from first glyph
        bbox_ref = draw.textbbox((0, 0), chars[0], font=font)
        line_h = (bbox_ref[3] - bbox_ref[1]) + LINE_GAP

        line_text = "".join(chars)
        char_entries = []

        for char_idx, ch in enumerate(chars):
            bbox = draw.textbbox((x, y), ch, font=font)  # (x1,y1,x2,y2)
            x1, y1, x2, y2 = bbox

            draw.text((x, y), ch, fill=(0, 0, 0), font=font)

            char_entry = {
                "char": ch,
                "line_idx": line_idx,
                "char_idx": char_idx,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            }
            char_entries.append(char_entry)
            all_chars.append(char_entry)

            glyph_w = (x2 - x1)
            x += glyph_w + CHAR_GAP

            # If we run out of width, stop line early (rare; avoids broken labels)
            if x > IMG_W - MARGIN_X:
                break

        label_lines.append({
            "line_idx": line_idx,
            "text": "".join([c["char"] for c in char_entries]),
            "chars": char_entries
        })

        y += line_h

        # Stop if out of page
        if y > IMG_H - MARGIN_Y:
            break

    full_text = "\n".join([l["text"] for l in label_lines])

    label = {
        "full_text": full_text,
        "lines": label_lines,
        "n_lines": len(label_lines),
        "n_chars": len(all_chars),
        "img_wh": [IMG_W, IMG_H],
        "font_path": FONT_PATH,
        "font_size": FONT_SIZE,
        "seed": SEED,
        "source_csv": CSV_PATH,
        "alphabet_column": ALPHABET_COLUMN,
    }
    return img, label


def main():
    random.seed(SEED)

    ensure_dirs()
    df = pd.read_csv(CSV_PATH)

    alphabet = build_alphabet_from_map_dic(df)
    print(f"[OK] CSV: {CSV_PATH}")
    print(f"[OK] Alphabet column: {ALPHABET_COLUMN}")
    print(f"[OK] Alphabet size: {len(alphabet)}")
    print(f"[OK] Alphabet: {''.join(alphabet)}")

    font = load_font()

    for i in range(N_IMAGES):
        lines = sample_page(alphabet)
        img, label = render_page(lines, font)

        stem = f"page_{i:06d}"
        img_path = os.path.join(OUT_IMG, f"{stem}.png")
        lbl_path = os.path.join(OUT_LBL, f"{stem}.json")

        img.save(img_path)
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{N_IMAGES}] generated")

    print("\nDone.")
    print(f"Images: {OUT_IMG}")
    print(f"Labels: {OUT_LBL}")


if __name__ == "__main__":
    main()


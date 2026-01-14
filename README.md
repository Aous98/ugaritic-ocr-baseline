# ugaritic-ocr-baseline


Project root:
- /home/aous/Desktop/research/2025/synth/

## scripts

### 01_generate_templates.py
- Location:
  - /home/aous/Desktop/research/2025/synth/scripts/01_generate_templates.py
- Purpose:
  - Generate synthetic template “pages” from the Ugaritic alphabet (random characters).
  - Output is white background + black glyphs, with perfect ground-truth labels.
- Inputs:
  - /home/aous/Desktop/research/2025/synth/data/00_raw/map_dic.csv  (uses column: Letter)
  - Font: /usr/share/fonts/truetype/noto/NotoSansUgaritic-Regular.ttf
- Outputs:
  - Images: /home/aous/Desktop/research/2025/synth/data/01_templates/images/*.png
  - Labels: /home/aous/Desktop/research/2025/synth/data/01_templates/labels/*.json
- Run:
  - python3 /home/aous/Desktop/research/2025/synth/scripts/01_generate_templates.py

### 02_extract_style_crops.py
- Location:
  - /home/aous/Desktop/research/2025/synth/scripts/02_extract_style_crops.py
- Purpose:
  - Extract style reference crops from real images into two domains:
    - clay: photo tablets (RS*)
    - hand: facsimile drawings (KTU*)
  - For hand images, uses robust acceptance based on:
    - nonblack_frac (gray >= 25)
    - Otsu stroke_frac
- Inputs:
  - /home/aous/Desktop/research/2025/synth/real imgs/*
- Outputs:
  - Clay crops: /home/aous/Desktop/research/2025/synth/data/02_style_refs/clay/*.png
  - Hand crops: /home/aous/Desktop/research/2025/synth/data/02_style_refs/hand/*.png
  - Manifest: /home/aous/Desktop/research/2025/synth/data/02_style_refs/_meta/crops_manifest.json
- Run:
  - python3 /home/aous/Desktop/research/2025/synth/scripts/02_extract_style_crops.py



### 04a_prepare_lora_clay_quickset.py
- Location:
  - /home/aous/Desktop/research/2025/synth/scripts/04a_prepare_lora_clay_quickset.py
- Purpose:
  - Create a small “quick test” LoRA training set from extracted clay style crops.
  - Samples a fixed number of crops (default 300), resizes them to a stable SD1.5 resolution (512×512), and writes one caption per image.
- Inputs:
  - Source clay crops:
    - /home/aous/Desktop/research/2025/synth/data/02_style_refs/clay/
- Processing:
  - Randomly samples up to N images with a fixed seed for reproducibility.
  - Resizes each crop to TARGET_SIZE (default 512).
  - Writes images to:
    - data/04_lora/clay/images/
  - Writes captions (same tokenized style prompt for all images) to:
    - data/04_lora/clay/captions/
- Outputs:
  - Resized training images:
    - /home/aous/Desktop/research/2025/synth/data/04_lora/clay/images/clay_XXXXX.png
  - Caption files:
    - /home/aous/Desktop/research/2025/synth/data/04_lora/clay/captions/clay_XXXXX.txt
- Notes:
  - This dataset is intended for a fast smoke-test LoRA run (not final training quality).
  - Captions include a dedicated trigger token (e.g., `ugaritic_clay_style`) used later when applying the LoRA in generation.

### LoRA training (kohya sd-scripts) — clay_quick_r8_unetonly
- Location (trainer):
  - /home/aous/Desktop/research/2025/synth/external/sd-scripts/train_network.py
- Purpose:
  - Train a quick UNet-only LoRA that captures clay tablet / carved inscription style from real clay crops.
  - Intended as a smoke test to shift SD1.5 outputs toward the Ugaritic clay domain without changing layout logic.
- Dataset:
  - Parent directory (kohya format):
    - /home/aous/Desktop/research/2025/synth/data/04_lora/clay/kohya/
  - Subfolder:
    - 20_claystyle/  (300 images, repeats=20)
- Base model:
  - /home/aous/Desktop/research/2025/data/generate/sd15/v1-5-pruned.safetensors
- Key settings:
  - UNet-only LoRA, rank=8 (network_dim=8, alpha=8)
  - resolution=512
  - max_train_steps=600
  - mixed_precision=fp16
  - memory-efficient attention via --sdpa
  - caption extension forced to .txt
- Output:
  - /home/aous/Desktop/research/2025/synth/data/04_lora/clay/lora_quick/clay_quick_r8_unetonly.safetensors
- Result:
  - Training completed successfully (600/600 steps), avr_loss ≈ 0.185

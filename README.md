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


### 03a_build_lineband_control.py
- Location:
  - /home/aous/Desktop/research/2025/synth/scripts/03a_build_lineband_control.py
- Purpose:
  - Generate low-frequency ControlNet conditioning images that encode only the vertical layout of text lines.
  - Each output image contains horizontal “line bands” corresponding to text lines, without any character-level or stroke-level information.
- Motivation:
  - Designed specifically for dense multi-line Ugaritic pages where character-level ControlNet representations (scribble, canny, lineart) collapse for long sequences.
  - Shifts ControlNet responsibility from glyph geometry to page-level layout, enabling stable diffusion generation for pages with many characters.
- Inputs:
  - Template labels:
    - data/01_templates/labels/*.json
      - Uses `lines[*].chars[*].bbox_xyxy` and `img_wh`
  - (Template images are not required directly; only their dimensions are used via the JSON.)
- Processing:
  - For each line in the JSON:
    - Computes vertical extent from character bounding boxes.
    - Expands the extent by a fixed padding factor.
    - Renders a full-width horizontal band at the corresponding vertical position.
  - Background is set to 0 (black); line bands are set to 255 (white).
- Outputs:
  - Control images:
    - data/03_controls/lineband/*.png
    - Same resolution as the original template image.
- Notes:
  - Output images are intended to be used as ControlNet conditioning inputs.
  - These images encode layout only (number of lines and spacing), not character shapes.
  - Compatible with subsequent diffusion generation using ControlNet + LoRA.
- Run:
  - python3 /home/aous/Desktop/research/2025/synth/scripts/03a_build_lineband_control.py
  
  
### 03b_test_single_lineband_diffusion_local.py
- Location:
  - /home/aous/Desktop/research/2025/synth/scripts/03b_test_single_lineband_diffusion_local.py
- Purpose:
  - Run a single-image diffusion sanity check using the line-band ControlNet representation to verify that dense multi-line pages do not collapse.
  - Produces one clay-like synthetic image (layout-stable) before any LoRA training or batch generation.
- Inputs:
  - Line-band control image:
    - data/03_controls/lineband/page_000000.png
  - Local Stable Diffusion base model (SD 1.5) directory:
    - Path set in `SD_BASE_PATH` inside the script
  - Local ControlNet Scribble model directory:
    - Path set in `CONTROLNET_SCRIBBLE_PATH` inside the script
- Processing:
  - Loads SD 1.5 + ControlNet from local filesystem only (`local_files_only=True`), preventing downloads.
  - Resizes the line-band control image to `GEN_SIZE` (default 512) and converts it to RGB for ControlNet conditioning.
  - Runs `StableDiffusionControlNetPipeline` with a clay-tablet prompt and negative prompt to generate one output image.
- Outputs:
  - Single test image:
    - data/03_synth/_tests/test_lineband_clay.png
- Notes:
  - This script is intended as a minimal “proof of pipeline” step:
    - Confirms ControlNet conditioning works for long pages when using layout-only control.
    - Does not enforce Ugaritic glyph fidelity; Ugaritic-specific appearance is added later via LoRA.
- Run:
  - (Activate env)
  - python3 /home/aous/Desktop/research/2025/synth/scripts/03b_test_single_lineband_diffusion_local.py

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

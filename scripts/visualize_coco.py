#!/usr/bin/env python3
"""
visualize_coco.py

Reads a COCO-format annotation file and draws every bounding box onto the
corresponding image. Annotated copies are written to the output directory,
mirroring the input sub-folder structure.

Usage
-----
    python visualize_coco.py
    python visualize_coco.py --data-dir ./my_images --output-dir ./vis \
                             --annotations ./instances.json
"""

import argparse
import colorsys
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# =============================================================================
#  CONFIGURATION  —  edit these, or override them from the command line
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR_NAME = "data/raw/valid"
OUTPUT_DIR_NAME = "visualized_valid"
ANNOTATION_FILE = "data/raw/annotations/valid_annotations.json"

DATA_DIR = BASE_DIR / DATA_DIR_NAME
OUTPUT_DIR = BASE_DIR / OUTPUT_DIR_NAME
ANNOTATION_PATH = BASE_DIR / ANNOTATION_FILE


# ---- Drawing options --------------------------------------------------------
LINE_WIDTH_SCALE   = 0.004    # box thickness = image_width  * this  (min 2 px)
FONT_SIZE_SCALE    = 0.030    # font size     = image_height * this  (min 12 px)
DRAW_LABELS        = True     # draw the category name above each box
SHOW_SCORE         = True     # append the score if the annotation has one
DRAW_FILLED_LABEL  = True     # draw a filled background behind the label
SAVE_EMPTY_IMAGES  = True     # also write out images that have no annotations

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =============================================================================


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def color_for(category_id) -> tuple:
    """Deterministic, visually distinct colour for a category id."""
    try:
        cid = int(category_id)
    except (TypeError, ValueError):
        cid = 0
    hue = (cid * 0.618033988749895) % 1.0          # golden-ratio hue stepping
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def load_font(size: int):
    """Try to get a scalable font, otherwise fall back to the bitmap default."""
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf",
                 "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 9.2
    except TypeError:
        return ImageFont.load_default()


def resolve_image_path(data_dir: Path, file_name: str):
    """Locate an image, tolerating a flat folder or a nested COCO layout."""
    candidate = data_dir / file_name
    if candidate.is_file():
        return candidate
    candidate = data_dir / Path(file_name).name      # fall back to basename
    if candidate.is_file():
        return candidate
    return None


def output_path_for(out_dir: Path, file_name: str) -> Path:
    """Keep the relative sub-structure of the COCO file_name."""
    rel = Path(file_name)
    if rel.is_absolute() or ".." in rel.parts:
        rel = Path(rel.name)
    return out_dir / rel


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def draw_one(img_path: Path, anns, cat_name, out_path: Path) -> int:
    """Draw all annotations on one image. Returns the number of boxes drawn."""
    try:
        img = Image.open(img_path)
        img.load()
    except Exception as exc:                            # noqa: BLE001
        print(f"    [skip] cannot open {img_path.name}: {exc}")
        return 0

    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    line_w    = max(2, int(round(width * LINE_WIDTH_SCALE)))
    font_size = max(12, int(round(height * FONT_SIZE_SCALE)))
    font      = load_font(font_size)
    draw      = ImageDraw.Draw(img)

    drawn = 0
    for ann in anns:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        x, y, w, h = (float(v) for v in bbox[:4])
        if w <= 0 or h <= 0:
            continue

        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))

        color = color_for(ann.get("category_id", 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)

        if DRAW_LABELS:
            label = str(cat_name.get(ann.get("category_id"),
                                     ann.get("category_id", "object")))
            if SHOW_SCORE and "score" in ann:
                try:
                    label = f"{label} {float(ann['score']):.2f}"
                except (TypeError, ValueError):
                    pass

            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = max(2, font_size // 6)

            ty0 = y0 - (th + 2 * pad)
            if ty0 < 0:                      # keep the label inside the image
                ty0 = y0
            tx0 = min(max(0, x0), max(0, width - (tw + 2 * pad)))

            if DRAW_FILLED_LABEL:
                draw.rectangle([tx0, ty0, tx0 + tw + 2 * pad, ty0 + th + 2 * pad],
                               fill=color)
                lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                text_color = (0, 0, 0) if lum > 150 else (255, 255, 255)
            else:
                text_color = color

            draw.text((tx0 + pad, ty0 + pad - tb[1]), label,
                      fill=text_color, font=font)

        drawn += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(out_path, quality=95)
    else:
        img.save(out_path)

    return drawn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw COCO bounding boxes onto every image in a folder.")
    parser.add_argument("--data-dir",    default=None,
                        help=f"input image folder (default: {DATA_DIR_NAME})")
    parser.add_argument("--output-dir",  default=None,
                        help=f"output folder (default: {OUTPUT_DIR_NAME})")
    parser.add_argument("--annotations", default=None,
                        help=f"COCO json file (default: {ANNOTATION_FILE})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir \
        else (BASE_DIR / DATA_DIR_NAME)
    out_dir  = Path(args.output_dir).expanduser().resolve() if args.output_dir \
        else (BASE_DIR / OUTPUT_DIR_NAME)
    ann_file = Path(args.annotations).expanduser().resolve() if args.annotations \
        else (BASE_DIR / ANNOTATION_FILE)

    # ---- sanity checks ------------------------------------------------------
    if not ann_file.is_file():
        print(f"[error] annotation file not found: {ann_file}")
        return 1
    if not data_dir.is_dir():
        print(f"[error] image directory not found: {data_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load COCO ----------------------------------------------------------
    with open(ann_file, "r", encoding="utf-8") as fh:
        coco = json.load(fh)

    images = coco.get("images", [])
    anns   = coco.get("annotations", [])
    cats   = coco.get("categories", [])

    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}

    anns_by_image = defaultdict(list)
    for a in anns:
        anns_by_image[a["image_id"]].append(a)

    print(f"Images in json : {len(images)}")
    print(f"Annotations    : {len(anns)}")
    print(f"Categories     : {len(cats)}")
    print(f"Input folder   : {data_dir}")
    print(f"Output folder  : {out_dir}")
    print("-" * 60)

    # ---- process ------------------------------------------------------------
    written = missing = no_ann = total_boxes = 0

    for idx, info in enumerate(images, 1):
        file_name = info.get("file_name")
        if not file_name:
            continue

        img_path = resolve_image_path(data_dir, file_name)
        if img_path is None:
            print(f"  [missing] {file_name}")
            missing += 1
            continue

        img_anns = anns_by_image.get(info["id"], [])

        if not img_anns and not SAVE_EMPTY_IMAGES:
            no_ann += 1
            continue

        out_path = output_path_for(out_dir, file_name)
        n = draw_one(img_path, img_anns, cat_name, out_path)
        total_boxes += n
        written += 1

        if not img_anns:
            no_ann += 1

        if idx % 100 == 0 or idx == len(images):
            print(f"  ... {idx}/{len(images)} processed")

    # ---- summary ------------------------------------------------------------
    print("-" * 60)
    print(f"Images written : {written}")
    print(f"Boxes drawn    : {total_boxes}")
    print(f"Images w/o ann : {no_ann}")
    print(f"Missing files  : {missing}")
    print(f"Results saved  : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
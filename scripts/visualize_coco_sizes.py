#!/usr/bin/env python3
"""Draw COCO ground truth with size tags; no model inference is performed.

Sizes use annotation ``area`` in original-image pixels squared. The saved
COCO evaluation ranges include both endpoints, so boundary boxes get two tags.
Drawing helpers are reused from visualize_coco.py without modifying that file.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from visualize_coco import (
    BASE_DIR,
    FONT_SIZE_SCALE,
    LINE_WIDTH_SCALE,
    color_for,
    load_font,
    output_path_for,
    resolve_image_path,
)


def size_group(area: float) -> str:
    """Match the inclusive area ranges used by the baseline COCO evaluator."""
    if not math.isfinite(area) or not 0 <= area <= 10_000_000_000:
        raise ValueError(f"Area outside the baseline COCO ranges: {area}")
    if area < 1024:
        return "small"
    if area == 1024:
        return "small/medium"
    if area < 9216:
        return "medium"
    if area == 9216:
        return "medium/large"
    return "large"


def box_line_width(image_width: int) -> int:
    """Halve the original script's actual outline, rounded to whole pixels."""
    original = max(2, int(round(image_width * LINE_WIDTH_SCALE)))
    return max(1, int(round(original / 2)))


def draw_one(image_path: Path, annotations: list, category_names: dict,
             output_path: Path) -> int:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    drawn = 0

    for annotation in annotations:
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, box_width, box_height = map(float, bbox)
        if box_width <= 0 or box_height <= 0:
            continue

        # Use the same stored area as evaluation, not resized or relative area.
        area = float(annotation["area"])
        group = size_group(area)
        category_id = annotation["category_id"]
        category = category_names.get(category_id, str(category_id))
        label = f"{category} | {group} | {area:g} px²"
        color = color_for(category_id)

        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + box_width)), int(round(y + box_height))
        draw.rectangle((x0, y0, x1, y1), outline=color,
                       width=box_line_width(width))

        font_size = max(12, int(round(height * FONT_SIZE_SCALE)))
        while True:
            font = load_font(font_size)
            bounds = draw.textbbox((0, 0), label, font=font)
            text_width = bounds[2] - bounds[0]
            text_height = bounds[3] - bounds[1]
            padding = max(2, font_size // 6)
            if text_width + 2 * padding <= width or font_size <= 6:
                break
            font_size -= 1

        label_width = text_width + 2 * padding
        label_height = text_height + 2 * padding
        left = min(max(0, x0), max(0, width - label_width))
        if y0 >= label_height:
            top = y0 - label_height
        elif y1 + label_height <= height:
            top = max(0, y1)
        else:
            top = min(max(0, y0), max(0, height - label_height))
        draw.rectangle((left, top, left + label_width, top + label_height),
                       fill=color)
        luminance = sum(c * weight for c, weight in
                        zip(color, (0.299, 0.587, 0.114)))
        draw.text((left + padding - bounds[0], top + padding - bounds[1]),
                  label, font=font,
                  fill=(0, 0, 0) if luminance > 150 else (255, 255, 255))
        drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(output_path, quality=95)
    else:
        image.save(output_path)
    return drawn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=BASE_DIR / "data/raw/valid")
    parser.add_argument("--annotations", type=Path,
                        default=BASE_DIR / "data/raw/annotations/kaggle_val_annotations.json")
    parser.add_argument("--output-dir", type=Path,
                        default=BASE_DIR / "runs/baseline-001/eval-epoch19/visual-checks/ground-truth-sizes")
    parser.add_argument("--limit", type=int,
                        help="Render only the first N image records for a quick preview.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    data_dir = args.data_dir.expanduser().resolve()
    annotation_file = args.annotations.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.is_dir() or not annotation_file.is_file():
        parser.error("The input image directory and annotation file must exist")
    if output_dir == data_dir or data_dir in output_dir.parents:
        parser.error("Choose an output directory outside the source image directory")

    with annotation_file.open(encoding="utf-8") as source:
        coco = json.load(source)
    category_names = {category["id"]: category["name"]
                      for category in coco["categories"]}
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    images = coco["images"]
    if args.limit is not None:
        images = images[:args.limit]
    written = boxes = missing = 0
    for image in images:
        file_name = image["file_name"]
        image_path = resolve_image_path(data_dir, file_name)
        if image_path is None:
            print(f"[missing] {file_name}")
            missing += 1
            continue
        boxes += draw_one(image_path, annotations_by_image[image["id"]],
                          category_names, output_path_for(output_dir, file_name))
        written += 1
        if written % 100 == 0:
            print(f"Images written: {written}/{len(images)}")

    print(f"Images written: {written}; boxes drawn: {boxes}; missing images: {missing}")
    print(f"Results saved: {output_dir}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

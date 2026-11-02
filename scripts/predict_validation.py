#!/usr/bin/env python3
"""Render validation predictions in one process, loading EMA weights once.

Run from the project root. Relative input paths resolve against that root.
Existing output images with matching filenames are replaced. The score threshold
only controls drawing; this script does not calculate or modify AP metrics.
"""

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from visualize_coco import FONT_SIZE_SCALE, color_for, load_font
from visualize_coco_sizes import box_line_width

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "thrid-party/RT-DETR/rtdetrv2_pytorch"))
from src.core import YAMLConfig


def project_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_pipeline(config: Path, checkpoint: Path, device: torch.device,
                  batch_size: int):
    """Build the original validation pipeline and restore EMA exactly once."""
    cfg = YAMLConfig(str(config))
    # Full checkpoint restoration makes a separate backbone download unnecessary.
    cfg.yaml_cfg.setdefault("PResNet", {})["pretrained"] = False
    loader_cfg = cfg.yaml_cfg["val_dataloader"]
    loader_cfg.pop("batch_size", None)
    loader_cfg.update(total_batch_size=batch_size, num_workers=0,
                      shuffle=False, drop_last=False)
    for key in ("img_folder", "ann_file"):
        loader_cfg["dataset"][key] = str(project_path(loader_cfg["dataset"][key]))

    print(f"Loading EMA checkpoint once: {checkpoint}", flush=True)
    # This is the user's trusted local training checkpoint, including EMA state.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "ema" not in state or "module" not in state["ema"]:
        raise ValueError("Checkpoint has no EMA module; baseline requires EMA weights")
    model = cfg.model
    model.load_state_dict(state["ema"]["module"], strict=True)
    del state
    model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    loader = cfg.val_dataloader
    return model, postprocessor, loader


@torch.inference_mode()
def predict_batch(model, postprocessor, samples, targets, device):
    """Use the same original-size conversion as the evaluation entry point."""
    original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
    outputs = model(samples.to(device))
    results = postprocessor(outputs, original_sizes)
    return [{key: value.cpu() for key, value in result.items()} for result in results]


def draw_prediction(source: Path, destination: Path, prediction: dict,
                    category_names: dict, threshold: float) -> int:
    with Image.open(source) as original:
        image = original.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    keep = prediction["scores"] > threshold
    boxes = prediction["boxes"][keep]
    scores = prediction["scores"][keep]
    labels = prediction["labels"][keep]
    if not torch.isfinite(boxes).all() or not torch.isfinite(prediction["scores"]).all():
        raise ValueError(f"Non-finite prediction for {source.name}")

    for box, score, label_id in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x0, y0, x1, y1 = box
        color = color_for(label_id)
        draw.rectangle(box, outline=color, width=box_line_width(width))
        name = category_names.get(label_id, str(label_id))
        text = f"{name} ({label_id}) | {score:.3f}"
        font_size = max(12, round(height * FONT_SIZE_SCALE))
        while True:
            font = load_font(font_size)
            bounds = draw.textbbox((0, 0), text, font=font)
            padding = max(2, font_size // 6)
            label_width = bounds[2] - bounds[0] + 2 * padding
            label_height = bounds[3] - bounds[1] + 2 * padding
            if label_width <= width or font_size <= 6:
                break
            font_size -= 1
        left = min(max(0, x0), max(0, width - label_width))
        if y0 >= label_height:
            top = y0 - label_height
        elif y1 + label_height <= height:
            top = max(0, y1)
        else:
            top = min(max(0, y0), max(0, height - label_height))
        draw.rectangle((left, top, left + label_width, top + label_height), fill=color)
        luminance = sum(c * w for c, w in zip(color, (0.299, 0.587, 0.114)))
        draw.text((left + padding - bounds[0], top + padding - bounds[1]),
                  text, font=font,
                  fill=(0, 0, 0) if luminance > 150 else (255, 255, 255))

    destination.parent.mkdir(parents=True, exist_ok=True)
    options = {"quality": 95} if destination.suffix.lower() in (".jpg", ".jpeg") else {}
    image.save(destination, **options)
    return len(boxes)


def image_paths(dataset, output_dir: Path, image_ids: list) -> dict:
    """Check all selected filenames before writing any predictions."""
    output_dir = output_dir.resolve()
    data_dir = Path(dataset.img_folder).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the source image directory")
    paths = {}
    destinations = set()
    for image_id in image_ids:
        file_name = Path(dataset.coco.imgs[image_id]["file_name"])
        if file_name.is_absolute() or ".." in file_name.parts:
            raise ValueError(f"Expected a relative image filename: {file_name}")
        source = (data_dir / file_name).resolve()
        destination = (output_dir / file_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing validation image: {source}")
        if destination == source or output_dir not in destination.parents:
            raise ValueError(f"Output would overwrite a source or escape output directory: {destination}")
        if destination in destinations:
            raise ValueError(f"Duplicate output filename: {file_name}")
        destinations.add(destination)
        paths[image_id] = (source, destination)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path,
                        default=Path("configs/training/baseline-001-eval.yml"))
    parser.add_argument("-r", "--resume", type=Path,
                        default=Path("runs/baseline-001/checkpoints/checkpoint0019.pth"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/baseline-001/eval-epoch19/visual-checks/predictions"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--limit", type=int, help="Only process the first N images for a preview")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")

    written = displayed = empty = 0
    started = time.monotonic()
    try:
        config = project_path(args.config)
        checkpoint = project_path(args.resume)
        output_dir = project_path(args.output_dir)
        device = torch.device(args.device)
        model, postprocessor, loader = load_pipeline(config, checkpoint, device, args.batch_size)
        dataset = loader.dataset
        image_ids = dataset.ids[:args.limit] if args.limit is not None else dataset.ids
        total = len(image_ids)
        if not total:
            raise ValueError("Validation dataset is empty")
        paths = image_paths(dataset, output_dir, image_ids)
        category_names = dataset.category2name
        print(f"Predicting {total} images on {device}; batch size {args.batch_size}; "
              f"display score > {args.score_threshold}", flush=True)
        print(f"Output: {output_dir} (matching image files will be overwritten)", flush=True)

        batches = itertools.islice(loader, math.ceil(total / args.batch_size))
        for samples, targets in batches:
            remaining = total - written
            samples, targets = samples[:remaining], targets[:remaining]
            predictions = predict_batch(model, postprocessor, samples, targets, device)
            for target, prediction in zip(targets, predictions):
                image_id = target["image_id"].item()
                source, destination = paths[image_id]
                count = draw_prediction(source, destination, prediction,
                                        category_names, args.score_threshold)
                written += 1
                displayed += count
                empty += count == 0
                if written == 1 or written % 10 == 0 or written == total:
                    elapsed = time.monotonic() - started
                    print(f"[{written}/{total}] images saved; {elapsed:.1f}s elapsed", flush=True)
        if written != total:
            raise RuntimeError(f"Expected {total} images, saved {written}")
    except Exception as error:
        print(f"[error] Stopped after {written} saved images: {error}", file=sys.stderr)
        return 1

    print(f"Complete: {written} images; {displayed} displayed boxes; "
          f"{empty} images without detections above threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

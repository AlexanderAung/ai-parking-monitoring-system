#!/usr/bin/env python3
"""Join matching ground-truth and prediction images side by side.

Ground truth is on the left; predictions are on the right. Images retain their
original scale. Output filenames and relative subdirectories match the inputs.
Existing comparisons with matching filenames are overwritten.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from visualize_coco import IMAGE_EXTENSIONS, load_font

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUAL_CHECKS = PROJECT_ROOT / "runs/baseline-001/eval-epoch19/visual-checks"
GUTTER = 16
HEADER_HEIGHT = 36


def image_files(directory: Path) -> dict[Path, Path]:
    return {path.relative_to(directory): path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


def compose_pair(ground_truth_path: Path, prediction_path: Path) -> Image.Image:
    """Preserve pixel coordinates by rejecting mismatched dimensions."""
    with Image.open(ground_truth_path) as source:
        ground_truth = source.convert("RGB")
    with Image.open(prediction_path) as source:
        prediction = source.convert("RGB")
    if ground_truth.size != prediction.size:
        raise ValueError(
            f"Dimension mismatch for {ground_truth_path.name}: "
            f"ground truth {ground_truth.size}, predictions {prediction.size}"
        )

    width, height = ground_truth.size
    comparison = Image.new("RGB", (2 * width + GUTTER, height + HEADER_HEIGHT),
                           (240, 240, 240))
    comparison.paste(ground_truth, (0, HEADER_HEIGHT))
    comparison.paste(prediction, (width + GUTTER, HEADER_HEIGHT))
    draw = ImageDraw.Draw(comparison)
    for text, left in (("Ground truth", 0), ("Predictions", width + GUTTER)):
        font_size = 18
        while True:
            font = load_font(font_size)
            bounds = draw.textbbox((0, 0), text, font=font)
            if bounds[2] - bounds[0] <= width - 12 or font_size <= 6:
                break
            font_size -= 1
        top = (HEADER_HEIGHT - (bounds[3] - bounds[1])) // 2 - bounds[1]
        draw.text((left + 6 - bounds[0], top), text, font=font, fill=(20, 20, 20))
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path,
                        default=VISUAL_CHECKS / "ground-truth-sizes")
    parser.add_argument("--predictions-dir", type=Path,
                        default=VISUAL_CHECKS / "predictions")
    parser.add_argument("--output-dir", type=Path,
                        default=VISUAL_CHECKS / "comparison")
    parser.add_argument("--limit", type=int, help="Create only the first N comparisons")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    ground_truth_dir = args.ground_truth_dir.expanduser().resolve()
    predictions_dir = args.predictions_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    written = 0
    try:
        for directory in (ground_truth_dir, predictions_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Input directory does not exist: {directory}")
            if output_dir.is_relative_to(directory) or directory.is_relative_to(output_dir):
                raise ValueError("The output directory must be separate from both input directories")

        ground_truth = image_files(ground_truth_dir)
        predictions = image_files(predictions_dir)
        missing_predictions = sorted(ground_truth.keys() - predictions.keys())
        missing_ground_truth = sorted(predictions.keys() - ground_truth.keys())
        if missing_predictions or missing_ground_truth:
            raise ValueError(
                f"Unpaired images: {len(missing_predictions)} missing predictions "
                f"(examples: {[str(p) for p in missing_predictions[:5]]}); "
                f"{len(missing_ground_truth)} missing ground truth "
                f"(examples: {[str(p) for p in missing_ground_truth[:5]]})"
            )
        if not ground_truth:
            raise ValueError("No matching images found")

        names = sorted(ground_truth)
        print(f"Matched image pairs: {len(names)}", flush=True)
        if args.limit is not None:
            names = names[:args.limit]
        print(f"Saving {len(names)} comparisons to: {output_dir}", flush=True)
        for name in names:
            destination = (output_dir / name).resolve()
            if output_dir not in destination.parents:
                raise ValueError(f"Output path escapes comparison directory: {destination}")
            comparison = compose_pair(ground_truth[name], predictions[name])
            destination.parent.mkdir(parents=True, exist_ok=True)
            options = {"quality": 95} if name.suffix.lower() in (".jpg", ".jpeg") else {}
            comparison.save(destination, **options)
            written += 1
            if written == 1 or written % 100 == 0 or written == len(names):
                print(f"[{written}/{len(names)}] comparisons saved", flush=True)
        print(f"Complete: {written} comparisons saved.")
        return 0
    except Exception as error:
        print(f"[error] Stopped after {written} comparisons: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

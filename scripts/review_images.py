"""Save selected detection errors as annotated images for manual review."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

SEED = 0
N_FN_NOT_DETECTED = 50
N_FN_LOW_CONF = 10
SCORE_THRESH = 0.5
ROOT = Path(__file__).resolve().parent.parent
MATCHES_PATH = ROOT / "runs/baseline-001/eval-epoch19/error-analysis/matches.csv"
PREDICTIONS_PATH = ROOT / "runs/baseline-001/eval-epoch19/fresh-evaluation/predictions.json"
GT_PATH = ROOT / "data/raw/annotations/kaggle_val_annotations.json"
IMAGES_DIR = ROOT / "data/raw/valid"
OUTPUT_DIR = ROOT / "runs/baseline-001/eval-epoch19/error-analysis/review"


def draw_box(image, box, scale, color, thickness):
    """Draw an original-pixel XYWH box at the display scale."""
    x, y, width, height = box
    start = (round(x * scale), round(y * scale))
    end = (round((x + width) * scale), round((y + height) * scale))
    cv2.rectangle(image, start, end, color, thickness)


def main():
    """Select error rows, draw each target, and save images and blank review tags."""
    with MATCHES_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(enumerate(csv.DictReader(file)))
    selected = [(index, row) for index, row in rows if row["type"] == "FP"]
    rng = random.Random(SEED)
    for subtype, limit in (
        ("not_detected", N_FN_NOT_DETECTED),
        ("low_confidence", N_FN_LOW_CONF),
    ):
        candidates = [
            (index, row) for index, row in rows
            if row["type"] == "FN" and row["fn_type"] == subtype
        ]
        selected.extend(rng.sample(candidates, min(limit, len(candidates))))

    with GT_PATH.open(encoding="utf-8") as file:
        gt = json.load(file)
    with PREDICTIONS_PATH.open(encoding="utf-8") as file:
        predictions = json.load(file)
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    for annotation in gt["annotations"]:
        gt_by_image[annotation["image_id"]].append(annotation["bbox"])
    for prediction in predictions:
        if prediction["score"] >= SCORE_THRESH:
            pred_by_image[prediction["image_id"]].append(prediction["bbox"])

    for folder in ("fp", "fn"):
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
    tags = []
    folder_counts = Counter({"fp": 0, "fn": 0})
    subtype_counts = Counter({
        "duplicate": 0, "poor_localization": 0, "background": 0,
        "not_detected": 0, "low_confidence": 0,
    })
    for index, row in selected:
        image_path = IMAGES_DIR / row["file_name"]
        if not image_path.is_file():
            print(f"Warning: missing image {image_path}; skipping row {index}")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: unreadable image {image_path}; skipping row {index}")
            continue
        height, width = image.shape[:2]
        scale = max(1.0, 900 / max(height, width))
        if scale > 1:
            image = cv2.resize(
                image, (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        image_id = int(row["image_id"])
        for box in gt_by_image[image_id]:
            draw_box(image, box, scale, (0, 255, 0), 2)
        for box in pred_by_image[image_id]:
            draw_box(image, box, scale, (0, 0, 255), 2)
        is_fp = row["type"] == "FP"
        prefix = "pred" if is_fp else "gt"
        target = [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "w", "h")]
        draw_box(image, target, scale, (0, 0, 255) if is_fp else (0, 255, 0), 4)

        subtype = row["fp_type"] if is_fp else row["fn_type"]
        score_name = "score" if is_fp else "best_overlap_score"
        banner = [
            f'{row["type"]} | {subtype}',
            f'{score_name}: {row[score_name] or "empty"}',
            f'box_area_ratio: {row["box_area_ratio"]}',
        ]
        image = cv2.copyMakeBorder(image, 90, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for line_index, text in enumerate(banner):
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            font_scale = 0.6 * min(1.0, (image.shape[1] - 20) / max(text_width, 1))
            cv2.putText(image, text, (10, 23 + line_index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        folder = row["type"].lower()
        filename = f'{row["type"]}_{subtype}_{image_id}_{index}.jpg'
        output_path = OUTPUT_DIR / folder / filename
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not save {output_path}")
        tags.append({
            "filename": filename, "type": row["type"], "subtype": subtype,
            "image_id": image_id, "box_area_ratio": row["box_area_ratio"], "category": "",
        })
        folder_counts[folder] += 1
        subtype_counts[subtype] += 1

    with (OUTPUT_DIR / "review_tags.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "filename", "type", "subtype", "image_id", "box_area_ratio", "category",
        ])
        writer.writeheader()
        writer.writerows(tags)
    print("Images saved per folder:", dict(folder_counts))
    print("Images saved per subtype:", dict(subtype_counts))


if __name__ == "__main__":
    main()

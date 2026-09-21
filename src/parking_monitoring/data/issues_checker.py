from pathlib import Path
import csv
import json
from collections import defaultdict

from PIL import Image, ImageDraw

# ================= CONFIG =================
ISSUES_CSV = Path("./audit_report/issues.csv")
COCO_JSON = Path("../../data/raw/annotations/train_annotations.json")
IMAGE_DIR = Path("../../data/raw/train")
OUTPUT_DIR = Path("./audit_inspection")

TARGET_CHECKS = {
    "box_tiny",
    "box_suspiciously_large",
}
# ==========================================


def load_coco():
    data = json.loads(COCO_JSON.read_text(encoding="utf-8"))

    images = {
        img["file_name"]: img
        for img in data["images"]
    }

    annotations = defaultdict(list)

    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    return images, annotations


def select_issue_images():
    selected = set()

    with open(ISSUES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["check"] in TARGET_CHECKS:
                selected.add(Path(row["file"]).name)

    return selected


def draw_boxes():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images, annotations = load_coco()
    selected = select_issue_images()

    copied = 0

    for filename in selected:
        if filename not in images:
            print(f"Not found in COCO: {filename}")
            continue

        img_info = images[filename]
        image_id = img_info["id"]

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for ann in annotations[image_id]:
            x, y, w, h = ann["bbox"]

            draw.rectangle(
                [x, y, x + w, y + h],
                outline="red",
                width=3,
            )

            draw.text(
                (x, max(0, y - 15)),
                f"{ann['category_id']} {w:.0f}x{h:.0f}",
                fill="red",
            )

        image.save(OUTPUT_DIR / filename)
        copied += 1

    print(f"Selected: {len(selected)}")
    print(f"Saved:    {copied}")
    print(f"Output:   {OUTPUT_DIR}")


if __name__ == "__main__":
    draw_boxes()
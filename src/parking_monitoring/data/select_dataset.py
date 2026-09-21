"""
Select a clean baseline subset from a COCO object-detection dataset.

Selection policy
----------------
EXCLUDE:
1. Images with ERROR-level audit issues.
2. Orphan images (no corresponding annotation).
3. Images with more than one bounding box.

THEN:
4. Calculate bbox relative-area P5 and P95 on the remaining images.
5. Keep images whose bbox relative area is between P5 and P95 inclusive.

The raw dataset is never modified.
"""

from pathlib import Path
import csv
import json
import shutil

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input dataset
IMAGE_DIR = Path("../../data/raw/train")
COCO_JSON = Path("../../data/raw/annotations/train_annotations.json")

# Audit output
AUDIT_DIR = Path("./audit_report")
ISSUES_CSV = AUDIT_DIR / "issues.csv"

# Output dataset
OUTPUT_DIR = Path("../../data/selected_v0_1")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_COCO_JSON = OUTPUT_DIR / "annotations.json"

# Bbox relative-area selection
LOWER_PERCENTILE = 5
UPPER_PERCENTILE = 95

# Safety: do not overwrite an existing output directory
OVERWRITE = False

# ============================================================================


def load_coco(coco_path: Path) -> dict:
    """Load a COCO annotation JSON file."""
    with open(coco_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_error_images(issues_csv: Path) -> set[str]:
    """
    Return image paths that have ERROR-level audit issues.

    WARN-level issues do not automatically exclude an image.
    """
    error_images = set()

    with open(issues_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["severity"] == "ERROR":
                error_images.add(row["file"])

    return error_images


def image_has_error(
    image_path: Path,
    file_name: str,
    error_images: set[str],
) -> bool:
    """
    Match an image against the paths recorded by audit.py.

    The audit may store different representations of the same path,
    so check several reasonable forms.
    """
    return (
        str(image_path) in error_images
        or file_name in error_images
        or Path(file_name).name in error_images
        or image_path.name in error_images
    )


def get_relative_area(
    annotation: dict,
    image_width: int,
    image_height: int,
) -> float:
    """
    Calculate bounding-box area relative to image area.

    COCO bbox format:
        [x, y, width, height]
    """
    _, _, width, height = annotation["bbox"]

    return (width * height) / (image_width * image_height)


def main():
    print("=" * 70)
    print("DATASET SELECTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"IMAGE_DIR does not exist: {IMAGE_DIR}"
        )

    if not COCO_JSON.is_file():
        raise FileNotFoundError(
            f"COCO_JSON does not exist: {COCO_JSON}"
        )

    if not ISSUES_CSV.is_file():
        raise FileNotFoundError(
            f"ISSUES_CSV does not exist: {ISSUES_CSV}"
        )

    # ------------------------------------------------------------------
    # 2. Prepare output directory
    # ------------------------------------------------------------------

    if OUTPUT_DIR.exists():

        if not OVERWRITE:
            raise FileExistsError(
                f"Output directory already exists: {OUTPUT_DIR}\n"
                f"Set OVERWRITE = True if you intentionally want to replace it."
            )

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Load dataset
    # ------------------------------------------------------------------

    print("\n[1/6] Loading COCO dataset...")

    coco = load_coco(COCO_JSON)

    images = coco["images"]
    annotations = coco["annotations"]

    print(f"      images       : {len(images)}")
    print(f"      annotations  : {len(annotations)}")

    # image_id -> image
    image_by_id = {
        image["id"]: image
        for image in images
    }

    # image_id -> annotations
    annotations_by_image = {}

    for annotation in annotations:
        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    # ------------------------------------------------------------------
    # 4. Load audit errors
    # ------------------------------------------------------------------

    print("\n[2/6] Loading audit errors...")

    error_images = load_error_images(ISSUES_CSV)

    print(
        f"      images with ERROR issues : "
        f"{len(error_images)}"
    )

    # ------------------------------------------------------------------
    # 5. Apply structural selection rules
    # ------------------------------------------------------------------

    print("\n[3/6] Applying structural filters...")

    eligible_images = []

    excluded_errors = 0
    excluded_orphans = 0
    excluded_multi_bbox = 0

    for image in images:

        image_id = image["id"]
        file_name = image["file_name"]

        image_path = IMAGE_DIR / file_name

        # --------------------------------------------------------------
        # Rule 1: exclude ERROR-level audit issues
        # --------------------------------------------------------------

        if image_has_error(
            image_path,
            file_name,
            error_images,
        ):
            excluded_errors += 1
            continue

        # --------------------------------------------------------------
        # Rule 2: exclude orphan images
        #
        # An image is considered orphan if it has no annotation.
        # --------------------------------------------------------------

        image_annotations = annotations_by_image.get(
            image_id,
            []
        )

        if not image_annotations:
            excluded_orphans += 1
            continue

        # --------------------------------------------------------------
        # Rule 3: exclude images with >1 bbox
        # --------------------------------------------------------------

        if len(image_annotations) > 1:
            excluded_multi_bbox += 1
            continue

        # The image file itself must also exist.
        if not image_path.is_file():
            excluded_orphans += 1
            continue

        eligible_images.append(image)

    print(f"      ERROR issues excluded : {excluded_errors}")
    print(f"      orphan images excluded : {excluded_orphans}")
    print(f"      multi-bbox excluded    : {excluded_multi_bbox}")
    print(f"      eligible images        : {len(eligible_images)}")

    # ------------------------------------------------------------------
    # 6. Calculate bbox-area distribution
    # ------------------------------------------------------------------

    print("\n[4/6] Calculating bbox relative-area percentiles...")

    relative_areas = []

    for image in eligible_images:

        annotation = annotations_by_image[image["id"]][0]

        relative_area = get_relative_area(
            annotation,
            image["width"],
            image["height"],
        )

        relative_areas.append(relative_area)

    if not relative_areas:
        raise RuntimeError(
            "No eligible images remain after structural filtering."
        )

    p5, p95 = np.percentile(
        relative_areas,
        [LOWER_PERCENTILE, UPPER_PERCENTILE],
    )

    print(
        f"      P{LOWER_PERCENTILE} relative area  : "
        f"{p5:.8f}"
    )
    print(
        f"      P{UPPER_PERCENTILE} relative area : "
        f"{p95:.8f}"
    )

    # ------------------------------------------------------------------
    # 7. Apply bbox-area filter
    # ------------------------------------------------------------------

    print("\n[5/6] Selecting images by bbox relative area...")

    selected_images = []
    selected_annotations = []

    excluded_area = 0

    for image in eligible_images:

        annotation = annotations_by_image[image["id"]][0]

        relative_area = get_relative_area(
            annotation,
            image["width"],
            image["height"],
        )

        if p5 <= relative_area <= p95:
            selected_images.append(image)
            selected_annotations.append(annotation)
        else:
            excluded_area += 1

    print(f"      excluded by area : {excluded_area}")
    print(f"      selected images   : {len(selected_images)}")

    # ------------------------------------------------------------------
    # 8. Copy selected images
    # ------------------------------------------------------------------

    print("\n[6/6] Writing selected dataset...")

    for image in selected_images:

        src = IMAGE_DIR / image["file_name"]
        dst = OUTPUT_IMAGE_DIR / image["file_name"]

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # 9. Create new COCO annotation file
    # ------------------------------------------------------------------

    selected_image_ids = {
        image["id"]
        for image in selected_images
    }

    selected_annotations = [
        annotation
        for annotation in selected_annotations
        if annotation["image_id"] in selected_image_ids
    ]

    output_coco = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco["categories"],
    }

    with open(
        OUTPUT_COCO_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_coco,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)

    print(f"Input images          : {len(images)}")
    print(f"Input annotations     : {len(annotations)}")
    print(f"ERROR images removed  : {excluded_errors}")
    print(f"Orphan images removed : {excluded_orphans}")
    print(f">1 bbox removed       : {excluded_multi_bbox}")
    print(f"Area-filtered         : {excluded_area}")
    print(f"Selected images       : {len(selected_images)}")
    print(f"Selected annotations  : {len(selected_annotations)}")

    print(
        f"Relative-area P{LOWER_PERCENTILE}  : "
        f"{p5:.8f}"
    )
    print(
        f"Relative-area P{UPPER_PERCENTILE} : "
        f"{p95:.8f}"
    )

    print(f"\nOutput: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
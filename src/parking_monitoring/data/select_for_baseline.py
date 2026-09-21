"""
Create a representative baseline subset from a cleaned COCO dataset.

Sampling strategy
-----------------
1. Read the cleaned candidate dataset.
2. Calculate bbox relative area for each image.
3. Divide images into relative-area bins.
4. Sample proportionally from each bin.
5. Use a fixed random seed for reproducibility.
6. Copy selected images and annotations to a new dataset.

The input dataset is never modified.
"""

from pathlib import Path
import json
import shutil
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: cleaned candidate dataset
INPUT_DIR = Path("../../data/selected_v0_1")
INPUT_IMAGE_DIR = INPUT_DIR / "images"
INPUT_COCO_JSON = INPUT_DIR / "annotations.json"

# Output: baseline dataset
OUTPUT_DIR = Path("../../data/baseline_v1_0")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_COCO_JSON = OUTPUT_DIR / "annotations.json"

# Number of images to select
TARGET_IMAGES = 10_000

# Reproducibility
RANDOM_SEED = 42

# Number of relative-area strata
N_BINS = 5

# Safety: do not overwrite existing output
OVERWRITE = False

# ============================================================================


def load_coco(path: Path) -> dict:
    """Load COCO annotation file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_relative_area(image: dict, annotation: dict) -> float:
    """
    Calculate bbox area relative to image area.

    COCO bbox format:
        [x, y, width, height]
    """
    _, _, width, height = annotation["bbox"]

    return (width * height) / (
        image["width"] * image["height"]
    )


def main():

    print("=" * 70)
    print("BASELINE DATASET SAMPLING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------

    if not INPUT_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Input image directory does not exist:\n{INPUT_IMAGE_DIR}"
        )

    if not INPUT_COCO_JSON.is_file():
        raise FileNotFoundError(
            f"Input COCO JSON does not exist:\n{INPUT_COCO_JSON}"
        )

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------

    print("\n[1/5] Loading dataset...")

    coco = load_coco(INPUT_COCO_JSON)

    images = coco["images"]
    annotations = coco["annotations"]

    print(f"      images       : {len(images)}")
    print(f"      annotations  : {len(annotations)}")

    if len(images) < TARGET_IMAGES:
        raise ValueError(
            f"Dataset contains only {len(images)} images, "
            f"but TARGET_IMAGES={TARGET_IMAGES}."
        )

    # ------------------------------------------------------------------
    # 3. Build annotation index
    # ------------------------------------------------------------------

    annotations_by_image = {}

    for annotation in annotations:
        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    # We expect exactly one bbox per image because of the previous
    # dataset-selection step.
    invalid_images = []

    for image in images:
        image_id = image["id"]
        image_annotations = annotations_by_image.get(image_id, [])

        if len(image_annotations) != 1:
            invalid_images.append(image)

    if invalid_images:
        raise ValueError(
            f"Found {len(invalid_images)} images without exactly "
            f"one annotation. The input dataset does not satisfy "
            f"the expected baseline-selection assumptions."
        )

    # ------------------------------------------------------------------
    # 4. Calculate bbox relative areas
    # ------------------------------------------------------------------

    print("\n[2/5] Calculating bbox relative areas...")

    records = []

    for image in images:

        annotation = annotations_by_image[image["id"]][0]

        relative_area = get_relative_area(
            image,
            annotation,
        )

        records.append(
            {
                "image": image,
                "annotation": annotation,
                "relative_area": relative_area,
            }
        )

    # ------------------------------------------------------------------
    # 5. Create strata
    # ------------------------------------------------------------------

    print("\n[3/5] Creating relative-area strata...")

    areas = [
        record["relative_area"]
        for record in records
    ]

    min_area = min(areas)
    max_area = max(areas)

    if min_area == max_area:
        raise ValueError(
            "All images have the same bbox relative area. "
            "Stratification is not meaningful."
        )

    # Equal-width bins between minimum and maximum area.
    bin_width = (max_area - min_area) / N_BINS

    strata = [[] for _ in range(N_BINS)]

    for record in records:

        area = record["relative_area"]

        if area == max_area:
            bin_index = N_BINS - 1
        else:
            bin_index = int(
                (area - min_area) / bin_width
            )

        strata[bin_index].append(record)

    print("\n      Stratum distribution:")

    for i, stratum in enumerate(strata):

        lower = min_area + i * bin_width
        upper = lower + bin_width

        print(
            f"      {i + 1}: "
            f"{lower:.6f} - {upper:.6f} "
            f"-> {len(stratum)} images"
        )

    # ------------------------------------------------------------------
    # 6. Calculate proportional sample sizes
    # ------------------------------------------------------------------

    print("\n[4/5] Sampling proportionally...")

    total_images = len(records)

    sample_sizes = []

    for stratum in strata:

        proportion = len(stratum) / total_images

        sample_size = round(
            proportion * TARGET_IMAGES
        )

        sample_sizes.append(sample_size)

    # Fix rounding difference so total is exactly TARGET_IMAGES.
    difference = TARGET_IMAGES - sum(sample_sizes)

    if difference != 0:

        # Add/remove from the largest strata first.
        order = sorted(
            range(N_BINS),
            key=lambda i: len(strata[i]),
            reverse=True,
        )

        step = 1 if difference > 0 else -1

        for i in range(abs(difference)):
            sample_sizes[order[i % N_BINS]] += step

    print("\n      Samples per stratum:")

    for i, size in enumerate(sample_sizes):
        print(
            f"      {i + 1}: "
            f"{size} / {len(strata[i])}"
        )

    # ------------------------------------------------------------------
    # 7. Reproducible random sampling
    # ------------------------------------------------------------------

    rng = random.Random(RANDOM_SEED)

    selected_records = []

    for stratum, sample_size in zip(
        strata,
        sample_sizes,
    ):

        selected_records.extend(
            rng.sample(
                stratum,
                sample_size,
            )
        )

    # Shuffle final dataset order so strata aren't grouped together.
    rng.shuffle(selected_records)

    if len(selected_records) != TARGET_IMAGES:
        raise RuntimeError(
            f"Expected {TARGET_IMAGES} images, "
            f"but selected {len(selected_records)}."
        )

    selected_images = [
        record["image"]
        for record in selected_records
    ]

    selected_annotations = [
        record["annotation"]
        for record in selected_records
    ]

    # ------------------------------------------------------------------
    # 8. Prepare output
    # ------------------------------------------------------------------

    if OUTPUT_DIR.exists():

        if not OVERWRITE:
            raise FileExistsError(
                f"Output directory already exists:\n{OUTPUT_DIR}\n"
                f"Set OVERWRITE=True if you intentionally want "
                f"to replace it."
            )

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 9. Copy images
    # ------------------------------------------------------------------

    print("\n[5/5] Copying images...")

    copied = 0

    for image in selected_images:

        src = INPUT_IMAGE_DIR / image["file_name"]
        dst = OUTPUT_IMAGE_DIR / image["file_name"]

        if not src.is_file():
            raise FileNotFoundError(
                f"Selected image does not exist:\n{src}"
            )

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(src, dst)

        copied += 1

    # ------------------------------------------------------------------
    # 10. Write COCO annotations
    # ------------------------------------------------------------------

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
    # 11. Final summary
    # ------------------------------------------------------------------

    selected_areas = [
        record["relative_area"]
        for record in selected_records
    ]

    print("\n" + "=" * 70)
    print("BASELINE DATASET CREATED")
    print("=" * 70)

    print(f"Input images          : {len(images)}")
    print(f"Target images         : {TARGET_IMAGES}")
    print(f"Selected images       : {len(selected_images)}")
    print(f"Selected annotations  : {len(selected_annotations)}")
    print(f"Random seed           : {RANDOM_SEED}")
    print(f"Strata                : {N_BINS}")

    print(
        f"Selected area range   : "
        f"{min(selected_areas):.8f} - "
        f"{max(selected_areas):.8f}"
    )

    print(f"Images copied         : {copied}")

    print(f"\nOutput directory:")
    print(f"  {OUTPUT_DIR}")

    print(f"\nCOCO annotations:")
    print(f"  {OUTPUT_COCO_JSON}")

    print("=" * 70)


if __name__ == "__main__":
    main()
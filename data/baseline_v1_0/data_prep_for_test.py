import json
from pathlib import Path

print(Path.cwd())
DATASET_DIR = Path("baseline_v1_0")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"


def check_dataset_structure():
    print("=== 1. Dataset Structure ===")

    if not IMAGES_DIR.is_dir():
        print("❌ images/ folder does not exist")
        return False

    if not ANNOTATIONS_FILE.is_file():
        print("❌ annotations.json does not exist")
        return False

    print("✅ images/ exists")
    print("✅ annotations.json exists")
    return True


def load_coco():
    print("\n=== 2. Load COCO JSON ===")

    try:
        with open(ANNOTATIONS_FILE, "r") as f:
            coco = json.load(f)
    except Exception as e:
        print(f"❌ Cannot read annotations.json: {e}")
        return None

    print("✅ annotations.json is valid JSON")
    return coco


def validate_coco(coco):
    print("\n=== 3. Validate COCO Structure ===")

    required = ["images", "annotations", "categories"]

    for key in required:
        if key not in coco:
            print(f"❌ Missing '{key}'")
            return False

        if not isinstance(coco[key], list):
            print(f"❌ '{key}' must be a list")
            return False

        print(f"✅ '{key}' exists")

    # IDs from COCO
    image_ids = {image["id"] for image in coco["images"]}
    category_ids = {category["id"] for category in coco["categories"]}

    # Check duplicate image IDs
    if len(image_ids) != len(coco["images"]):
        print("❌ Duplicate image IDs found")
        return False

    # Check duplicate category IDs
    if len(category_ids) != len(coco["categories"]):
        print("❌ Duplicate category IDs found")
        return False

    print(f"✅ {len(image_ids)} unique image IDs")
    print(f"✅ {len(category_ids)} unique category IDs")

    # Check annotations
    for i, ann in enumerate(coco["annotations"]):

        if "image_id" not in ann:
            print(f"❌ Annotation {i}: missing image_id")
            return False

        if ann["image_id"] not in image_ids:
            print(
                f"❌ Annotation {i}: "
                f"image_id {ann['image_id']} does not exist"
            )
            return False

        if "category_id" not in ann:
            print(f"❌ Annotation {i}: missing category_id")
            return False

        if ann["category_id"] not in category_ids:
            print(
                f"❌ Annotation {i}: "
                f"category_id {ann['category_id']} does not exist"
            )
            return False

        bbox = ann.get("bbox")

        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"❌ Annotation {i}: bbox must be [x, y, width, height]")
            return False

        x, y, width, height = bbox

        if not all(isinstance(v, (int, float)) for v in bbox):
            print(f"❌ Annotation {i}: bbox contains non-numeric values")
            return False

        if width <= 0 or height <= 0:
            print(f"❌ Annotation {i}: width/height must be > 0")
            return False

    print(f"✅ {len(coco['annotations'])} annotations are valid")

    return True


def check_images_exist(coco):
    print("\n=== 4. Check Image Files ===")

    missing = []

    for image in coco["images"]:
        image_path = IMAGES_DIR / image["file_name"]

        if not image_path.is_file():
            missing.append(image["file_name"])

    if missing:
        print(f"❌ {len(missing)} referenced images are missing")

        for filename in missing[:10]:
            print(f"   - {filename}")

        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")

        return False

    print(f"✅ All {len(coco['images'])} referenced images exist")
    return True


def main():
    print("COCO Dataset Smoke-Test Validator")
    print("=" * 40)

    if not check_dataset_structure():
        return

    coco = load_coco()

    if coco is None:
        return

    coco_valid = validate_coco(coco)

    if not coco_valid:
        print("\n❌ COCO validation failed")
        return

    images_valid = check_images_exist(coco)

    if not images_valid:
        print("\n❌ Image validation failed")
        return

    print("\n" + "=" * 40)
    print("✅ DATASET VALIDATION PASSED")
    print("No files were modified.")


if __name__ == "__main__":
    main()
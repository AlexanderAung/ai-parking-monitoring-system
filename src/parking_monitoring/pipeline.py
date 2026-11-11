"""Original-image cropping and a manual geometry check; OCR orchestration is pending."""

import json
import math
from pathlib import Path
from uuid import uuid4

from PIL import Image

from parking_monitoring.config import load_config
from parking_monitoring.detector import RTDETRDetector


def crop_plate(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    padding_pixels: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Extract an original-resolution crop and return its exact bounds.

    Args:
        image: Nonempty original Pillow image. Preserve its source pixels.
        bbox_xyxy: Four finite Python int/float coordinates, ordered and
            expressed in original-image pixels. Booleans are invalid.
        padding_pixels: Nonnegative integer padding on all sides, excluding bool.

    Returns:
        An independent, nonempty crop preserving the source mode/pixel values,
        and integer (left, top, right, bottom) bounds after padding,
        outward rounding and clamping. The source image is preserved.
        Left/top are inclusive; right/bottom are exclusive.

    Raises:
        ValueError: Image, box or padding is invalid, or the unpadded box
            has no positive-area overlap with the image.
    """
    # 1. Validate the image: require a nonempty Pillow image.
    # Readability is checked when extracting its pixels in step 6.
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("image must be a nonempty Pillow image")

    # 2. Validate the box: require four finite Python int/float coordinates,
    # excluding booleans, then require x_min < x_max and y_min < y_max.
    if not isinstance(bbox_xyxy, tuple) or len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must be a tuple of four coordinates")
    for name, value in zip(
        ("x_min", "y_min", "x_max", "y_max"), bbox_xyxy, strict=True
    ):
        # Python integers are finite already; converting very large integers
        # to float merely for validation could overflow unnecessarily.
        if type(value) not in (int, float) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise ValueError(f"bbox_xyxy.{name} must be a finite int or float, excluding bool")
    x_min, y_min, x_max, y_max = bbox_xyxy
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("bbox_xyxy must satisfy x_min < x_max and y_min < y_max")

    # 3. Validate padding: require a nonnegative integer; reject booleans,
    # floats and strings rather than silently converting them.
    if type(padding_pixels) is not int or padding_pixels < 0:
        raise ValueError("padding_pixels must be a nonnegative integer, excluding bool")

    width, height = image.size
    # 4. Check original overlap before padding: reject wholly outside boxes.
    # Padding cannot rescue an outside box; partial overlap remains valid.
    if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
        raise ValueError("bbox_xyxy has no positive-area overlap with the image")

    # 5. Calculate crop bounds: expand all sides by padding, round left/top
    # down and right/bottom up, clamp x to [0, width] and y to [0, height],
    # then ensure the final region is nonempty.
    # For integer padding p, floor(x - p) == floor(x) - p (likewise ceil).
    # Use integer arithmetic to preserve this rule without float overflow or
    # losing fractional boundaries when padding is very large.
    left = max(0, min(width, math.floor(x_min) - padding_pixels))
    top = max(0, min(height, math.floor(y_min) - padding_pixels))
    right = max(0, min(width, math.ceil(x_max) + padding_pixels))
    bottom = max(0, min(height, math.ceil(y_max) + padding_pixels))
    if left >= right or top >= bottom:
        raise ValueError("Crop bounds are empty after rounding and clamping")

    bounds = (left, top, right, bottom)

    # 6. Extract original pixels into a separate crop, preserving the source
    # image, its mode and its pixel values.
    try:
        cropped_image = image.crop(bounds)
    except (OSError, ValueError) as exc:
        # 8. Report invalid/unusable inputs with descriptive ValueError messages.
        # Validation errors above name the input; pixel-read errors retain cause.
        raise ValueError(f"Cannot extract pixels from the source image: {exc}") from exc

    # 7. Return both the crop and the exact bounds used. Left/top are
    # inclusive; right/bottom are exclusive.
    return cropped_image, bounds


def main() -> None:
    """Save and display the first plate crop for the selected wide/tall examples."""
    d_config = load_config("configs/mvp.yaml")
    project_root = Path(__file__).resolve().parents[2]
    image_paths = (
        project_root / "data/raw/valid/pic_286_jpg.rf.40a15bf1e3a6971f689511c928707991.jpg",
        project_root / "data/mvp/images/image037_jpg.rf.83a6a458fd5ff401455f8cdda536a5e4.jpg",
    )
    detector = RTDETRDetector(d_config.detector)

    for image_path in image_paths:
        with Image.open(image_path) as img_a:
            print("Image:", image_path)
            print("Original image size:", img_a.size)
            detection = detector.predict(img_a)
            if not detection:
                print("No plate passed the detector threshold; skipping this image.")
                continue

            cropped_image, bounds = crop_plate(
                img_a,
                detection[0].bbox_xyxy,
                padding_pixels=d_config.crop_padding_pixels,
            )
            with cropped_image:
                label = "wide" if img_a.width > img_a.height else "tall"
                output_dir = d_config.storage.output_dir / f"{label}-check-{uuid4().hex[:8]}"
                output_dir.mkdir(parents=True)
                crop_path = output_dir / "crop.png"
                cropped_image.save(crop_path)
                details = {
                    "source": str(image_path),
                    "original_size_wh": img_a.size,
                    "detection_count": len(detection),
                    "detector_box_xyxy": detection[0].bbox_xyxy,
                    "detector_score": detection[0].score,
                    "crop_bounds_xyxy": bounds,
                    "crop_size_wh": cropped_image.size,
                    "padding_pixels": d_config.crop_padding_pixels,
                }
                (output_dir / "result.json").write_text(
                    json.dumps(details, indent=2) + "\n", encoding="utf-8"
                )
                print("Saved crop:", crop_path)
                print("Crop bounds:", bounds)
                print("Crop size:", cropped_image.size)
                cropped_image.show()


if __name__ == "__main__":
    main()

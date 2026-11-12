"""Shared result types for detection, OCR, and image processing.

Use ``dataclasses.asdict(result)`` to produce JSON-serializable data. Callers
should supply Python scalars rather than NumPy values or framework tensors.
These dataclasses describe the result contract; they do not perform runtime
validation, normalization, or confidence-threshold decisions.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Detection:
    """A plate box in original-image pixels and its detector score."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float


@dataclass
class OCRResult:
    """A plate reading, preserving the raw text and optional OCR score.

    Normalization uppercases text and removes whitespace without substituting
    characters. Missing text or scores use None. Status is ``read`` for text
    passing the acceptance rule, ``uncertain`` for text below the threshold,
    ``unreadable`` for a successful call without usable text, or ``error`` for
    a technical failure. A ``read`` status does not guarantee correctness.
    """

    raw_text: str | None
    normalized_text: str | None
    score: float | None
    status: Literal["read", "uncertain", "unreadable", "error"]
    error_message: str | None = None


@dataclass
class PlateResult:
    """One detection and its OCR result.

    Crop bounds are rounded and clamped original-image pixel coordinates.
    Left/top are inclusive; right/bottom are exclusive.
    """

    detection: Detection
    crop_bbox_xyxy: tuple[int, int, int, int]
    ocr: OCRResult


@dataclass
class ImageResult:
    """Results for one image, independent of UI, storage, and ground truth.

    ``image_size_wh`` stores original width and height. An empty ``plates``
    list means no plate passed the detector threshold.

    ``timings_ms`` uses detection, cropping, ocr, and total keys. Durations
    exclude model initialization; total is measured across image processing.
    ``model_info`` identifies the detector checkpoint and OCR model, using
    detector_checkpoint and ocr_model keys. It may include version/hash details.
    Image pixels, expected text, and persistence metadata are stored separately.
    """

    source_id: str
    image_size_wh: tuple[int, int]
    plates: list[PlateResult]
    timings_ms: dict[str, float]
    model_info: dict[str, str]

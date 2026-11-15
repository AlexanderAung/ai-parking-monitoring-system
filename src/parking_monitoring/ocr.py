"""Recognize Latin plate text using local English PaddleOCR 2.x models."""

import importlib
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

from parking_monitoring.config import OCRConfig
from parking_monitoring.schemas import OCRResult


@dataclass(frozen=True)
class _TextRegion:
    """Validated backend text and its crop-relative axis-aligned extent."""

    left: float
    top: float
    right: float
    bottom: float
    text: str
    score: float | None


class PaddleOCREngine:
    """Reuse locally initialized PaddleOCR models across plate crops."""

    def __init__(self, config: OCRConfig) -> None:
        """Validate resources and initialize local OCR models once.

        Args:
            config: English PP-OCRv4 settings, local model paths and
                acceptance threshold. Requires the PaddleOCR 2.x API.

        Raises:
            FileNotFoundError: A required local model file is missing.
            ValueError: Settings are invalid.
            RuntimeError: Backend or model initialization fails.

        Must not automatically download models.
        """
        # Validate settings. Check that the supplied OCR configuration has
        # supported values and valid types/ranges.
        if not isinstance(config, OCRConfig):
            # Preserve the approved ValueError contract for invalid settings.
            raise ValueError("config must be an OCRConfig")  # noqa: TRY004
        if config.engine != "paddleocr":
            raise ValueError("ocr.engine must be 'paddleocr'")
        if config.language != "en" or config.model_version != "PP-OCRv4":
            raise ValueError(
                "This local MVP wrapper requires language='en', model_version='PP-OCRv4'"
            )
        for field in ("use_gpu", "use_angle_cls", "enable_mkldnn"):
            if type(getattr(config, field)) is not bool:
                raise ValueError(f"ocr.{field} must be a boolean")
        if type(config.cpu_threads) is not int or config.cpu_threads < 1:
            raise ValueError(
                "ocr.cpu_threads must be a positive integer, excluding bool"
            )
        if (
            type(config.acceptance_threshold) not in (int, float)
            or not 0 <= config.acceptance_threshold <= 1
        ):
            raise ValueError(
                "ocr.acceptance_threshold must be a finite number in [0, 1]"
            )

        # Check local resources. Verify the required model files before
        # backend initialization.
        # The installed PaddleOCR constructor checks all three model directories,
        # even with angle classification disabled; prevent missing files from triggering downloads.
        model_dirs: dict[str, Path] = {}
        for name in ("det", "rec", "cls"):
            directory = getattr(config, f"{name}_model_dir")
            if not isinstance(directory, Path):
                raise ValueError(f"ocr.{name}_model_dir must be a pathlib.Path")  # noqa: TRY004
            # Use resolved local directories, never a URL or a default cache.
            try:
                directory = directory.resolve()
                for filename in ("inference.pdmodel", "inference.pdiparams"):
                    model_file = directory / filename
                    if not model_file.is_file():
                        raise FileNotFoundError(
                            f"Missing local OCR {name} model file: {model_file}"
                        )
                    with model_file.open("rb") as stream:
                        if not stream.read(1):
                            raise RuntimeError(
                                f"Empty local OCR {name} model file: {model_file}"
                            )
            except FileNotFoundError:
                raise
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Cannot use local OCR {name} model at {directory}: {exc}"
                ) from exc
            model_dirs[name] = directory

        # Initialize and retain the backend.
        # Pass explicit configured settings, preserve low-confidence readings,
        # and keep one reusable OCR instance.
        # Report backend initialization failures as RuntimeError, preserving their cause.
        try:
            installed_version = version("paddleocr")
            if installed_version.split(".", 1)[0] != "2":
                raise RuntimeError(
                    f"This wrapper requires PaddleOCR 2.x; installed {installed_version}"
                )
            # Delay heavyweight imports until resources/settings are valid.
            paddleocr = importlib.import_module("paddleocr")
            if config.use_gpu:
                paddle = importlib.import_module("paddle")
                if (
                    not paddle.is_compiled_with_cuda()
                    or paddle.device.cuda.device_count() < 1
                ):
                    raise RuntimeError(
                        "ocr.use_gpu=True requires an available CUDA Paddle device"
                    )
            self._ocr = paddleocr.PaddleOCR(
                lang=config.language,
                ocr_version=config.model_version,
                use_gpu=config.use_gpu,
                cpu_threads=config.cpu_threads,
                use_angle_cls=config.use_angle_cls,
                enable_mkldnn=config.enable_mkldnn,
                det_model_dir=str(model_dirs["det"]),
                rec_model_dir=str(model_dirs["rec"]),
                cls_model_dir=str(model_dirs["cls"]),
                drop_score=0.0,
                use_onnx=False,
                save_crop_res=False,
                show_log=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize local PaddleOCR: {exc}") from exc
        self._config = config

    def recognize(self, crop: Image.Image) -> OCRResult:
        """Recognize one plate crop without modifying its pixels.

        Args:
            crop: A usable, nonempty Pillow plate image.

        Returns:
            Ordered raw text, normalized text, optional score and
            read/uncertain/unreadable/error status. Technical recognition
            failures return error status with an explanatory message.
            Text regions are joined with spaces; normalization uppercases
            and removes whitespace without substituting characters. The
            plate score is the minimum region score, or None if any score
            is missing. Missing/below-threshold scores give uncertain text.

        Raises:
            ValueError: The input crop is invalid.
        """

        # Validate the crop. Require a usable, nonempty Pillow image;
        # invalid input raises ValueError.
        if not isinstance(crop, Image.Image) or crop.width <= 0 or crop.height <= 0:
            raise ValueError("crop must be a nonempty Pillow image")

        # Prepare a separate image copy. Convert it into the backend’s required
        # array/color format without modifying the original crop.
        try:
            with crop.convert("RGB") as rgb:
                # PaddleOCR's OpenCV input is BGR; copy also removes negative
                # strides and makes the array independent of Pillow pixels.
                image_bgr = np.asarray(rgb, dtype=np.uint8)[:, :, ::-1].copy()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read crop pixels: {exc}") from exc

        # Run OCR. Perform text detection and recognition, using the configured
        # angle-classifier setting. Technical inference failures return an error result.
        try:
            output = self._ocr.ocr(
                image_bgr, det=True, rec=True, cls=self._config.use_angle_cls
            )
        except Exception as exc:  # noqa: BLE001
            # Backend operators raise several exception types. This approved
            # boundary returns a crop-local failure; BaseException propagates.
            return OCRResult(
                None,
                None,
                None,
                "error",
                f"OCR inference failed ({type(exc).__name__}): {exc}",
            )

        # Validate and unpack the output. Extract text regions, positions,
        # and optional scores. Distinguish empty successful output from malformed backend output.
        try:
            # One ndarray input produces one page. PaddleOCR 2.x uses [None]
            # for no detections; [] and [[]] likewise contain no regions.
            if not isinstance(output, list) or len(output) > 1:
                raise ValueError("Expected a single-image PaddleOCR result list")
            page = output[0] if output else None
            if page is None:
                page = []
            if not isinstance(page, (list, tuple)):
                raise TypeError(
                    "OCR image result must contain a sequence of text regions"
                )
            regions: list[_TextRegion] = []
            for index, item in enumerate(page):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(
                        f"OCR region {index} must contain a box and reading"
                    )
                box, reading = item
                points = np.asarray(box)
                if (
                    points.shape != (4, 2)
                    or points.dtype.kind not in "iuf"
                    or not np.isfinite(points).all()
                ):
                    raise ValueError(
                        f"OCR region {index} must have four finite numeric corners"
                    )
                left, top = (float(value) for value in points.min(axis=0))
                right, bottom = (float(value) for value in points.max(axis=0))
                if (
                    not all(
                        math.isfinite(value) for value in (left, top, right, bottom)
                    )
                    or left >= right
                    or top >= bottom
                    or not math.isfinite(right - left)
                    or not math.isfinite(bottom - top)
                ):
                    raise ValueError(
                        f"OCR region {index} has an empty or unrepresentable text box"
                    )
                if (
                    not isinstance(reading, (list, tuple))
                    or len(reading) != 2
                    or not isinstance(reading[0], str)
                ):
                    raise ValueError(
                        f"OCR region {index} must contain (text, optional score)"
                    )
                text, score = reading
                if score is not None:
                    if isinstance(score, (bool, np.bool_)) or not isinstance(
                        score, (int, float, np.integer, np.floating)
                    ):
                        raise ValueError(
                            f"OCR region {index} score must be numeric or None"
                        )
                    score = float(score)
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(
                            f"OCR region {index} score must be finite and in [0, 1]"
                        )
                regions.append(_TextRegion(left, top, right, bottom, text, score))
        except (TypeError, ValueError, OverflowError) as exc:
            return OCRResult(None, None, None, "error", f"Invalid OCR output: {exc}")

        # Establish reading order. Group regions into rows, order rows top
        # to bottom and regions left to right, then join their text with spaces into raw_text.
        rows: list[list[_TextRegion]] = []
        for region in sorted(
            regions, key=lambda item: (item.top / 2 + item.bottom / 2, item.left)
        ):
            center = region.top / 2 + region.bottom / 2
            height = region.bottom - region.top
            matching_row = None
            closest_distance = math.inf
            for row in rows:
                row_center = median(item.top / 2 + item.bottom / 2 for item in row)
                row_height = median(item.bottom - item.top for item in row)
                distance = abs(center - row_center)
                # A scale-relative heuristic tolerates small within-row shifts.
                # Median row geometry avoids one large region merging two rows.
                if (
                    distance <= 0.5 * min(height, row_height)
                    and distance < closest_distance
                ):
                    matching_row, closest_distance = row, distance
            if matching_row is None:
                rows.append([region])
            else:
                matching_row.append(region)
        rows.sort(
            key=lambda row: median(item.top / 2 + item.bottom / 2 for item in row)
        )
        ordered_regions = [
            item
            for row in rows
            for item in sorted(row, key=lambda item: (item.left, item.top))
        ]
        raw_text = " ".join(item.text for item in ordered_regions) or None

        # Normalize the text. Uppercase and remove whitespace to produce
        # normalized_text. Preserve recognized characters without substitutions.
        normalized_text = None
        if raw_text is not None:
            normalized_text = "".join(raw_text.upper().split()) or None

        # Build OCRResult. Use the minimum region score when scores are available, then apply:
        scores = [item.score for item in ordered_regions if item.score is not None]
        score = min(scores) if scores and len(scores) == len(ordered_regions) else None
        if normalized_text is None:
            return OCRResult(raw_text, None, score, "unreadable")
        if score is None or score < self._config.acceptance_threshold:
            return OCRResult(raw_text, normalized_text, score, "uncertain")
        return OCRResult(raw_text, normalized_text, score, "read")

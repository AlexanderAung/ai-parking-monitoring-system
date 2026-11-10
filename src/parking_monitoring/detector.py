"""RT-DETR checkpoint loading and plate inference. Implementation pending."""

from PIL import Image

from parking_monitoring.config import DetectorConfig
from parking_monitoring.schemas import Detection


class RTDETRDetector:
    """Reusable plate detector with original-image coordinate output."""

    def __init__(self, config: DetectorConfig) -> None:
        """Validate detector resources and load the model once.

        Args:
            config: Detector paths, device and score threshold.

        Raises:
            FileNotFoundError: A required detector file is missing.
            ValueError: Detector settings are invalid.
            RuntimeError: Model, EMA weights or device cannot initialize.
        """
        raise NotImplementedError

    def predict(self, image: Image.Image) -> list[Detection]:
        """Preprocess an image, run inference and postprocess predictions.

        Args:
            image: A nonempty Pillow image. Preserve its original pixels.

        Returns:
            Detections with floating-point xyxy boxes in original-image
            pixels and Python float scores above the configured threshold.
            Returns an empty list when no detection passes the threshold.
            Cropping handles subsequent rounding and clamping.

        Raises:
            ValueError: The image input is invalid.
            RuntimeError: Inference or postprocessing fails.
        """

        raise NotImplementedError

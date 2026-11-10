"""Reusable RT-DETR inference without constructing an annotation-based loader."""

import importlib
import sys
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image

from parking_monitoring.config import DetectorConfig
from parking_monitoring.schemas import Detection


def _load_backend() -> tuple[type, type]:
    """Import the repository's vendored model builder and transform container."""
    backend_root = (
        Path(__file__).resolve().parents[2]
        / "thrid-party/RT-DETR/rtdetrv2_pytorch"
    )
    if not (backend_root / "src/core/__init__.py").is_file():
        raise FileNotFoundError(f"Missing RT-DETR source tree: {backend_root}")

    # The upstream package is named 'src'; use the same import root as the
    # existing validation script, without importing that script or its CLI.
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    core = importlib.import_module("src.core")
    if Path(core.__file__).resolve().parent != backend_root / "src/core":
        raise RuntimeError("A different 'src' package is shadowing the RT-DETR backend")
    transforms = importlib.import_module("src.data.transforms")
    return core.YAMLConfig, transforms.Compose


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
        if not isinstance(config, DetectorConfig):
            raise ValueError("config must be a DetectorConfig")
        if (
            type(config.score_threshold) not in (int, float)
            or not 0 <= config.score_threshold <= 1
        ):
            raise ValueError("detector.score_threshold must be a finite number in [0, 1]")
        if not isinstance(config.device, str) or not config.device.strip():
            raise ValueError("detector.device must be a nonempty device string")

        try:
            device = torch.device(config.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid detector.device {config.device!r}: {exc}") from exc

        for label, path in (
            ("model configuration", config.config_path),
            ("checkpoint", config.checkpoint_path),
        ):
            if not isinstance(path, Path):
                raise ValueError(f"Detector {label} must be a pathlib.Path")
            if not path.is_file():
                raise FileNotFoundError(f"Detector {label} file not found: {path}")

        try:
            YAMLConfig, Compose = _load_backend()
            cfg = YAMLConfig(str(config.config_path))
            if cfg.yaml_cfg.get("num_classes") != 1 or cfg.yaml_cfg.get(
                "remap_mscoco_category", False
            ):
                raise ValueError("The plate detector requires one class without COCO remapping")

            # Restore the complete local training checkpoint instead of fetching
            # a separate pretrained backbone, as in predict_validation.py.
            cfg.yaml_cfg.setdefault("PResNet", {})["pretrained"] = False
            state = torch.load(
                config.checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                not isinstance(state, dict)
                or not isinstance(state.get("ema"), dict)
                or not isinstance(state["ema"].get("module"), dict)
            ):
                raise ValueError("Checkpoint has no EMA module; this baseline requires EMA weights")
            model = cfg.model
            model.load_state_dict(state["ema"]["module"], strict=True)
            del state
            self._model = model.to(device).eval()
            self._postprocessor = cfg.postprocessor.to(device).eval()

            # Build only transforms. Never access cfg.val_dataloader, which
            # would require COCO images/annotations just to predict one image.
            transform_config = deepcopy(
                cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]
            )
            if transform_config.pop("type") != "Compose":
                raise ValueError("Validation transforms must use Compose")
            self._preprocess = Compose(**transform_config)
            size = cfg.yaml_cfg["eval_spatial_size"]
            if (
                not isinstance(size, (list, tuple))
                or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
            ):
                raise ValueError("eval_spatial_size must contain positive height and width")
            self._input_size_hw = (size[0], size[1])
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Cannot initialize RT-DETR from {config.config_path} with EMA "
                f"checkpoint {config.checkpoint_path} on {device}: {exc}"
            ) from exc

        self._device = device
        self._score_threshold = float(config.score_threshold)

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

        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty Pillow image")
        width, height = image.size
        try:
            # convert returns a separate image, including for RGB inputs. The
            # original remains available at full resolution for safe cropping.
            rgb = image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read image pixels: {exc}") from exc

        try:
            with torch.inference_mode(), rgb:
                tensor = self._preprocess(rgb)
                expected_shape = (3, *self._input_size_hw)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != expected_shape
                    or tensor.dtype != torch.float32
                ):
                    raise ValueError(
                        f"Preprocessing must return a float32 tensor shaped {expected_shape}"
                    )
                samples = tensor.unsqueeze(0).to(self._device)
                # RT-DETR uses [width, height] here, not [height, width]. Its
                # postprocessor maps normalized boxes to original pixels once.
                original_sizes = torch.tensor(
                    [[width, height]], dtype=torch.int64, device=self._device
                )
                outputs = self._model(samples)
                results = self._postprocessor(outputs, original_sizes)
                if not isinstance(results, list) or len(results) != 1:
                    raise ValueError("Postprocessor must return one image result")
                result = results[0]
                boxes = result["boxes"].detach().cpu()
                scores = result["scores"].detach().cpu()
                labels = result["labels"].detach().cpu()
                if (
                    boxes.ndim != 2
                    or boxes.shape[1] != 4
                    or scores.ndim != 1
                    or labels.ndim != 1
                    or len(boxes) != len(scores)
                    or len(boxes) != len(labels)
                ):
                    raise ValueError("Postprocessor returned inconsistent boxes, scores or labels")
                if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                    raise ValueError("Postprocessor returned non-finite coordinates or scores")
                if (
                    (labels != 0).any()
                    or ((scores < 0) | (scores > 1)).any()
                    or (boxes[:, 2:] <= boxes[:, :2]).any()
                ):
                    raise ValueError("Postprocessor returned invalid plate boxes, scores or labels")

                keep = scores > self._score_threshold
                detections = []
                for box, score in zip(boxes[keep].tolist(), scores[keep].tolist()):
                    x_min, y_min, x_max, y_max = box
                    detections.append(
                        Detection(
                            bbox_xyxy=(float(x_min), float(y_min), float(x_max), float(y_max)),
                            score=float(score),
                        )
                    )
                return detections
        except Exception as exc:
            raise RuntimeError(
                f"RT-DETR inference failed for a {width}x{height} image: {exc}"
            ) from exc

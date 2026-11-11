"""Load explicit, validated MVP settings without initializing models.

Relative configuration and resource paths resolve against the project root,
not the working directory. Components validate their own resource availability.
"""

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DetectorConfig:
    """Settings for detector initialization and filtering."""

    config_path: Path
    checkpoint_path: Path
    device: str
    score_threshold: float


@dataclass(frozen=True)
class OCRConfig:
    """Settings for OCR initialization and reading acceptance."""

    engine: str
    language: str
    model_version: str
    use_gpu: bool
    cpu_threads: int
    use_angle_cls: bool
    enable_mkldnn: bool
    det_model_dir: Path
    rec_model_dir: Path
    cls_model_dir: Path
    acceptance_threshold: float


@dataclass(frozen=True)
class StorageConfig:
    """Locations for persisted observations and output files."""

    database_path: Path
    crops_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class AppConfig:
    """Application settings grouped by component."""

    detector: DetectorConfig
    ocr: OCRConfig
    storage: StorageConfig
    crop_padding_pixels: int


class ConfigError(ValueError):
    """Configuration could not be read or contains invalid settings."""


def _mapping(value: Any, label: str, expected: set[str]) -> dict[str, Any]:
    """Require exactly the named fields so typos cannot silently become defaults."""
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{label} keys must be strings")

    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ConfigError(f"{label}: unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"{label}: missing fields: {', '.join(sorted(missing))}")
    return value


def _text(value: Any, label: str) -> str:
    """Require a nonempty string without silently converting other YAML types."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    """Reject quoted booleans such as 'false'."""
    if type(value) is not bool:
        raise ConfigError(f"{label} must be a boolean; received {value!r}")
    return value


def _integer(value: Any, label: str, minimum: int) -> int:
    """Require an integer at least minimum, excluding Python's boolean subtype."""
    if type(value) is not int or value < minimum:
        raise ConfigError(
            f"{label} must be an integer >= {minimum}; received {value!r}"
        )
    return value


def _threshold(value: Any, label: str) -> float:
    """Require a finite score in the inclusive range [0, 1]."""
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ConfigError(f"{label} must be a finite number in [0, 1]; received {value!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{label} must be finite; received {value!r}")
    return float(value)


def _resolve_path(value: str | Path, label: str, root: Path) -> Path:
    """Resolve a path without requiring its resource to exist or creating it."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"{label} must be a nonempty path")
    if "\x00" in str(value):
        raise ConfigError(f"{label} must not contain null characters")
    try:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"Cannot resolve {label}: {exc}") from exc


def _resource_path(value: Any, label: str, root: Path) -> Path:
    """Require a YAML path string, then resolve it against the project root."""
    return _resolve_path(_text(value, label), label, root)


def load_config(
    path: str | Path = "configs/mvp.yaml",
    *,
    project_root: str | Path | None = None,
) -> AppConfig:
    """Read YAML and return validated settings with absolute resource paths.

    Args:
        path: YAML file path, absolute or relative to the project root.
        project_root: Explicit root override. None uses the source-checkout root.

    Returns:
        Settings grouped by component, with resource paths resolved.

    Raises:
        ConfigError: If the YAML cannot be read or settings are invalid.

    Relative resource paths use the project root, including when the YAML
    is elsewhere. Does not check resource availability, initialize models,
    download files, or create outputs.
    """
    root = (
        PROJECT_ROOT
        if project_root is None
        else _resolve_path(project_root, "project_root", Path.cwd())
    )
    config_path = _resolve_path(path, "configuration path", root)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration {config_path}: {exc}") from exc

    raw = _mapping(raw, "configuration", {field.name for field in fields(AppConfig)})
    detector = _mapping(
        raw["detector"], "detector", {field.name for field in fields(DetectorConfig)}
    )
    ocr = _mapping(raw["ocr"], "ocr", {field.name for field in fields(OCRConfig)})
    storage = _mapping(
        raw["storage"], "storage", {field.name for field in fields(StorageConfig)}
    )

    engine = _text(ocr["engine"], "ocr.engine")
    if engine != "paddleocr":
        raise ConfigError("ocr.engine must be 'paddleocr'")

    return AppConfig(
        detector=DetectorConfig(
            config_path=_resource_path(
                detector["config_path"], "detector.config_path", root
            ),
            checkpoint_path=_resource_path(
                detector["checkpoint_path"], "detector.checkpoint_path", root
            ),
            device=_text(detector["device"], "detector.device"),
            score_threshold=_threshold(
                detector["score_threshold"], "detector.score_threshold"
            ),
        ),
        ocr=OCRConfig(
            engine=engine,
            language=_text(ocr["language"], "ocr.language"),
            model_version=_text(ocr["model_version"], "ocr.model_version"),
            use_gpu=_boolean(ocr["use_gpu"], "ocr.use_gpu"),
            cpu_threads=_integer(ocr["cpu_threads"], "ocr.cpu_threads", 1),
            use_angle_cls=_boolean(ocr["use_angle_cls"], "ocr.use_angle_cls"),
            enable_mkldnn=_boolean(ocr["enable_mkldnn"], "ocr.enable_mkldnn"),
            det_model_dir=_resource_path(
                ocr["det_model_dir"], "ocr.det_model_dir", root
            ),
            rec_model_dir=_resource_path(
                ocr["rec_model_dir"], "ocr.rec_model_dir", root
            ),
            cls_model_dir=_resource_path(
                ocr["cls_model_dir"], "ocr.cls_model_dir", root
            ),
            acceptance_threshold=_threshold(
                ocr["acceptance_threshold"], "ocr.acceptance_threshold"
            ),
        ),
        storage=StorageConfig(
            database_path=_resource_path(
                storage["database_path"], "storage.database_path", root
            ),
            crops_dir=_resource_path(storage["crops_dir"], "storage.crops_dir", root),
            output_dir=_resource_path(storage["output_dir"], "storage.output_dir", root),
        ),
        crop_padding_pixels=_integer(
            raw["crop_padding_pixels"], "crop_padding_pixels", 0
        ),
    )

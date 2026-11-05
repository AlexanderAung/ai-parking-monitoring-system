"""Load explicit, validated MVP settings without initializing models.

Relative configuration and resource paths resolve against the project root,
not the working directory. See config.md for the design and implementation log.
"""

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """An actionable error in the configuration or its referenced paths."""


@dataclass(frozen=True)
class DetectorConfig:
    config_path: Path
    checkpoint_path: Path
    device: str
    score_threshold: float


@dataclass(frozen=True)
class OCRConfig:
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
    database_path: Path
    crops_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class AppConfig:
    detector: DetectorConfig
    ocr: OCRConfig
    storage: StorageConfig
    crop_padding_pixels: int


def _mapping(value: Any, label: str, expected: set[str]) -> dict:
    """Require a mapping with exactly the expected keys."""
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ConfigError(
            f"{label}: unknown keys: {', '.join(sorted(map(str, unknown)))}"
        )
    if missing:
        raise ConfigError(f"{label}: missing keys: {', '.join(sorted(missing))}")
    return value


def _choice(value: Any, label: str, allowed: set[str]) -> str:
    """Validate a supported string setting without silently coercing values."""
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(
            f"{label} must be one of {sorted(allowed)}; received {value!r}"
        )
    return value


def _boolean(value: Any, label: str) -> bool:
    """Require YAML booleans, rejecting quoted strings such as 'false'."""
    if type(value) is not bool:
        raise ConfigError(f"{label} must be a boolean; received {value!r}")
    return value


def _integer(value: Any, label: str, minimum: int) -> int:
    """Require an integer at least minimum; booleans are not integers here."""
    if type(value) is not int or value < minimum:
        raise ConfigError(
            f"{label} must be an integer >= {minimum}; received {value!r}"
        )
    return value


def _threshold(value: Any, label: str) -> float:
    """Require a finite numeric score threshold in the inclusive range [0, 1]."""
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ConfigError(f"{label} must be between 0 and 1; received {value!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{label} must be finite; received {value!r}")
    return float(value)


def _path(value: Any, label: str, root: Path, kind: str) -> Path:
    """Resolve paths, require inputs, and check existing output path types."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty path string")
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if kind == "input_file" and not path.is_file():
        raise ConfigError(f"{label}: required file not found: {path}")
    if kind == "input_dir" and not path.is_dir():
        raise ConfigError(f"{label}: required directory not found: {path}")
    if kind == "output_file" and path.exists() and not path.is_file():
        raise ConfigError(f"{label}: expected an output file path: {path}")
    if kind == "output_dir" and path.exists() and not path.is_dir():
        raise ConfigError(f"{label}: expected an output directory path: {path}")
    # A nonexistent output is valid, but its nearest existing ancestor must
    # be a directory. Do not create directories or probe by writing files.
    if kind.startswith("output"):
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir():
            raise ConfigError(f"{label}: output parent is not a directory: {parent}")
    return path


def load_config(
    path: str | Path = "configs/mvp.yaml", *, project_root: str | Path | None = None
) -> AppConfig:
    """Read YAML and return validated settings with absolute paths.

    project_root can override the source-checkout root for tests or another
    deployment layout. Relative paths within YAML always use this root, even
    when the YAML file itself is outside it. No models are loaded or downloaded,
    and no output files or directories are created.
    """
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    config_path = Path(path).expanduser()
    config_path = (
        config_path if config_path.is_absolute() else root / config_path
    ).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot load configuration {config_path}: {exc}") from exc

    raw = _mapping(raw, "configuration", {field.name for field in fields(AppConfig)})
    detector = _mapping(
        raw["detector"], "detector", {field.name for field in fields(DetectorConfig)}
    )
    ocr = _mapping(raw["ocr"], "ocr", {field.name for field in fields(OCRConfig)})
    storage = _mapping(
        raw["storage"], "storage", {field.name for field in fields(StorageConfig)}
    )

    detector_config = DetectorConfig(
        config_path=_path(
            detector["config_path"], "detector.config_path", root, "input_file"
        ),
        checkpoint_path=_path(
            detector["checkpoint_path"], "detector.checkpoint_path", root, "input_file"
        ),
        device=_choice(detector["device"], "detector.device", {"cpu"}),
        score_threshold=_threshold(
            detector["score_threshold"], "detector.score_threshold"
        ),
    )
    use_gpu = _boolean(ocr["use_gpu"], "ocr.use_gpu")
    if use_gpu:
        raise ConfigError(
            "ocr.use_gpu must be false: this MVP currently supports CPU only"
        )
    ocr_config = OCRConfig(
        engine=_choice(ocr["engine"], "ocr.engine", {"paddleocr"}),
        language=_choice(ocr["language"], "ocr.language", {"en"}),
        model_version=_choice(ocr["model_version"], "ocr.model_version", {"PP-OCRv4"}),
        use_gpu=use_gpu,
        cpu_threads=_integer(ocr["cpu_threads"], "ocr.cpu_threads", 1),
        use_angle_cls=_boolean(ocr["use_angle_cls"], "ocr.use_angle_cls"),
        enable_mkldnn=_boolean(ocr["enable_mkldnn"], "ocr.enable_mkldnn"),
        det_model_dir=_path(
            ocr["det_model_dir"], "ocr.det_model_dir", root, "input_dir"
        ),
        rec_model_dir=_path(
            ocr["rec_model_dir"], "ocr.rec_model_dir", root, "input_dir"
        ),
        cls_model_dir=_path(
            ocr["cls_model_dir"], "ocr.cls_model_dir", root, "input_dir"
        ),
        acceptance_threshold=_threshold(
            ocr["acceptance_threshold"], "ocr.acceptance_threshold"
        ),
    )
    # PaddleOCR 2.10 initializes/download-checks all three model directories,
    # even with angle classification disabled. Require local files to avoid it.
    for name in ("det_model_dir", "rec_model_dir", "cls_model_dir"):
        directory = getattr(ocr_config, name)
        for filename in ("inference.pdmodel", "inference.pdiparams"):
            if not (directory / filename).is_file():
                raise ConfigError(
                    f"ocr.{name}: required file not found: {directory / filename}"
                )

    return AppConfig(
        detector=detector_config,
        ocr=ocr_config,
        storage=StorageConfig(
            database_path=_path(
                storage["database_path"], "storage.database_path", root, "output_file"
            ),
            crops_dir=_path(
                storage["crops_dir"], "storage.crops_dir", root, "output_dir"
            ),
            output_dir=_path(
                storage["output_dir"], "storage.output_dir", root, "output_dir"
            ),
        ),
        crop_padding_pixels=_integer(
            raw["crop_padding_pixels"], "crop_padding_pixels", 0
        ),
    )

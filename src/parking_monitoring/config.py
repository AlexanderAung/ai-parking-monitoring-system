"""Load explicit, validated MVP settings without initializing models.

Relative configuration and resource paths resolve against the project root,
not the working directory. See config.md for the design and implementation log.
"""
from dataclasses import dataclass 
from pathlib import Path 


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
    raise NotImplementedError

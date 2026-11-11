"""Configuration contract tests using temporary YAML and no model assets."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from parking_monitoring.config import AppConfig, ConfigError, load_config


@pytest.fixture
def settings() -> dict[str, Any]:
    """Use non-default values so ignored settings cannot pass unnoticed."""
    return {
        "detector": {
            "config_path": "detector/eval.yml",
            "checkpoint_path": "models/checkpoint.pth",
            "device": "cpu",
            "score_threshold": 0.75,
        },
        "ocr": {
            "engine": "paddleocr",
            "language": "en",
            "model_version": "PP-OCRv4",
            "use_gpu": False,
            "cpu_threads": 3,
            "use_angle_cls": True,
            "enable_mkldnn": True,
            "det_model_dir": "models/ocr/det",
            "rec_model_dir": "models/ocr/rec",
            "cls_model_dir": "models/ocr/cls",
            "acceptance_threshold": 0.65,
        },
        "crop_padding_pixels": 4,
        "storage": {
            "database_path": "output/observations.sqlite3",
            "crops_dir": "output/crops",
            "output_dir": "output/results",
        },
    }


def _write_yaml(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content), encoding="utf-8")


@pytest.fixture
def config_file(tmp_path: Path, settings: dict[str, Any]) -> Path:
    path = tmp_path / "configs" / "mvp.yaml"
    _write_yaml(path, settings)
    return path


def test_loads_values_and_resolves_every_resource_path(
    tmp_path: Path, config_file: Path
) -> None:
    before = set(tmp_path.rglob("*"))

    config = load_config(config_file, project_root=tmp_path)

    assert isinstance(config, AppConfig)
    assert config.detector.config_path == tmp_path / "detector/eval.yml"
    assert config.detector.checkpoint_path == tmp_path / "models/checkpoint.pth"
    assert config.detector.device == "cpu"
    assert config.detector.score_threshold == 0.75
    assert config.ocr.engine == "paddleocr"
    assert config.ocr.language == "en"
    assert config.ocr.model_version == "PP-OCRv4"
    assert config.ocr.use_gpu is False
    assert config.ocr.cpu_threads == 3
    assert config.ocr.use_angle_cls is True
    assert config.ocr.enable_mkldnn is True
    assert config.ocr.acceptance_threshold == 0.65
    assert config.ocr.det_model_dir == tmp_path / "models/ocr/det"
    assert config.ocr.rec_model_dir == tmp_path / "models/ocr/rec"
    assert config.ocr.cls_model_dir == tmp_path / "models/ocr/cls"
    assert config.crop_padding_pixels == 4
    assert config.storage.database_path == tmp_path / "output/observations.sqlite3"
    assert config.storage.crops_dir == tmp_path / "output/crops"
    assert config.storage.output_dir == tmp_path / "output/results"
    # Loading succeeds without any input models or output locations on disk.
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "detector").exists()
    assert not (tmp_path / "output").exists()
    assert set(tmp_path.rglob("*")) == before


def test_repository_config_loads_from_another_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.detector.checkpoint_path == (
        root / "runs/baseline-001/checkpoints/checkpoint0019.pth"
    )
    assert config.detector.score_threshold == 0.6
    assert config.ocr.use_angle_cls is False
    assert config.storage.database_path == root / "artifacts/mvp/observations.sqlite3"


def test_root_override_controls_relative_yaml_path(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(config_file.parent)

    config = load_config(project_root=str(tmp_path))

    assert config.detector.checkpoint_path == tmp_path / "models/checkpoint.pth"


def test_external_yaml_uses_project_root_for_resource_paths(
    tmp_path: Path, settings: dict[str, Any]
) -> None:
    path = tmp_path / "external" / "settings.yaml"
    root = tmp_path / "project"
    _write_yaml(path, settings)

    config = load_config(str(path), project_root=root)

    assert config.detector.config_path == root / "detector/eval.yml"
    assert config.ocr.rec_model_dir == root / "models/ocr/rec"
    assert not root.exists()


def test_absolute_resource_path_is_not_rebased(
    tmp_path: Path, settings: dict[str, Any], config_file: Path
) -> None:
    absolute = tmp_path / "external" / "weights.pth"
    settings["detector"]["checkpoint_path"] = str(absolute)
    _write_yaml(config_file, settings)

    assert load_config(config_file, project_root=tmp_path).detector.checkpoint_path == absolute


@pytest.mark.parametrize("section", [None, "detector", "ocr", "storage"])
def test_settings_are_frozen(
    section: str | None, tmp_path: Path, config_file: Path
) -> None:
    config = load_config(config_file, project_root=tmp_path)
    target = config if section is None else getattr(config, section)
    field = {
        None: "crop_padding_pixels",
        "detector": "score_threshold",
        "ocr": "cpu_threads",
        "storage": "database_path",
    }[section]

    with pytest.raises(FrozenInstanceError):
        setattr(target, field, None)


@pytest.mark.parametrize("section", [None, "detector", "ocr", "storage"])
@pytest.mark.parametrize("problem", ["missing", "unknown", "nonstring-key"])
def test_invalid_fields_have_context(
    section: str | None,
    problem: str,
    tmp_path: Path,
    settings: dict[str, Any],
    config_file: Path,
) -> None:
    target = settings if section is None else settings[section]
    label = "configuration" if section is None else section
    if problem == "missing":
        missing = next(iter(target))
        del target[missing]
        expected = f"{label}: missing fields: {missing}"
    elif problem == "unknown":
        target["typo_field"] = 1
        expected = f"{label}: unknown fields: typo_field"
    else:
        target[123] = "not a field name"
        expected = f"{label} keys must be strings"
    _write_yaml(config_file, settings)

    with pytest.raises(ConfigError) as error:
        load_config(config_file, project_root=tmp_path)
    assert expected in str(error.value)


@pytest.mark.parametrize("content", [None, [], "text", 42])
def test_root_must_be_a_mapping(
    content: Any, tmp_path: Path, config_file: Path
) -> None:
    _write_yaml(config_file, content)
    with pytest.raises(ConfigError, match="configuration must be a mapping"):
        load_config(config_file, project_root=tmp_path)


@pytest.mark.parametrize("section", ["detector", "ocr", "storage"])
def test_sections_must_be_mappings(
    section: str, tmp_path: Path, settings: dict[str, Any], config_file: Path
) -> None:
    settings[section] = []
    _write_yaml(config_file, settings)
    with pytest.raises(ConfigError, match=f"{section} must be a mapping"):
        load_config(config_file, project_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "field"),
    [("detector", "score_threshold"), ("ocr", "acceptance_threshold")],
)
@pytest.mark.parametrize(
    "value", [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), "0.6", True, None]
)
def test_invalid_thresholds(
    section: str,
    field: str,
    value: Any,
    tmp_path: Path,
    settings: dict[str, Any],
    config_file: Path,
) -> None:
    settings[section][field] = value
    _write_yaml(config_file, settings)
    with pytest.raises(ConfigError) as error:
        load_config(config_file, project_root=tmp_path)
    assert f"{section}.{field}" in str(error.value)


@pytest.mark.parametrize("value", [0, 1, 0.5])
def test_threshold_boundaries_are_accepted(
    value: float,
    tmp_path: Path,
    settings: dict[str, Any],
    config_file: Path,
) -> None:
    settings["detector"]["score_threshold"] = value
    settings["ocr"]["acceptance_threshold"] = value
    _write_yaml(config_file, settings)

    config = load_config(config_file, project_root=tmp_path)

    assert config.detector.score_threshold == value
    assert config.ocr.acceptance_threshold == value
    assert type(config.detector.score_threshold) is float
    assert type(config.ocr.acceptance_threshold) is float


@pytest.mark.parametrize("field", ["use_gpu", "use_angle_cls", "enable_mkldnn"])
@pytest.mark.parametrize("value", ["false", 0, None])
def test_booleans_are_not_silently_coerced(
    field: str,
    value: Any,
    tmp_path: Path,
    settings: dict[str, Any],
    config_file: Path,
) -> None:
    settings["ocr"][field] = value
    _write_yaml(config_file, settings)
    with pytest.raises(ConfigError) as error:
        load_config(config_file, project_root=tmp_path)
    assert f"ocr.{field} must be a boolean" in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu_threads", 0),
        ("cpu_threads", -1),
        ("cpu_threads", 1.5),
        ("cpu_threads", True),
        ("cpu_threads", "2"),
        ("crop_padding_pixels", -1),
        ("crop_padding_pixels", 0.5),
        ("crop_padding_pixels", False),
        ("crop_padding_pixels", "0"),
    ],
)
def test_invalid_integer_settings(
    field: str,
    value: Any,
    tmp_path: Path,
    settings: dict[str, Any],
    config_file: Path,
) -> None:
    target = settings["ocr"] if field == "cpu_threads" else settings
    target[field] = value
    _write_yaml(config_file, settings)
    with pytest.raises(ConfigError) as error:
        load_config(config_file, project_root=tmp_path)
    assert field in str(error.value)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("detector", "config_path", ""),
        ("detector", "checkpoint_path", "  "),
        ("detector", "checkpoint_path", "bad\x00path"),
        ("ocr", "rec_model_dir", None),
        ("storage", "database_path", 123),
        ("detector", "device", ""),
        ("ocr", "language", None),
        ("ocr", "model_version", 4),
        ("ocr", "engine", "otherocr"),
    ],
)
def test_invalid_paths_and_text_values(
    section: str,
    field: str,
    value: Any,
    tmp_path: Path,
    settings: dict[str, Any],
    config_file: Path,
) -> None:
    settings[section][field] = value
    _write_yaml(config_file, settings)
    with pytest.raises(ConfigError) as error:
        load_config(config_file, project_root=tmp_path)
    assert f"{section}.{field}" in str(error.value)


@pytest.mark.parametrize(
    "content", ["", "detector: [", "!!python/object:builtins.object {}"]
)
def test_empty_malformed_and_unsafe_yaml_raise_config_error(
    content: str, tmp_path: Path
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, project_root=tmp_path)


def test_invalid_encoding_preserves_underlying_cause(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigError) as error:
        load_config(path, project_root=tmp_path)
    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, UnicodeError)


@pytest.mark.parametrize("name", ["missing.yaml", "."])
def test_unreadable_configuration_has_path_and_cause(
    name: str, tmp_path: Path
) -> None:
    path = tmp_path / name
    with pytest.raises(ConfigError) as error:
        load_config(path, project_root=tmp_path)
    assert str(path.resolve()) in str(error.value)
    assert isinstance(error.value.__cause__, OSError)

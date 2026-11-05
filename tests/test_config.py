"""Configuration checks using temporary files, without models or downloads."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from parking_monitoring.config import ConfigError, load_config


@pytest.fixture
def config_tree(tmp_path):
    """Create small input placeholders; tests never deserialize model weights."""
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/mvp.yaml").read_text()
    )
    for field in ("config_path", "checkpoint_path"):
        path = tmp_path / raw["detector"][field]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for field in ("det_model_dir", "rec_model_dir", "cls_model_dir"):
        path = tmp_path / raw["ocr"][field]
        path.mkdir(parents=True, exist_ok=True)
        for name in ("inference.pdmodel", "inference.pdiparams"):
            (path / name).touch()
    path = tmp_path / "configs/mvp.yaml"
    path.write_text(yaml.safe_dump(raw))
    return tmp_path, path, raw


def test_resolves_against_root_without_creating_outputs(
    config_tree, monkeypatch, tmp_path
):
    root, path, raw = config_tree
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(project_root=root)
    assert config.detector.checkpoint_path == root / raw["detector"]["checkpoint_path"]
    assert config.ocr.acceptance_threshold == 0.5
    assert config.crop_padding_pixels == 0
    assert not config.storage.output_dir.exists()
    assert not config.storage.crops_dir.exists()
    assert not config.storage.database_path.exists()
    with pytest.raises(FrozenInstanceError):
        config.crop_padding_pixels = 1
    # Both an absolute YAML path and absolute resource paths are supported.
    raw["detector"]["checkpoint_path"] = str(config.detector.checkpoint_path)
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path, project_root=root) == config


@pytest.mark.parametrize(
    "section,key,value,error",
    [
        ("detector", "score_threshold", 1.5, "detector.score_threshold"),
        ("detector", "score_threshold", True, "detector.score_threshold"),
        ("detector", "score_threshold", float("nan"), "detector.score_threshold"),
        ("detector", "score_threshold", float("inf"), "detector.score_threshold"),
        ("detector", "device", "cuda", "detector.device"),
        ("ocr", "acceptance_threshold", -0.1, "ocr.acceptance_threshold"),
        ("ocr", "cpu_threads", 0, "ocr.cpu_threads"),
        ("ocr", "cpu_threads", True, "ocr.cpu_threads"),
        ("ocr", "cpu_threads", 1.5, "ocr.cpu_threads"),
        ("ocr", "use_gpu", True, "ocr.use_gpu"),
        ("ocr", "use_angle_cls", "false", "ocr.use_angle_cls"),
        ("ocr", "language", "my", "ocr.language"),
        ("ocr", "rec_model_dir", "", "ocr.rec_model_dir"),
    ],
)
def test_invalid_settings(config_tree, section, key, value, error):
    root, path, raw = config_tree
    raw[section][key] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match=error):
        load_config(path, project_root=root)


@pytest.mark.parametrize("padding", [-1, True, 0.5])
def test_invalid_padding(config_tree, padding):
    root, path, raw = config_tree
    raw["crop_padding_pixels"] = padding
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="crop_padding_pixels"):
        load_config(path, project_root=root)


@pytest.mark.parametrize(
    "contents", ["", "[]", "detector: [", "!!python/object:builtins.object {}"]
)
def test_invalid_yaml(config_tree, contents):
    root, path, _ = config_tree
    path.write_text(contents)
    with pytest.raises(ConfigError):
        load_config(path, project_root=root)


@pytest.mark.parametrize("case", ["missing", "unknown_nested", "unknown_root"])
def test_missing_and_unknown_keys(config_tree, case):
    root, path, raw = config_tree
    modified = deepcopy(raw)
    if case == "missing":
        del modified["ocr"]["language"]
    elif case == "unknown_nested":
        modified["detector"]["score_threshhold"] = 0.5
    else:
        modified["extra"] = {}
    path.write_text(yaml.safe_dump(modified))
    with pytest.raises(ConfigError, match="missing keys|unknown keys"):
        load_config(path, project_root=root)


def test_missing_config(config_tree):
    root, _, _ = config_tree
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config("missing.yaml", project_root=root)


@pytest.mark.parametrize("target", ["checkpoint", "ocr_weights"])
def test_missing_model_file(config_tree, target):
    root, path, raw = config_tree
    missing = (
        root / raw["detector"]["checkpoint_path"]
        if target == "checkpoint"
        else root / raw["ocr"]["rec_model_dir"] / "inference.pdiparams"
    )
    missing.unlink()
    with pytest.raises(ConfigError, match="required file not found") as error:
        load_config(path, project_root=root)
    assert str(missing) in str(error.value)


def test_output_path_collisions(config_tree):
    root, path, raw = config_tree
    collision = root / "occupied"
    collision.touch()
    raw["storage"]["output_dir"] = str(collision / "nested")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="parent is not a directory"):
        load_config(path, project_root=root)

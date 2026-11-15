"""Detector wrapper unit tests; no vendored model or real checkpoint is loaded."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from parking_monitoring import detector as detector_module
from parking_monitoring.config import DetectorConfig
from parking_monitoring.detector import RTDETRDetector
from parking_monitoring.schemas import Detection


class TinyModel(torch.nn.Module):
    """Use real strict state-dictionary loading with just one parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(-1.0))
        self.calls: list[dict] = []

    def forward(self, samples: torch.Tensor) -> dict:
        self.calls.append(
            {
                "samples": samples.clone(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "training": self.training,
            }
        )
        return {"synthetic_model_output": self.scale}


class ControlledPostprocessor(torch.nn.Module):
    """A controlled coordinate-conversion boundary, not RT-DETR integration."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.override = None

    def forward(self, outputs: dict, original_sizes: torch.Tensor) -> list[dict]:
        self.calls.append(
            {
                "outputs": outputs,
                "sizes": original_sizes.clone(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "training": self.training,
            }
        )
        if self.override is not None:
            return self.override
        # The wrapper must supply dimensions in the right order and must not
        # apply another resize to the original-pixel coordinates returned here.
        boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
        boxes *= original_sizes[0].repeat(2)
        return [{"boxes": boxes, "scores": torch.tensor([0.75]), "labels": torch.tensor([0])}]


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    architecture = tmp_path / "model.yml"
    checkpoint = tmp_path / "synthetic.pth"
    architecture.write_text("synthetic architecture", encoding="utf-8")
    checkpoint.write_bytes(b"not a real checkpoint")
    state = SimpleNamespace(
        config=DetectorConfig(architecture, checkpoint, "cpu", 0.5),
        model=TinyModel(),
        postprocessor=ControlledPostprocessor(),
        parsed=[],
        yaml={
            "num_classes": 1,
            "remap_mscoco_category": False,
            "PResNet": {"pretrained": True},
            "eval_spatial_size": [8, 12],
            "val_dataloader": {
                "dataset": {
                    "transforms": {
                        "type": "Compose",
                        "ops": [
                            {"type": "Resize", "size": [8, 12]},
                            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        ],
                    }
                }
            },
        },
    )

    class ControlledConfig:
        def __init__(self, path: str) -> None:
            self.path = path
            self.yaml_cfg = deepcopy(state.yaml)
            state.parsed.append(self)

        @property
        def model(self) -> TinyModel:
            # Fail if a separate backbone download could be requested.
            assert self.yaml_cfg["PResNet"]["pretrained"] is False
            return state.model

        @property
        def postprocessor(self) -> ControlledPostprocessor:
            return state.postprocessor

        @property
        def val_dataloader(self):
            pytest.fail("Unit inference must not construct an annotation-based loader")

    state.preprocess = Mock(return_value=torch.full((3, 8, 12), 0.25))
    state.compose = Mock(return_value=state.preprocess)
    state.backend = Mock(return_value=(ControlledConfig, state.compose))
    state.load = Mock(
        return_value={
            "ema": {"module": {"scale": torch.tensor(3.0)}},
            "model": {"scale": torch.tensor(99.0)},
        }
    )
    monkeypatch.setattr(detector_module, "_load_backend", state.backend)
    monkeypatch.setattr(detector_module.torch, "load", state.load)
    return state


def test_initializes_from_ema_and_configured_transforms(harness: SimpleNamespace) -> None:
    RTDETRDetector(harness.config)

    assert harness.model.scale.item() == 3.0  # Not raw model weights (99).
    assert harness.model.training is False
    assert harness.postprocessor.training is False
    harness.load.assert_called_once_with(
        harness.config.checkpoint_path, map_location="cpu", weights_only=False
    )
    assert harness.parsed[0].path == str(harness.config.config_path)
    harness.compose.assert_called_once_with(
        ops=harness.yaml["val_dataloader"]["dataset"]["transforms"]["ops"]
    )
    assert harness.yaml["PResNet"]["pretrained"] is True


@pytest.mark.parametrize("value", [True, "0.5", None, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_fails_before_loading(harness: SimpleNamespace, value) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        RTDETRDetector(replace(harness.config, score_threshold=value))
    harness.backend.assert_not_called()
    harness.load.assert_not_called()


@pytest.mark.parametrize("device", [None, "", "not-a-torch-device"])
def test_invalid_device_is_a_settings_error(harness: SimpleNamespace, device) -> None:
    with pytest.raises(ValueError, match="detector.device"):
        RTDETRDetector(replace(harness.config, device=device))
    harness.load.assert_not_called()


def test_wrong_settings_object_is_rejected(harness: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="DetectorConfig"):
        RTDETRDetector({})
    harness.backend.assert_not_called()


@pytest.mark.parametrize("field", ["config_path", "checkpoint_path"])
def test_missing_files_raise_file_not_found(harness: SimpleNamespace, field: str) -> None:
    path = getattr(harness.config, field)
    path.unlink()
    with pytest.raises(FileNotFoundError) as error:
        RTDETRDetector(harness.config)
    assert str(path) in str(error.value)
    harness.backend.assert_not_called()


@pytest.mark.parametrize("field", ["config_path", "checkpoint_path"])
def test_paths_must_be_path_objects(harness: SimpleNamespace, field: str) -> None:
    config = replace(harness.config, **{field: str(getattr(harness.config, field))})
    with pytest.raises(ValueError, match="pathlib.Path"):
        RTDETRDetector(config)


@pytest.mark.parametrize("checkpoint", [None, {}, {"ema": {}}, {"ema": {"module": None}}])
def test_checkpoint_requires_ema(harness: SimpleNamespace, checkpoint) -> None:
    harness.load.return_value = checkpoint
    with pytest.raises(RuntimeError, match="Checkpoint has no EMA module") as error:
        RTDETRDetector(harness.config)
    assert isinstance(error.value.__cause__, ValueError)


def test_ema_parameters_must_match_strictly(harness: SimpleNamespace) -> None:
    harness.load.return_value = {
        "ema": {"module": {"scale": torch.tensor(3.0), "unexpected": torch.tensor(0.0)}}
    }
    with pytest.raises(RuntimeError, match="Unexpected key") as error:
        RTDETRDetector(harness.config)
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize("field,value", [("num_classes", 2), ("remap_mscoco_category", True)])
def test_non_plate_configuration_is_rejected(
    harness: SimpleNamespace, field: str, value
) -> None:
    harness.yaml[field] = value
    with pytest.raises(RuntimeError, match="one class without COCO remapping"):
        RTDETRDetector(harness.config)
    harness.load.assert_not_called()


@pytest.mark.parametrize("size", [None, [8], [0, 12], [8, "12"], [False, 12]])
def test_invalid_model_input_size_is_reported(harness: SimpleNamespace, size) -> None:
    harness.yaml["eval_spatial_size"] = size
    with pytest.raises(RuntimeError, match="eval_spatial_size"):
        RTDETRDetector(harness.config)


def test_unsupported_transform_container_is_reported(harness: SimpleNamespace) -> None:
    harness.yaml["val_dataloader"]["dataset"]["transforms"]["type"] = "Other"
    with pytest.raises(RuntimeError, match="Validation transforms must use Compose"):
        RTDETRDetector(harness.config)


@pytest.mark.parametrize("stage", ["backend", "load", "device"])
def test_initialization_failure_preserves_context_and_cause(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    failure = RuntimeError("synthetic initialization failure")
    if stage == "backend":
        harness.backend.side_effect = failure
    elif stage == "load":
        harness.load.side_effect = failure
    else:
        monkeypatch.setattr(harness.model, "to", Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as error:
        RTDETRDetector(harness.config)
    assert str(harness.config.config_path) in str(error.value)
    assert str(harness.config.checkpoint_path) in str(error.value)
    assert "cpu" in str(error.value)
    assert error.value.__cause__ is failure


def test_missing_included_file_keeps_file_not_found(harness: SimpleNamespace) -> None:
    harness.backend.side_effect = FileNotFoundError("missing included model YAML")
    with pytest.raises(FileNotFoundError, match="included model YAML"):
        RTDETRDetector(harness.config)


@pytest.mark.parametrize(
    "size,expected",
    [
        ((1280, 720), (320.0, 180.0, 960.0, 540.0)),
        ((720, 1280), (180.0, 320.0, 540.0, 960.0)),
    ],
)
def test_original_sizes_and_returned_coordinate_mapping(
    harness: SimpleNamespace, size: tuple[int, int], expected: tuple[float, ...]
) -> None:
    detector = RTDETRDetector(harness.config)
    with Image.new("RGB", size) as image:
        result = detector.predict(image)

    assert result == [Detection(expected, 0.75)]
    assert all(type(value) is float for value in result[0].bbox_xyxy)
    assert type(result[0].score) is float
    model_call = harness.model.calls[0]
    assert model_call["samples"].shape == (1, 3, 8, 12)
    assert model_call["samples"].dtype == torch.float32
    assert model_call["samples"].device.type == "cpu"
    torch.testing.assert_close(model_call["samples"], torch.full((1, 3, 8, 12), 0.25))
    assert model_call["inference_mode"] is True
    assert model_call["training"] is False
    post_call = harness.postprocessor.calls[0]
    assert post_call["outputs"]["synthetic_model_output"] is harness.model.scale
    assert post_call["sizes"].tolist() == [list(size)]
    assert post_call["inference_mode"] is True
    assert post_call["training"] is False


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_prediction_preserves_source_and_preprocesses_an_rgb_copy(
    harness: SimpleNamespace, mode: str
) -> None:
    detector = RTDETRDetector(harness.config)
    observed = []

    def preprocess(image: Image.Image) -> torch.Tensor:
        observed.append((image.mode, image.size, image))
        # A mutating third-party transform still must not alter the source.
        image.putpixel((0, 0), (255, 0, 0))
        return torch.zeros(3, 8, 12)

    harness.preprocess.side_effect = preprocess
    with Image.new(mode, (40, 20)) as image:
        before = image.tobytes()
        detector.predict(image)
        assert image.mode == mode
        assert image.size == (40, 20)
        assert image.tobytes() == before
        assert observed[0][:2] == ("RGB", (40, 20))
        assert observed[0][2] is not image


def test_reuses_initialized_resources_for_multiple_images(harness: SimpleNamespace) -> None:
    detector = RTDETRDetector(harness.config)
    for size in [(80, 40), (40, 80)]:
        with Image.new("RGB", size) as image:
            detector.predict(image)
    harness.load.assert_called_once()
    harness.backend.assert_called_once()
    harness.compose.assert_called_once()
    assert harness.preprocess.call_count == 2
    assert len(harness.model.calls) == 2
    assert [call["sizes"].tolist() for call in harness.postprocessor.calls] == [
        [[80, 40]], [[40, 80]]
    ]


@pytest.mark.parametrize("image", [None, "photo.jpg"])
def test_image_paths_and_non_images_are_invalid(harness: SimpleNamespace, image) -> None:
    detector = RTDETRDetector(harness.config)
    with pytest.raises(ValueError, match="nonempty Pillow image"):
        detector.predict(image)
    harness.preprocess.assert_not_called()


@pytest.mark.parametrize("size", [(0, 1), (1, 0)])
def test_empty_image_is_rejected(harness: SimpleNamespace, size: tuple[int, int]) -> None:
    detector = RTDETRDetector(harness.config)
    with (
        Image.new("RGB", size) as image,
        pytest.raises(ValueError, match="nonempty Pillow image"),
    ):
        detector.predict(image)


def test_closed_image_is_an_input_error(harness: SimpleNamespace) -> None:
    detector = RTDETRDetector(harness.config)
    image = Image.new("RGB", (10, 10))
    image.close()
    with pytest.raises(ValueError, match="Cannot read image pixels") as error:
        detector.predict(image)
    assert isinstance(error.value.__cause__, ValueError)
    assert not harness.model.calls


@pytest.mark.parametrize(
    "tensor",
    [None, torch.zeros(3, 8, 12, dtype=torch.uint8), torch.zeros(8, 12, 3), torch.zeros(3, 7, 12)],
)
def test_bad_preprocessing_output_stops_before_inference(
    harness: SimpleNamespace, tensor
) -> None:
    detector = RTDETRDetector(harness.config)
    harness.preprocess.return_value = tensor
    with (
        Image.new("RGB", (20, 10)) as image,
        pytest.raises(RuntimeError, match="Preprocessing must return"),
    ):
        detector.predict(image)
    assert not harness.model.calls


def _prediction() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor([[1.0, 2.0, 8.0, 9.0]]),
        "scores": torch.tensor([0.75]),
        "labels": torch.tensor([0]),
    }


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_threshold_is_strictly_greater(harness: SimpleNamespace, threshold: float) -> None:
    detector = RTDETRDetector(replace(harness.config, score_threshold=threshold))
    harness.postprocessor.override = [{
        "boxes": torch.tensor([[1.0, 2.0, 8.0, 9.0]]).repeat(3, 1),
        "scores": torch.tensor([0.0, 0.5, 1.0]),
        "labels": torch.zeros(3, dtype=torch.int64),
    }]
    with Image.new("RGB", (20, 10)) as image:
        result = detector.predict(image)
    expected_scores = {0.0: [0.5, 1.0], 0.5: [1.0], 1.0: []}[threshold]
    assert [detection.score for detection in result] == expected_scores


def test_empty_predictions_are_a_successful_empty_list(harness: SimpleNamespace) -> None:
    detector = RTDETRDetector(harness.config)
    harness.postprocessor.override = [{
        "boxes": torch.empty(0, 4),
        "scores": torch.empty(0),
        "labels": torch.empty(0, dtype=torch.int64),
    }]
    with Image.new("RGB", (20, 10)) as image:
        assert detector.predict(image) == []


def test_original_pixel_boxes_are_not_rescaled_rounded_or_clamped(
    harness: SimpleNamespace,
) -> None:
    detector = RTDETRDetector(harness.config)
    expected = (-1.25, 2.5, 25.75, 14.0)
    result = _prediction()
    result["boxes"] = torch.tensor([expected])
    harness.postprocessor.override = [result]
    with Image.new("RGB", (20, 10)) as image:
        assert detector.predict(image)[0].bbox_xyxy == expected


@pytest.mark.parametrize(
    "problem",
    [
        "no-image-result", "two-image-results", "missing-boxes", "box-shape",
        "score-count", "label-count", "nan-box", "infinite-score",
        "negative-score", "score-above-one", "foreign-label", "reversed-box", "zero-area",
    ],
)
def test_malformed_predictions_raise_a_technical_error(
    harness: SimpleNamespace, problem: str
) -> None:
    detector = RTDETRDetector(harness.config)
    result = _prediction()
    replacements = {
        "box-shape": ("boxes", torch.zeros(1, 3)),
        "score-count": ("scores", torch.empty(0)),
        "label-count": ("labels", torch.empty(0, dtype=torch.int64)),
        "nan-box": ("boxes", torch.tensor([[float("nan"), 0.0, 1.0, 1.0]])),
        "infinite-score": ("scores", torch.tensor([float("inf")])),
        "negative-score": ("scores", torch.tensor([-0.1])),
        "score-above-one": ("scores", torch.tensor([1.1])),
        "foreign-label": ("labels", torch.tensor([1])),
        "reversed-box": ("boxes", torch.tensor([[8.0, 2.0, 1.0, 9.0]])),
        "zero-area": ("boxes", torch.tensor([[1.0, 2.0, 1.0, 9.0]])),
    }
    if problem in replacements:
        field, value = replacements[problem]
        result[field] = value
    if problem == "missing-boxes":
        del result["boxes"]
    harness.postprocessor.override = (
        [] if problem == "no-image-result"
        else [result, result] if problem == "two-image-results"
        else [result]
    )
    with (
        Image.new("RGB", (20, 10)) as image,
        pytest.raises(RuntimeError, match="inference failed for a 20x10 image") as error,
    ):
        detector.predict(image)
    assert isinstance(error.value.__cause__, (ValueError, KeyError))


@pytest.mark.parametrize("stage", ["preprocess", "model", "postprocessor"])
def test_inference_failures_preserve_the_original_cause(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    detector = RTDETRDetector(harness.config)
    failure = RuntimeError("synthetic inference failure")
    if stage == "preprocess":
        harness.preprocess.side_effect = failure
    else:
        module = harness.model if stage == "model" else harness.postprocessor
        monkeypatch.setattr(module, "forward", Mock(side_effect=failure))
    with Image.new("RGB", (20, 10)) as image, pytest.raises(RuntimeError) as error:
        detector.predict(image)
    assert error.value.__cause__ is failure

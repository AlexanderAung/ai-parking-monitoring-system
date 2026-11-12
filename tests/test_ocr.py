"""OCR wrapper contracts with synthetic files/backend; no real models are loaded."""

import json
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
from PIL import Image

from parking_monitoring import ocr as ocr_module
from parking_monitoring.config import OCRConfig
from parking_monitoring.ocr import PaddleOCREngine
from parking_monitoring.schemas import OCRResult


def region(
    text: str = "ab 12 cde",
    score: object = 0.9,
    *,
    left: float = 0,
    top: float = 0,
    width: float = 20,
    height: float = 8,
) -> list:
    """Supply a backend-shaped reading with explicit test geometry."""
    return [
        [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
        ],
        (text, score),
    ]


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Keep all model/dependency access controlled at the wrapper boundary."""
    model_dirs = {}
    for name in ("det", "rec", "cls"):
        directory = tmp_path / "local models" / name
        directory.mkdir(parents=True)
        for filename in ("inference.pdmodel", "inference.pdiparams"):
            (directory / filename).write_bytes(
                b"synthetic unit-test model, not loadable"
            )
        model_dirs[name] = directory
    config = OCRConfig(
        engine="paddleocr",
        language="en",
        model_version="PP-OCRv4",
        use_gpu=False,
        cpu_threads=3,
        use_angle_cls=True,
        enable_mkldnn=True,
        det_model_dir=model_dirs["det"],
        rec_model_dir=model_dirs["rec"],
        cls_model_dir=model_dirs["cls"],
        acceptance_threshold=0.5,
    )
    backend = SimpleNamespace(ocr=Mock(return_value=[[region()]]))
    factory = Mock(return_value=backend)
    paddle = SimpleNamespace(
        is_compiled_with_cuda=Mock(return_value=True),
        device=SimpleNamespace(cuda=SimpleNamespace(device_count=Mock(return_value=1))),
    )

    def import_backend(name: str) -> SimpleNamespace:
        if name == "paddleocr":
            return SimpleNamespace(PaddleOCR=factory)
        if name == "paddle":
            return paddle
        pytest.fail(f"Unexpected backend import: {name}")

    imports = Mock(side_effect=import_backend)
    metadata = Mock(return_value="2.10.0")
    # Replace this module's reference, not the shared global importlib function.
    monkeypatch.setattr(ocr_module, "importlib", SimpleNamespace(import_module=imports))
    monkeypatch.setattr(ocr_module, "version", metadata)
    return SimpleNamespace(
        config=config,
        backend=backend,
        factory=factory,
        imports=imports,
        metadata=metadata,
        paddle=paddle,
        root=tmp_path,
    )


@pytest.fixture
def crop():
    with Image.new("RGB", (80, 48), (7, 33, 99)) as image:
        yield image


def test_constructor_passes_settings_and_preserves_all_local_files(harness) -> None:
    before = {
        path: path.read_bytes() for path in harness.root.rglob("*") if path.is_file()
    }
    PaddleOCREngine(harness.config)

    harness.metadata.assert_called_once_with("paddleocr")
    harness.imports.assert_called_once_with("paddleocr")
    harness.factory.assert_called_once_with(
        lang="en",
        ocr_version="PP-OCRv4",
        use_gpu=False,
        cpu_threads=3,
        use_angle_cls=True,
        enable_mkldnn=True,
        det_model_dir=str(harness.config.det_model_dir),
        rec_model_dir=str(harness.config.rec_model_dir),
        cls_model_dir=str(harness.config.cls_model_dir),
        drop_score=0.0,
        use_onnx=False,
        save_crop_res=False,
        show_log=False,
    )
    harness.backend.ocr.assert_not_called()
    assert {
        path: path.read_bytes() for path in harness.root.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize("value", [None, {}, "configs/mvp.yaml"])
def test_wrong_config_type_never_loads_backend(harness, value) -> None:
    with pytest.raises(ValueError, match="OCRConfig"):
        PaddleOCREngine(value)
    harness.imports.assert_not_called()
    harness.factory.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("engine", "other", "engine"),
        ("engine", None, "engine"),
        ("language", "latin", "language"),
        ("language", None, "language"),
        ("model_version", "PP-OCRv3", "model_version"),
        ("model_version", "", "model_version"),
        *[
            (field, value, field)
            for field in ("use_gpu", "use_angle_cls", "enable_mkldnn")
            for value in (0, "false", None)
        ],
        *[
            ("cpu_threads", value, "cpu_threads")
            for value in (0, -1, 1.0, True, "1", None)
        ],
        *[
            ("acceptance_threshold", value, "acceptance_threshold")
            for value in (
                -0.1,
                1.1,
                float("nan"),
                float("inf"),
                -float("inf"),
                True,
                "0.5",
                None,
            )
        ],
    ],
)
def test_invalid_settings_fail_before_dependency_loading(
    harness, field, value, message
) -> None:
    with pytest.raises(ValueError, match=message):
        PaddleOCREngine(replace(harness.config, **{field: value}))
    harness.metadata.assert_not_called()
    harness.imports.assert_not_called()
    harness.factory.assert_not_called()


@pytest.mark.parametrize("name", ["det", "rec", "cls"])
def test_model_directories_must_be_paths(harness, name) -> None:
    with pytest.raises(ValueError, match=f"{name}_model_dir"):
        PaddleOCREngine(
            replace(
                harness.config, **{f"{name}_model_dir": "https://example.invalid/model"}
            )
        )
    harness.imports.assert_not_called()


@pytest.mark.parametrize(
    ("name", "filename"),
    list(product(("det", "rec", "cls"), ("inference.pdmodel", "inference.pdiparams"))),
)
def test_each_required_file_prevents_backend_loading_when_missing(
    harness, name, filename
) -> None:
    path = getattr(harness.config, f"{name}_model_dir") / filename
    path.unlink()
    with pytest.raises(FileNotFoundError, match=filename):
        PaddleOCREngine(replace(harness.config, use_angle_cls=False))
    assert not path.exists()
    harness.metadata.assert_not_called()
    harness.imports.assert_not_called()
    harness.factory.assert_not_called()


def test_directory_in_place_of_model_file_is_missing_resource(harness) -> None:
    path = harness.config.det_model_dir / "inference.pdmodel"
    path.unlink()
    path.mkdir()
    with pytest.raises(FileNotFoundError, match="inference.pdmodel"):
        PaddleOCREngine(harness.config)
    harness.imports.assert_not_called()


def test_empty_model_fails_before_backend_loading(harness) -> None:
    (harness.config.rec_model_dir / "inference.pdiparams").write_bytes(b"")
    with pytest.raises(RuntimeError, match="Empty local OCR rec model file"):
        PaddleOCREngine(harness.config)
    harness.imports.assert_not_called()


def test_unreadable_model_preserves_permission_error_cause(
    harness, monkeypatch
) -> None:
    target = harness.config.det_model_dir / "inference.pdmodel"
    original_open = Path.open
    failure = PermissionError("synthetic read denial")

    def guarded_open(path, *args, **kwargs):
        if path == target:
            raise failure
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(RuntimeError, match="Cannot use local OCR det model") as caught:
        PaddleOCREngine(harness.config)
    assert caught.value.__cause__ is failure
    harness.imports.assert_not_called()


@pytest.mark.parametrize("installed", ["3.0.0", "unknown"])
def test_incompatible_package_api_fails_before_backend_import(
    harness, installed
) -> None:
    harness.metadata.return_value = installed
    with pytest.raises(RuntimeError, match="requires PaddleOCR 2.x"):
        PaddleOCREngine(harness.config)
    harness.imports.assert_not_called()
    harness.factory.assert_not_called()


def test_missing_package_is_an_initialization_error(harness) -> None:
    failure = PackageNotFoundError("paddleocr")
    harness.metadata.side_effect = failure
    with pytest.raises(
        RuntimeError, match="Cannot initialize local PaddleOCR"
    ) as caught:
        PaddleOCREngine(harness.config)
    assert caught.value.__cause__ is failure
    harness.imports.assert_not_called()


def test_backend_import_failure_keeps_its_cause(harness) -> None:
    failure = ImportError("synthetic missing Paddle dependency")
    harness.imports.side_effect = failure
    with pytest.raises(RuntimeError, match="missing Paddle dependency") as caught:
        PaddleOCREngine(harness.config)
    assert caught.value.__cause__ is failure
    harness.factory.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [ValueError("bad model"), OSError("bad file"), RuntimeError("bad operator")],
)
def test_model_initialization_failure_keeps_its_cause(harness, failure) -> None:
    harness.factory.side_effect = failure
    with pytest.raises(
        RuntimeError, match="Cannot initialize local PaddleOCR"
    ) as caught:
        PaddleOCREngine(harness.config)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(("compiled", "devices"), [(False, 1), (True, 0)])
def test_requested_gpu_does_not_silently_fall_back_to_cpu(
    harness, compiled, devices
) -> None:
    harness.paddle.is_compiled_with_cuda.return_value = compiled
    harness.paddle.device.cuda.device_count.return_value = devices
    with pytest.raises(RuntimeError, match="available CUDA"):
        PaddleOCREngine(replace(harness.config, use_gpu=True))
    harness.factory.assert_not_called()


def test_available_gpu_setting_reaches_backend(harness) -> None:
    PaddleOCREngine(replace(harness.config, use_gpu=True))
    assert harness.imports.call_args_list == [call("paddleocr"), call("paddle")]
    assert harness.factory.call_args.kwargs["use_gpu"] is True


def test_reading_is_structured_and_json_serializable(harness, crop) -> None:
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert result == OCRResult("ab 12 cde", "AB12CDE", 0.9, "read")
    assert type(result.score) is float
    assert json.loads(json.dumps(asdict(result), allow_nan=False)) == asdict(result)


@pytest.mark.parametrize("classifier", [False, True])
def test_recognition_reuses_one_backend_and_passes_inference_flags(
    harness, crop, classifier
) -> None:
    engine = PaddleOCREngine(replace(harness.config, use_angle_cls=classifier))
    for _ in range(3):
        assert engine.recognize(crop).status == "read"
    harness.factory.assert_called_once()
    assert harness.backend.ocr.call_count == 3
    for invocation in harness.backend.ocr.call_args_list:
        assert invocation.kwargs == {"det": True, "rec": True, "cls": classifier}


@pytest.mark.parametrize(
    "value", [None, "crop.png", np.zeros((3, 4, 3), dtype=np.uint8)]
)
def test_invalid_crop_type_never_calls_backend(harness, value) -> None:
    engine = PaddleOCREngine(harness.config)
    with pytest.raises(ValueError, match="nonempty Pillow image"):
        engine.recognize(value)
    harness.backend.ocr.assert_not_called()


@pytest.mark.parametrize("size", [(0, 4), (4, 0), (0, 0)])
def test_empty_crop_dimensions_never_call_backend(harness, size) -> None:
    engine = PaddleOCREngine(harness.config)
    with Image.new("RGB", size) as image, pytest.raises(ValueError, match="nonempty"):
        engine.recognize(image)
    harness.backend.ocr.assert_not_called()


def test_closed_crop_is_a_caller_input_error(harness) -> None:
    engine = PaddleOCREngine(harness.config)
    image = Image.new("RGB", (3, 2))
    image.close()
    with pytest.raises(ValueError, match="Cannot read crop pixels"):
        engine.recognize(image)
    harness.backend.ocr.assert_not_called()


def test_pixel_read_failure_retains_its_cause(harness, crop, monkeypatch) -> None:
    engine = PaddleOCREngine(harness.config)
    failure = OSError("synthetic broken image stream")
    monkeypatch.setattr(Image.Image, "convert", Mock(side_effect=failure))
    with pytest.raises(ValueError, match="Cannot read crop pixels") as caught:
        engine.recognize(crop)
    assert caught.value.__cause__ is failure
    harness.backend.ocr.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "color", "expected_bgr"),
    [
        ("RGB", (7, 33, 99), (99, 33, 7)),
        ("RGBA", (7, 33, 99, 120), (99, 33, 7)),
        ("L", 43, (43, 43, 43)),
        ("P", 1, (99, 33, 7)),
    ],
)
def test_backend_gets_independent_bgr_pixels_without_mutating_crop(
    harness, mode, color, expected_bgr
) -> None:
    engine = PaddleOCREngine(harness.config)
    observed = []

    def mutate_backend_image(image, **kwargs):
        assert image.shape == (3, 7, 3)
        assert image.dtype == np.uint8
        assert image.flags.c_contiguous and image.flags.writeable
        np.testing.assert_array_equal(image[0, 0], expected_bgr)
        observed.append(image)
        image[:] = 0
        return [[region()]]

    harness.backend.ocr.side_effect = mutate_backend_image
    with Image.new(mode, (7, 3), color) as image:
        if mode == "P":
            image.putpalette([0, 0, 0, 7, 33, 99] + [0] * (256 * 3 - 6))
        image.info["review_marker"] = "preserve"
        before = (
            image.tobytes(),
            image.mode,
            image.size,
            image.getpalette(),
            dict(image.info),
        )
        for _ in range(2):
            assert engine.recognize(image).status == "read"
        after = (
            image.tobytes(),
            image.mode,
            image.size,
            image.getpalette(),
            dict(image.info),
        )
        assert after == before
    assert observed[0] is not observed[1]


def test_opened_source_file_is_not_rewritten(harness, crop, tmp_path) -> None:
    engine = PaddleOCREngine(harness.config)
    source = tmp_path / "source.png"
    crop.save(source)
    original_bytes = source.read_bytes()
    with Image.open(source) as loaded:
        assert engine.recognize(loaded).status == "read"
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("threshold", "score", "status"),
    [
        (0.0, 0.0, "read"),
        (0.5, 0.0, "uncertain"),
        (0.5, 0.499, "uncertain"),
        (0.5, 0.5, "read"),
        (0.5, 0.501, "read"),
        (1.0, 0.999, "uncertain"),
        (1.0, 1.0, "read"),
    ],
)
def test_threshold_is_inclusive_and_does_not_discard_low_text(
    harness, crop, threshold, score, status
) -> None:
    harness.backend.ocr.return_value = [[region(score=score)]]
    result = PaddleOCREngine(
        replace(harness.config, acceptance_threshold=threshold)
    ).recognize(crop)
    assert result == OCRResult("ab 12 cde", "AB12CDE", score, status)


@pytest.mark.parametrize("score", [np.float32(0.5), np.float64(0.5), np.int64(1)])
def test_numpy_scores_become_native_python_scalars(harness, crop, score) -> None:
    harness.backend.ocr.return_value = [[region(score=score)]]
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert result.status == "read"
    assert type(result.score) is float
    assert result.score == float(score)


@pytest.mark.parametrize(
    ("text", "normalized"),
    [
        ("ab 12 cde", "AB12CDE"),
        (" o0\ti1\nb8- ", "O0I1B8-"),
        ("ab\u00a012\u2003cde", "AB12CDE"),
    ],
)
def test_normalization_preserves_raw_text_and_never_substitutes_characters(
    harness, crop, text, normalized
) -> None:
    harness.backend.ocr.return_value = [[region(text)]]
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert result.raw_text == text
    assert result.normalized_text == normalized


@pytest.mark.parametrize("output", [[], [None], [[]]])
def test_successful_empty_output_is_unreadable(harness, crop, output) -> None:
    harness.backend.ocr.return_value = output
    assert PaddleOCREngine(harness.config).recognize(crop) == OCRResult(
        None, None, None, "unreadable"
    )


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_no_usable_normalized_text_is_unreadable_and_retains_available_raw_data(
    harness, crop, text
) -> None:
    harness.backend.ocr.return_value = [[region(text, 0.8)]]
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert result == OCRResult(text or None, None, 0.8, "unreadable")


@pytest.mark.parametrize("scores", [(None,), (None, 0.9), (0.9, None), (None, None)])
def test_any_missing_region_score_keeps_text_uncertain(harness, crop, scores) -> None:
    harness.backend.ocr.return_value = [
        [region("ab", score, left=index * 25) for index, score in enumerate(scores)]
    ]
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert result.raw_text == " ".join(["ab"] * len(scores))
    assert result.normalized_text == "AB" * len(scores)
    assert result.score is None
    assert result.status == "uncertain"
    assert result.error_message is None


def test_two_line_interview_example_uses_position_and_minimum_not_average(
    harness, crop
) -> None:
    harness.backend.ocr.return_value = [
        [
            region("12 cde", 0.42, top=20),
            region("ab", 0.94, top=0),
        ]
    ]
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert result == OCRResult("ab 12 cde", "AB12CDE", 0.42, "uncertain")


def test_shuffled_rows_are_assembled_top_to_bottom_then_left_to_right(
    harness, crop
) -> None:
    harness.backend.ocr.return_value = [
        [
            region("CDE", 0.6, left=40, top=20),
            region("12", 0.8, left=40, top=0),
            region("XY", 0.7, left=0, top=20),
            region("AB", 0.9, left=0, top=0),
        ]
    ]
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert result == OCRResult("AB 12 XY CDE", "AB12XYCDE", 0.6, "read")


def test_slightly_offset_regions_stay_on_one_row(harness, crop) -> None:
    harness.backend.ocr.return_value = [
        [
            region("12", 0.8, left=30, top=0),
            region("AB", 0.9, left=0, top=2),
        ]
    ]
    assert PaddleOCREngine(harness.config).recognize(crop).raw_text == "AB 12"


def test_numeric_numpy_corners_are_accepted(harness, crop) -> None:
    item = region()
    item[0] = np.asarray(item[0], dtype=np.float32)
    harness.backend.ocr.return_value = [[item]]
    assert PaddleOCREngine(harness.config).recognize(crop).status == "read"


@pytest.mark.parametrize(
    "output",
    [
        None,
        "not a page list",
        {},
        np.zeros((1, 1)),
        [None, None],
        [{}],
        [[None]],
        [[[]]],
        [[["only one field"]]],
        [[[None, ("ab", 0.9)]]],
        [[[[[0, 0], [1, 0], [1, 1]], ("ab", 0.9)]]],
        [[[[[False, False], [True, False], [True, True], [False, True]], ("ab", 0.9)]]],
        [[[[["0", "0"], ["1", "0"], ["1", "1"], ["0", "1"]], ("ab", 0.9)]]],
        [[[[[0, 0], [1, 0], [1, float("nan")], [0, 1]], ("ab", 0.9)]]],
        [[[[[0, 0], [1, 0], [1, float("inf")], [0, 1]], ("ab", 0.9)]]],
        [[region(width=0)]],
        [[region(height=0)]],
        [[[[[-1e308, 0], [1e308, 0], [1e308, 1], [-1e308, 1]], ("ab", 0.9)]]],
        [[[region()[0], None]]],
        [[[region()[0], ("ab",)]]],
        [[[region()[0], (None, 0.9)]]],
        [[[region()[0], "ab"]]],
        *[
            [[region(score=value)]]
            for value in (
                "0.9",
                True,
                np.bool_(False),
                -0.01,
                1.01,
                float("nan"),
                float("inf"),
                [],
                1j,
                10**1000,
            )
        ],
    ],
)
def test_malformed_output_is_error_instead_of_an_unreadable_plate(
    harness, crop, output
) -> None:
    harness.backend.ocr.return_value = output
    result = PaddleOCREngine(harness.config).recognize(crop)
    assert (result.raw_text, result.normalized_text, result.score, result.status) == (
        None,
        None,
        None,
        "error",
    )
    assert result.error_message.startswith("Invalid OCR output:")


def test_failed_crop_does_not_prevent_a_later_read_on_the_same_instance(
    harness, crop
) -> None:
    harness.backend.ocr.side_effect = [
        RuntimeError("synthetic operator failure"),
        [[region()]],
    ]
    engine = PaddleOCREngine(harness.config)
    failed = engine.recognize(crop)
    assert failed.status == "error"
    assert (
        failed.raw_text is None
        and failed.normalized_text is None
        and failed.score is None
    )
    assert "RuntimeError" in failed.error_message
    assert "synthetic operator failure" in failed.error_message
    assert engine.recognize(crop).status == "read"
    harness.factory.assert_called_once()


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_process_interruption_is_not_converted_to_recognition_error(
    harness, crop, failure_type
) -> None:
    engine = PaddleOCREngine(harness.config)
    harness.backend.ocr.side_effect = failure_type("stop")
    with pytest.raises(failure_type):
        engine.recognize(crop)


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_process_interruption_is_not_wrapped_during_initialization(
    harness, failure_type
) -> None:
    harness.factory.side_effect = failure_type("stop")
    with pytest.raises(failure_type):
        PaddleOCREngine(harness.config)

"""Focused geometry, labeling, immutability and saved-image integration checks."""

from copy import deepcopy
from dataclasses import replace

import pytest
from PIL import Image

import parking_monitoring.visualization as module
from parking_monitoring.schemas import Detection, ImageResult, OCRResult, PlateResult
from parking_monitoring.storage import SQLiteStorage


@pytest.fixture
def result():
    return ImageResult(
        source_id="example.png",
        image_size_wh=(320, 180),
        plates=[
            PlateResult(
                Detection((50.2, 60.2, 140.8, 120.7), 0.95),
                (45, 55, 146, 126),
                OCRResult("AB 123", "AB123", 0.9, "read"),
            )
        ],
        timings_ms={"total": 5},
        model_info={"detector_checkpoint": "local.pth"},
    )


@pytest.fixture
def image():
    with Image.new("RGB", (320, 180)) as source:
        yield source


@pytest.fixture
def draw_calls(monkeypatch):
    calls = {"rectangles": [], "text": []}
    rectangle = module.ImageDraw.ImageDraw.rectangle
    text = module.ImageDraw.ImageDraw.text

    def record_rectangle(draw, xy, *args, **kwargs):
        calls["rectangles"].append((tuple(xy), kwargs.copy()))
        return rectangle(draw, xy, *args, **kwargs)

    def record_text(draw, xy, value, *args, **kwargs):
        calls["text"].append((value, kwargs.copy()))
        return text(draw, xy, value, *args, **kwargs)

    monkeypatch.setattr(module.ImageDraw.ImageDraw, "rectangle", record_rectangle)
    monkeypatch.setattr(module.ImageDraw.ImageDraw, "text", record_text)
    return calls


def test_source_result_and_padded_crop_are_unchanged(image, result):
    before = image.tobytes()
    original_result = deepcopy(result)
    with module.annotate_image(image, result) as output:
        assert (
            output is not image and output.mode == "RGB" and output.size == image.size
        )
        assert output.tobytes() != before
        assert output.getpixel((50, 100)) == (0, 160, 0)
        assert output.getpixel((45, 100)) == (0, 0, 0)  # No padded-crop outline.
        output.putpixel((0, 0), (255, 255, 255))
    assert image.tobytes() == before and result == original_result


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_zero_plates_returns_independent_rgb_copy(result, mode):
    result = replace(result, plates=[])
    with Image.new(mode, result.image_size_wh) as source:
        before = source.tobytes()
        palette = source.getpalette()
        with module.annotate_image(source, result) as output:
            assert output is not source and output.mode == "RGB"
            assert output.size == source.size
            assert output.getpixel((0, 0)) == (0, 0, 0)
            output.putpixel((0, 0), (255, 255, 255))
        assert source.tobytes() == before and source.getpalette() == palette


@pytest.mark.parametrize(
    ("status", "color"),
    [
        ("read", (0, 160, 0)),
        ("uncertain", (230, 160, 0)),
        ("unreadable", (120, 120, 120)),
        ("error", (220, 0, 0)),
    ],
)
def test_each_status_has_its_agreed_outline_color(image, result, status, color):
    result.plates[0].ocr.status = status
    with module.annotate_image(image, result) as output:
        assert output.getpixel((50, 100)) == color


@pytest.mark.parametrize("status", ["unknown", None])
def test_unknown_status_raises_before_drawing(image, result, status):
    result.plates[0].ocr.status = status
    with pytest.raises(ValueError, match="unknown OCR status"):
        module.annotate_image(image, result)
    assert image.getpixel((50, 100)) == (0, 0, 0)


def test_mismatched_image_size_is_rejected(image, result):
    with pytest.raises(ValueError, match="image_size_wh"):
        module.annotate_image(image, replace(result, image_size_wh=(180, 320)))


@pytest.mark.parametrize(
    "box",
    [
        (float("nan"), 1, 20, 20),
        (0, 1, float("inf"), 20),
        (20, 1, 10, 20),
        (1, 20, 10, 10),
        (1, 1, 1, 20),
        (-20, 1, 0, 20),
        (320, 1, 340, 20),
        (1, 180, 20, 190),
        (True, 1, 20, 20),
        (1, 2, 3),
    ],
)
def test_invalid_detector_boxes_raise(image, result, box):
    result.plates[0].detection.bbox_xyxy = box
    before = deepcopy(result)
    with pytest.raises(ValueError, match="detector box"):
        module.annotate_image(image, result)
    assert result.plates[0].detection.bbox_xyxy is box
    assert image.getpixel((50, 100)) == (0, 0, 0)
    assert result.plates[0].crop_bbox_xyxy == before.plates[0].crop_bbox_xyxy


def test_partial_box_uses_exact_outward_and_inclusive_edges(result, draw_calls):
    result.image_size_wh = (100, 60)
    result.plates[0].detection.bbox_xyxy = (-2.4, 10.2, 100.0, 40.8)
    result.plates[0].crop_bbox_xyxy = (0, 6, 100, 45)
    result.plates[0].ocr = OCRResult(None, None, None, "error", "SECRET backend error")
    before = deepcopy(result)
    with (
        Image.new("RGB", (100, 60)) as source,
        module.annotate_image(source, result) as output,
    ):
        assert draw_calls["rectangles"][0][0] == (0, 10, 99, 40)
        assert output.getpixel((99, 40)) == (220, 0, 0)
        assert output.getpixel((99, 41)) == (0, 0, 0)
    assert result == before
    assert [text for text, _ in draw_calls["text"]] == ["(no text) | error | OCR n/a"]


def test_detector_box_touching_image_edges_uses_last_pixels(image, result, draw_calls):
    result.plates[0].detection.bbox_xyxy = (0, 0, 320, 180)
    with module.annotate_image(image, result) as output:
        assert draw_calls["rectangles"][0][0] == (0, 0, 319, 179)
        assert output.getpixel((319, 100)) == (0, 160, 0)
        assert output.getpixel((100, 179)) == (0, 160, 0)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_ocr_score_raises(image, result, score):
    result.plates[0].ocr.score = score
    with pytest.raises(ValueError, match="OCR score"):
        module.annotate_image(image, result)


@pytest.mark.parametrize(
    ("normalized", "score", "expected"),
    [
        (None, None, "(no text) | error | OCR n/a"),
        ("", 0, "(no text) | error | OCR 0.00"),
        ("AB123", None, "AB123 | error | OCR n/a"),
    ],
)
def test_none_values_and_error_message_label_policy(
    image, result, draw_calls, normalized, score, expected
):
    result.plates[0].ocr = OCRResult(
        "raw fallback forbidden", normalized, score, "error", "SECRET backend error"
    )
    with module.annotate_image(image, result):
        pass
    assert [text for text, _ in draw_calls["text"]] == [expected]


@pytest.mark.parametrize("placement", ["above", "below", "inside"])
def test_labels_stay_inside_canvas_near_edges(result, draw_calls, placement):
    boxes = {
        "above": (350, 50, 400, 100),
        "below": (350, 0, 400, 10),
        "inside": (350, 0, 400, 160),
    }
    result.image_size_wh = (400, 160)
    result.plates[0].detection.bbox_xyxy = boxes[placement]
    with Image.new("RGB", (400, 160)) as source, module.annotate_image(source, result):
        pass
    rectangle = draw_calls["rectangles"][1][0]
    left, top, right, bottom = rectangle
    assert 0 <= left <= right < 400 and 0 <= top <= bottom < 160
    if placement == "above":
        assert bottom < 50
    else:
        assert top == (10 if placement == "below" else 0)


def test_font_and_outline_scale_with_image_dimensions(result, draw_calls):
    result.image_size_wh = (1000, 1000)
    with (
        Image.new("RGB", (1000, 1000)) as source,
        module.annotate_image(source, result),
    ):
        pass
    assert draw_calls["rectangles"][0][1]["width"] == 2
    assert draw_calls["text"][0][1]["font"].size == 30


def test_long_plate_display_is_shortened_without_changing_ocr(result, draw_calls):
    result.image_size_wh = (140, 80)
    result.plates[0].detection.bbox_xyxy = (5, 35, 120, 70)
    result.plates[0].ocr.normalized_text = "A" * 100
    original = deepcopy(result)
    with Image.new("RGB", (140, 80)) as source, module.annotate_image(source, result):
        pass
    label = draw_calls["text"][0][0]
    assert "..." in label and label.endswith(" | read | OCR 0.90")
    assert result == original


def test_tiny_canvas_keeps_box_when_label_cannot_fit(result, draw_calls):
    result.image_size_wh = (1, 1)
    result.plates[0].detection.bbox_xyxy = (0, 0, 1, 1)
    with (
        Image.new("RGB", (1, 1)) as source,
        module.annotate_image(source, result) as output,
    ):
        assert output.getpixel((0, 0)) == (0, 160, 0)
    assert draw_calls["text"] == []


@pytest.mark.parametrize("mode", ["I;16", "I", "F"])
def test_high_depth_modes_are_explicitly_rejected(result, mode):
    with Image.new(mode, result.image_size_wh) as source:
        before = source.tobytes()
        with pytest.raises(ValueError, match="Unsupported display image mode"):
            module.annotate_image(source, result)
        assert source.tobytes() == before


def test_drawing_failure_closes_only_temporary_canvas(image, result, monkeypatch):
    made = []
    real_convert = Image.Image.convert

    def capture_canvas(source, *args, **kwargs):
        canvas = real_convert(source, *args, **kwargs)
        made.append(canvas)
        return canvas

    def fail_draw(canvas):
        raise OSError("drawing failed")

    monkeypatch.setattr(Image.Image, "convert", capture_canvas)
    monkeypatch.setattr(module.ImageDraw, "Draw", fail_draw)
    with pytest.raises(OSError, match="drawing failed"):
        module.annotate_image(image, result)
    with pytest.raises(ValueError, match="closed image"):
        made[0].getpixel((0, 0))
    assert image.getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.parametrize("empty", [False, True])
def test_saved_observations_annotate_without_changing_source_or_crops(
    tmp_path, image, result, empty
):
    statuses = ["read", "uncertain", "unreadable", "error"]
    result.plates = (
        []
        if empty
        else [
            replace(
                result.plates[0],
                detection=Detection((5 + 80 * index, 35, 65 + 80 * index, 75), 0.95),
                crop_bbox_xyxy=(3 + 80 * index, 33, 67 + 80 * index, 77),
                ocr=replace(result.plates[0].ocr, status=status),
            )
            for index, status in enumerate(statuses)
        ]
    )
    storage = SQLiteStorage(tmp_path / "observations.sqlite3")
    observation_id = storage.save_with_images(
        image, result, sources_dir=tmp_path / "sources", crops_dir=tmp_path / "crops"
    )
    reopened = SQLiteStorage(storage.database_path)
    restored = reopened.get(observation_id)
    files = reopened.get_image_files(observation_id)
    paths = [files.source_path, *files.crop_paths]
    before = [path.read_bytes() for path in paths]
    with (
        Image.open(files.source_path) as source,
        module.annotate_image(source, restored) as annotated,
    ):
        annotated.save(tmp_path / "overlay.png")
        with Image.open(tmp_path / "overlay.png") as output:
            assert output.size == source.size and output.mode == "RGB"
            if not empty:
                for index, color in enumerate(
                    [(0, 160, 0), (230, 160, 0), (120, 120, 120), (220, 0, 0)]
                ):
                    assert output.getpixel((5 + 80 * index, 60)) == color
    assert [path.read_bytes() for path in paths] == before
    assert reopened.get(observation_id) == result
    assert [p.ocr.status for p in restored.plates] == ([] if empty else statuses)

"""Crop contracts checked against explicit bounds and coordinate-coded pixels."""

from collections.abc import Iterator

import pytest
from PIL import Image

from parking_monitoring.pipeline import crop_plate


@pytest.fixture
def image() -> Iterator[Image.Image]:
    """Each pixel identifies its original (x, y), on a non-square source."""
    pixels = bytes(
        channel
        for y in range(60)
        for x in range(100)
        for channel in (x, y, x + y)
    )
    with Image.frombytes("RGB", (100, 60), pixels) as source:
        yield source


@pytest.mark.parametrize(
    "box,padding,expected",
    [
        ((10, 20, 30, 40), 0, (10, 20, 30, 40)),
        ((10.2, 20.7, 30.1, 40.3), 0, (10, 20, 31, 41)),
        ((10.2, 20.7, 30.1, 40.3), 2, (8, 18, 33, 43)),
        ((-3.2, 10.8, 30.1, 65.4), 0, (0, 10, 31, 60)),
        ((-3.2, 10.8, 30.1, 65.4), 2, (0, 8, 33, 60)),
        ((95.2, 55.8, 110.4, 70.2), 0, (95, 55, 100, 60)),
        ((-5, -5, 105, 65), 0, (0, 0, 100, 60)),
        ((99.9, 59.9, 100.1, 60.1), 0, (99, 59, 100, 60)),
        ((0, 0, 100, 60), 0, (0, 0, 100, 60)),
        ((0.2, 0.4, 0.9, 0.8), 0, (0, 0, 1, 1)),
        ((10, 20, 30, 40), 500, (0, 0, 100, 60)),
        ((-1e308, -1e308, 1e308, 1e308), 0, (0, 0, 100, 60)),
        ((-(10**400), -(10**400), 10**400, 10**400), 0, (0, 0, 100, 60)),
        ((10.2, 20.7, 30.1, 40.3), 10**400, (0, 0, 100, 60)),
    ],
)
def test_bounds_and_exact_original_pixels(image: Image.Image, box, padding, expected) -> None:
    before = image.tobytes()
    crop, bounds = crop_plate(image, box, padding)

    with crop:
        assert bounds == expected
        assert all(type(value) is int for value in bounds)
        left, top, right, bottom = expected
        assert crop.size == (right - left, bottom - top)
        # Independent pixel oracle: no call to Pillow crop to build expectations.
        expected_pixels = bytes(
            channel
            for y in range(top, bottom)
            for x in range(left, right)
            for channel in (x, y, x + y)
        )
        assert crop.mode == "RGB"
        assert crop.tobytes() == expected_pixels
    assert image.size == (100, 60)
    assert image.tobytes() == before


def test_recorded_detector_box_has_expected_bounds_and_size() -> None:
    box = (287.6219787597656, 223.14540100097656, 392.65081787109375, 332.2291564941406)
    with Image.new("RGB", (480, 480)) as source:
        crop, bounds = crop_plate(source, box)
        with crop:
            assert bounds == (287, 223, 393, 333)
            assert crop.size == (106, 110)


def test_mutating_crop_does_not_change_source(image: Image.Image) -> None:
    before = image.tobytes()
    crop, _ = crop_plate(image, (10, 20, 13, 22))
    with crop:
        assert crop is not image
        assert crop.getpixel((0, 0)) == (10, 20, 30)
        assert crop.getpixel((2, 1)) == (12, 21, 33)
        crop.putpixel((0, 0), (255, 255, 255))
        assert image.tobytes() == before
    assert image.getpixel((10, 20)) == (10, 20, 30)


@pytest.mark.parametrize(
    "mode,color", [("RGB", (11, 22, 33)), ("RGBA", (11, 22, 33, 44)), ("L", 128), ("P", 7)]
)
def test_mode_and_pixel_values_are_preserved(mode: str, color) -> None:
    with Image.new(mode, (10, 8), color) as source:
        if mode == "P":
            source.putpalette([channel for value in range(256) for channel in (value, 0, 255 - value)])
        crop, bounds = crop_plate(source, (2, 1, 7, 6))
        with crop:
            assert bounds == (2, 1, 7, 6)
            assert crop.mode == source.mode == mode
            assert crop.size == (5, 5)
            assert all(crop.getpixel((x, y)) == color for y in range(5) for x in range(5))
            if mode == "P":
                assert crop.getpalette() == source.getpalette()


@pytest.mark.parametrize("source", [None, "image.jpg"])
def test_non_image_input_is_rejected(source) -> None:
    with pytest.raises(ValueError, match="nonempty Pillow image"):
        crop_plate(source, (1, 1, 3, 3))


@pytest.mark.parametrize("size", [(0, 1), (1, 0)])
def test_empty_source_image_is_rejected(size: tuple[int, int]) -> None:
    with (
        Image.new("RGB", size) as source,
        pytest.raises(ValueError, match="nonempty Pillow image"),
    ):
        crop_plate(source, (0, 0, 1, 1))


@pytest.mark.parametrize("box", [None, [], [1, 1, 3, 3], (), (1, 2, 3), (1, 2, 3, 4, 5), "1234"])
def test_box_must_be_a_four_item_tuple(image: Image.Image, box) -> None:
    with pytest.raises(ValueError, match="tuple of four coordinates"):
        crop_plate(image, box)


@pytest.mark.parametrize("index,name", list(enumerate(("x_min", "y_min", "x_max", "y_max"))))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, False, "1", None])
def test_each_coordinate_must_be_finite_python_number(
    image: Image.Image, index: int, name: str, value
) -> None:
    box = [1, 1, 3, 3]
    box[index] = value
    with pytest.raises(ValueError, match=rf"bbox_xyxy\.{name} must be a finite"):
        crop_plate(image, tuple(box))


@pytest.mark.parametrize("box", [(5, 1, 2, 3), (1, 5, 3, 2), (1, 1, 1, 5), (1, 1, 5, 1)])
@pytest.mark.parametrize("padding", [0, 1000])
def test_reversed_and_zero_area_boxes_are_rejected(
    image: Image.Image, box, padding: int
) -> None:
    with pytest.raises(ValueError, match="x_min < x_max and y_min < y_max"):
        crop_plate(image, box, padding)


@pytest.mark.parametrize(
    "box",
    [
        (-20, 1, -10, 5), (110, 1, 120, 5),
        (1, -20, 5, -10), (1, 70, 5, 80),
        (-10, 1, 0, 5), (100, 1, 110, 5),
        (1, -10, 5, 0), (1, 60, 5, 70),
    ],
)
@pytest.mark.parametrize("padding", [0, 1000])
def test_outside_or_edge_touching_boxes_are_not_rescued_by_padding(
    image: Image.Image, box, padding: int
) -> None:
    with pytest.raises(ValueError, match="no positive-area overlap"):
        crop_plate(image, box, padding)


@pytest.mark.parametrize("padding", [-1, True, False, 0.5, "2", None])
def test_padding_must_be_a_nonnegative_integer(image: Image.Image, padding) -> None:
    with pytest.raises(ValueError, match="padding_pixels must be a nonnegative integer"):
        crop_plate(image, (1, 1, 3, 3), padding)


def test_closed_image_gives_actionable_value_error() -> None:
    source = Image.new("RGB", (10, 8))
    source.close()
    with pytest.raises(ValueError, match="Cannot extract pixels") as error:
        crop_plate(source, (1, 1, 3, 3))
    assert isinstance(error.value.__cause__, ValueError)


def test_pixel_read_failure_preserves_cause(
    image: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = OSError("synthetic image decoder failure")

    def unreadable(_bounds):
        raise failure

    monkeypatch.setattr(image, "crop", unreadable)
    with pytest.raises(ValueError, match="synthetic image decoder failure") as error:
        crop_plate(image, (1, 1, 3, 3))
    assert error.value.__cause__ is failure

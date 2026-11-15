"""Upload decoding and Streamlit interaction tests with isolated SQLite files."""

import json
import sqlite3
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import apps.streamlit_app as app
from apps.streamlit_app import decode_upload
from parking_monitoring.config import ConfigError, load_config
from parking_monitoring.schemas import Detection, ImageResult, OCRResult, PlateResult
from parking_monitoring.storage import SQLiteStorage


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_decoded_pixels_remain_usable_without_upload_buffer(image_format):
    with Image.new("RGB", (8, 5), "red") as source, BytesIO() as buffer:
        source.save(buffer, format=image_format)
        data = buffer.getvalue()
    with decode_upload(data) as decoded:
        assert decoded.mode == "RGB" and decoded.size == (8, 5)
        assert decoded.getpixel((0, 0))[0] > 250
        assert decoded.tobytes()


def test_exif_orientation_is_applied_once_and_pixels_are_independent():
    with Image.new("RGB", (3, 2), "red") as source, BytesIO() as buffer:
        source.putpixel((0, 0), (0, 255, 0))
        exif = Image.Exif()
        exif[274] = 6
        source.save(buffer, format="PNG", exif=exif)
        data = buffer.getvalue()
        with source.transpose(Image.Transpose.ROTATE_270) as expected:
            expected_pixels = expected.tobytes()
    with decode_upload(data) as decoded:
        assert decoded.size == (2, 3)
        assert decoded.tobytes() == expected_pixels
        assert decoded.getexif().get(274) is None
        decoded.putpixel((0, 0), (0, 0, 255))
    with decode_upload(data) as fresh:
        assert fresh.tobytes() == expected_pixels


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n", None])
def test_invalid_uploads_raise_clear_errors(data):
    with pytest.raises(ValueError, match="[Uu]pload"):
        decode_upload(data)


@pytest.mark.parametrize(("mode", "image_format"), [("RGB", "GIF"), ("I;16", "PNG")])
def test_unsupported_format_or_high_depth_is_rejected(mode, image_format):
    with Image.new(mode, (3, 2)) as source, BytesIO() as buffer:
        source.save(buffer, format=image_format)
        with pytest.raises(ValueError, match="Cannot decode upload"):
            decode_upload(buffer.getvalue())


def image_bytes(color="red"):
    with Image.new("RGB", (40, 25), color) as image, BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def button(at, label):
    return next(item for item in at.button if item.label == label)


@pytest.fixture
def ui(tmp_path, monkeypatch):
    config = load_config()
    config = replace(
        config,
        storage=replace(
            config.storage,
            database_path=tmp_path / "observations.sqlite3",
            output_dir=tmp_path / "output",
            crops_dir=tmp_path / "crops",
        ),
    )
    state = SimpleNamespace(initializations=0, calls=0, fail=False, empty=False)

    def process(image, source_id):
        state.calls += 1
        if state.fail:
            raise RuntimeError("forced inference failure")
        return ImageResult(
            source_id,
            image.size,
            []
            if state.empty
            else [
                PlateResult(
                    Detection((2, 3, 18, 15), 0.9),
                    (2, 3, 18, 15),
                    OCRResult("AB 123", "AB123", 0.95, "read"),
                ),
                PlateResult(
                    Detection((20, 3, 38, 15), 0.8),
                    (20, 3, 38, 15),
                    OCRResult(None, None, None, "error", "forced OCR failure"),
                ),
            ],
            {"total": 1.0},
            {"ocr_model": "controlled test engine"},
        )

    def initialize(_config):
        state.initializations += 1
        return SimpleNamespace(process=process)

    app._cached_pipeline.clear()
    app._inference_lock.clear()
    monkeypatch.setattr(app, "load_config", lambda: config)
    monkeypatch.setattr(app, "ParkingPipeline", initialize)

    def start():
        return AppTest.from_string(
            "from apps.streamlit_app import main\nmain()", default_timeout=10
        ).run()

    yield SimpleNamespace(config=config, state=state, start=start)
    app._cached_pipeline.clear()
    app._inference_lock.clear()


def test_upload_reruns_repeated_clicks_and_a_b_a_do_not_duplicate(ui):
    at = ui.start()
    assert not at.exception and ui.state.initializations == 0
    data = image_bytes()
    at.file_uploader[0].set_value(("same.png", data, "image/png")).run()
    assert ui.state.initializations == 0 and button(at, "Save observation").disabled
    button(at, "Process").click().run()
    assert not at.exception and not at.error
    assert ui.state.calls == ui.state.initializations == 1
    table = at.dataframe[0].value
    assert list(table["Status"]) == ["read", "error"]
    assert table.iloc[1]["OCR error"] == "forced OCR failure"
    # The button emitted before processing may still be enabled on this render.
    button(at, "Process").click().run()
    assert ui.state.calls == 1
    button(at, "Save observation").click().run()
    button(at, "Save observation").click().run()
    at.run()
    storage = SQLiteStorage(ui.config.storage.database_path)
    assert len(storage.list_recent()) == 1
    assert button(at, "Save observation").disabled

    # A different image with the same filename is a different hash/result.
    other = image_bytes("blue")
    at.file_uploader[0].set_value(("same.png", other, "image/png")).run()
    button(at, "Process").click().run()
    button(at, "Save observation").click().run()
    assert ui.state.calls == 2 and ui.state.initializations == 1
    at.file_uploader[0].set_value(("renamed.png", data, "image/png")).run()
    assert button(at, "Process").disabled
    assert button(at, "Save observation").disabled
    assert ui.state.calls == 2 and len(storage.list_recent()) == 2
    entry = at.session_state["uploads"][sha256(data).hexdigest()]
    assert entry["saved_id"] == 1 and entry["source_id"] == "same.png"
    assert entry["image"].tobytes() == bytes((255, 0, 0)) * (40 * 25)


def test_shared_model_cache_does_not_share_uploads_between_sessions(ui):
    first = ui.start()
    first.file_uploader[0].set_value(("first.png", image_bytes(), "image/png")).run()
    button(first, "Process").click().run()
    second = ui.start()
    assert second.session_state["uploads"] == {}
    assert ui.state.initializations == 1
    second.file_uploader[0].set_value(
        ("second.png", image_bytes("blue"), "image/png")
    ).run()
    button(second, "Process").click().run()
    assert not second.exception and ui.state.initializations == 1
    assert ui.state.calls == 2 and len(first.session_state["uploads"]) == 1


def test_invalid_upload_keeps_history_available_and_does_not_load_models(ui):
    at = ui.start()
    at.file_uploader[0].set_value(("bad.png", b"not image bytes", "image/png")).run()
    assert not at.exception and "Cannot decode upload" in at.error[0].value
    assert ui.state.initializations == 0 and at.session_state["uploads"] == {}
    assert any(item.value == "Recent history" for item in at.subheader)


def test_failed_processing_can_retry_without_a_false_empty_result(ui):
    ui.state.fail = True
    at = ui.start()
    at.file_uploader[0].set_value(("a.png", image_bytes(), "image/png")).run()
    button(at, "Process").click().run()
    assert "forced inference failure" in at.error[0].value
    assert button(at, "Save observation").disabled
    assert not any("No plates detected" in item.value for item in at.info)
    ui.state.fail = False
    button(at, "Process").click().run()
    assert not at.error and not at.exception and ui.state.calls == 2
    assert ui.state.initializations == 1


def test_failed_model_initialization_can_retry(ui, monkeypatch):
    factory = app.ParkingPipeline

    def fail(_config):
        raise RuntimeError("forced model initialization failure")

    monkeypatch.setattr(app, "ParkingPipeline", fail)
    at = ui.start()
    at.file_uploader[0].set_value(("a.png", image_bytes(), "image/png")).run()
    button(at, "Process").click().run()
    assert "forced model initialization failure" in at.error[0].value
    monkeypatch.setattr(app, "ParkingPipeline", factory)
    button(at, "Process").click().run()
    assert not at.exception and not at.error and ui.state.calls == 1


def test_zero_plates_is_a_valid_save_and_history_survives_new_session(ui):
    ui.state.empty = True
    at = ui.start()
    at.file_uploader[0].set_value(("empty.png", image_bytes(), "image/png")).run()
    button(at, "Process").click().run()
    assert any("No plates detected" in item.value for item in at.info)
    button(at, "Save observation").click().run()
    storage = SQLiteStorage(ui.config.storage.database_path)
    assert storage.get(1).plates == [] and storage.get_image_files(1).crop_paths == []
    app._cached_pipeline.clear()
    fresh = ui.start()
    fresh.selectbox[0].select(1).run()
    assert not fresh.exception and not fresh.error
    assert any("No plates detected" in item.value for item in fresh.info)
    assert ui.state.initializations == 1


def test_storage_save_failure_rolls_back_and_retry_sets_id_once(ui):
    at = ui.start()
    at.file_uploader[0].set_value(("a.png", image_bytes(), "image/png")).run()
    button(at, "Process").click().run()
    with sqlite3.connect(ui.config.storage.database_path) as connection:
        connection.execute("""
            CREATE TRIGGER reject_save BEFORE INSERT ON observations
            BEGIN SELECT RAISE(ABORT, 'forced storage failure'); END
        """)
    button(at, "Save observation").click().run()
    assert "forced storage failure" in at.error[0].value and not at.exception
    storage = SQLiteStorage(ui.config.storage.database_path)
    assert storage.list_recent() == []
    assert not button(at, "Save observation").disabled
    entry = next(iter(at.session_state["uploads"].values()))
    assert entry["result"] is not None and entry["saved_id"] is None
    with sqlite3.connect(ui.config.storage.database_path) as connection:
        connection.execute("DROP TRIGGER reject_save")
    button(at, "Save observation").click().run()
    assert entry["saved_id"] == 1 and len(storage.list_recent()) == 1
    button(at, "Save observation").click().run()
    assert len(storage.list_recent()) == 1 and ui.state.calls == 1


def test_missing_history_files_show_metadata_and_error_without_model_loading(ui):
    at = ui.start()
    at.file_uploader[0].set_value(("a.png", image_bytes(), "image/png")).run()
    button(at, "Process").click().run()
    button(at, "Save observation").click().run()
    storage = SQLiteStorage(ui.config.storage.database_path)
    storage.get_image_files(1).crop_paths[0].unlink()
    fresh = ui.start()
    fresh.selectbox[0].select(1).run()
    assert not fresh.exception
    assert "Cannot open saved observation" in fresh.error[0].value
    assert json.loads(fresh.json[0].value)["source_id"] == "a.png"
    assert ui.state.calls == ui.state.initializations == 1


@pytest.mark.parametrize("failure", ["config", "storage", "history"])
def test_boundary_errors_are_visible(ui, monkeypatch, failure):
    def fail(*_args, **_kwargs):
        if failure == "config":
            raise ConfigError("forced config failure")
        raise sqlite3.OperationalError(f"forced {failure} failure")

    if failure == "config":
        monkeypatch.setattr(app, "load_config", fail)
    elif failure == "storage":
        monkeypatch.setattr(app, "SQLiteStorage", fail)
    else:
        monkeypatch.setattr(SQLiteStorage, "list_recent", fail)
    at = ui.start()
    assert not at.exception and f"forced {failure} failure" in at.error[0].value
    assert ui.state.initializations == 0

"""Local still-image upload, processing, inspection, explicit save and history."""

import sqlite3
from _thread import LockType
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from threading import Lock

import streamlit as st
from PIL import Image, ImageOps

from parking_monitoring.config import PROJECT_ROOT, AppConfig, ConfigError, load_config
from parking_monitoring.pipeline import ParkingPipeline
from parking_monitoring.schemas import ImageResult
from parking_monitoring.storage import SQLiteStorage
from parking_monitoring.visualization import annotate_image


@st.cache_resource(show_spinner=False)
def _cached_pipeline(config_path: str, _config: AppConfig) -> ParkingPipeline:
    """Cache local models by configuration path, excluding the settings object."""
    return ParkingPipeline(_config)


@st.cache_resource(show_spinner=False)
def _inference_lock() -> LockType:
    """Serialize access to the shared detector/OCR engines across sessions."""
    return Lock()


def get_pipeline(config: AppConfig) -> ParkingPipeline:
    """Return the reusable pipeline for the fixed MVP configuration path.

    Call only on Process, inside the shared inference lock. Configuration changes
    at the same path require a server restart to replace cached models.
    Initialization failures propagate to the caller and can be retried.
    """
    # 1. The path is hashed; the underscore-prefixed settings object is not.
    config_path = str((PROJECT_ROOT / "configs/mvp.yaml").resolve())
    return _cached_pipeline(config_path, config)


def decode_upload(data: bytes) -> Image.Image:
    """Decode a JPEG/PNG into an independent, orientation-corrected RGB image.

    The caller owns the returned image and must use the same pixels for
    inference, annotation and saving. No models, files or database are accessed.
    Reject high-depth modes instead of silently saturating their pixel values.

    Raises:
        ValueError: Input is empty, invalid, unsupported or cannot be decoded.
    """
    # 1. Validate bytes and decode pixels before closing the upload buffer.
    if not isinstance(data, bytes) or not data:
        raise ValueError("Upload must contain nonempty image bytes")
    try:
        with BytesIO(data) as buffer, Image.open(buffer) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Upload must be a JPEG or PNG image")
            if source.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                raise ValueError(f"Unsupported upload image mode: {source.mode}")
            # 2. Apply EXIF orientation once, then own independent RGB pixels.
            with ImageOps.exif_transpose(source) as oriented:
                return oriented.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot decode upload: {exc}") from exc


def render_result(image: Image.Image, result: ImageResult) -> None:
    """Display annotation, crops, OCR text/status/errors, scores and timings.

    Use the same decoded source pixels as processing. Rendering creates no
    observations and does not modify the source or result. Image/annotation
    errors propagate to the caller; temporary display images are closed.
    """
    # 1. Show an independent annotation without modifying stored evidence.
    with annotate_image(image, result) as annotated:
        st.image(annotated, caption="Detected plates", width="stretch")

    # 2. Preserve every OCR outcome, including technical failures and no plates.
    if not result.plates:
        st.info("No plates detected above the configured threshold.")
    else:
        st.dataframe(
            [
                {
                    "Plate": index + 1,
                    "Raw text": plate.ocr.raw_text,
                    "Normalized text": plate.ocr.normalized_text,
                    "Status": plate.ocr.status,
                    "Detection score": plate.detection.score,
                    "OCR score": plate.ocr.score,
                    "OCR error": plate.ocr.error_message,
                }
                for index, plate in enumerate(result.plates)
            ],
            hide_index=True,
        )
        for index, plate in enumerate(result.plates):
            with image.crop(plate.crop_bbox_xyxy) as crop:
                st.image(crop, caption=f"Plate {index + 1}: {plate.ocr.status}")

    # 3. Report measured processing timings, excluding model initialization.
    with st.expander("Processing timings (ms)"):
        st.json(result.timings_ms)


def main() -> None:
    """Run upload, processing, inspection, explicit saving and history."""
    # 1. Load configuration and initialize per-hash session state.
    st.set_page_config(page_title="AI Parking Monitoring", layout="wide")
    st.title("AI Parking Monitoring")
    st.caption("Upload a still image, inspect the readings, then save explicitly.")
    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"Cannot load configuration: {exc}")
        return
    uploads = st.session_state.setdefault("uploads", {})
    storage = None
    try:
        storage = SQLiteStorage(config.storage.database_path)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f"Storage unavailable: {exc}")

    # 2. Decode/select the upload and retrieve its existing state.
    upload = st.file_uploader("Upload a JPEG or PNG image", type=["jpg", "jpeg", "png"])
    if upload is not None:
        data = upload.getvalue()
        upload_hash = sha256(data).hexdigest()
        if upload_hash not in uploads:
            try:
                image = decode_upload(data)
            except ValueError as exc:
                st.error(str(exc))
            else:
                uploads[upload_hash] = {
                    "image": image,
                    "source_id": upload.name,
                    "result": None,
                    "saved_id": None,
                }
        entry = uploads.get(upload_hash)
        if entry is not None:
            st.caption(f"Source: {entry['source_id']}")
            if entry["result"] is None:
                st.image(entry["image"], caption="Uploaded image", width="stretch")

            # 3. Process only on click when this hash has no result.
            process_clicked = st.button(
                "Process",
                key=f"process_{upload_hash}",
                disabled=entry["result"] is not None,
            )
            if process_clicked and entry["result"] is None:
                try:
                    with st.spinner("Processing image…"), _inference_lock():
                        pipeline = get_pipeline(config)
                        entry["result"] = pipeline.process(
                            entry["image"], entry["source_id"]
                        )
                except Exception as exc:  # noqa: BLE001
                    # Component failures must remain distinguishable from no detections.
                    st.error(f"Processing failed: {exc}")

            # 4. Display the result; save explicitly once, retaining the saved ID.
            if entry["result"] is not None:
                try:
                    render_result(entry["image"], entry["result"])
                except (OSError, ValueError) as exc:
                    st.error(f"Cannot display result: {exc}")
            save_clicked = st.button(
                "Save observation",
                key=f"save_{upload_hash}",
                disabled=(
                    entry["result"] is None
                    or entry["saved_id"] is not None
                    or storage is None
                ),
            )
            if (
                save_clicked
                and entry["result"] is not None
                and entry["saved_id"] is None
                and storage is not None
            ):
                try:
                    entry["saved_id"] = storage.save_with_images(
                        entry["image"],
                        entry["result"],
                        sources_dir=config.storage.output_dir / "sources",
                        crops_dir=config.storage.crops_dir,
                    )
                except (OSError, ValueError, sqlite3.Error) as exc:
                    st.error(f"Save failed; the result is available for retry: {exc}")
            if entry["saved_id"] is not None:
                st.success(f"Saved observation #{entry['saved_id']}.")

    # 5. Show recent history; display metadata plus errors for missing files.
    st.divider()
    st.subheader("Recent history")
    st.caption("Latest 20 saved observations. Saved history survives a server restart.")
    if storage is None:
        return
    try:
        recent = storage.list_recent()
    except (ValueError, sqlite3.Error) as exc:
        st.error(f"Cannot list history: {exc}")
        return
    if not recent:
        st.info("No saved observations yet.")
        return
    by_id = {item.observation_id: item for item in recent}
    selected_id = st.selectbox(
        "Saved observation",
        options=list(by_id),
        index=None,
        format_func=lambda value: (
            f"#{value} — {by_id[value].source_id} — {by_id[value].created_at}"
        ),
        key="history_selection",
        placeholder="Choose an observation",
    )
    if selected_id is None:
        return
    try:
        result = storage.get(selected_id)
        if result is None:
            st.error("This observation no longer exists.")
            return
        with st.expander("Saved metadata", expanded=True):
            st.json(asdict(result), expanded=False)
        files = storage.get_image_files(selected_id)
        if files is None:
            st.info("This observation contains metadata only; no images were saved.")
        else:
            with Image.open(files.source_path) as source:
                render_result(source, result)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f"Cannot open saved observation: {exc}")


if __name__ == "__main__":
    main()

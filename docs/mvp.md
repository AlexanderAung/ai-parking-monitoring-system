# Plate recognition MVP

Status: scaffold only. Application, inference wrappers, and tests are not implemented.

## Scope

Upload a still image, detect license plates with the trained RT-DETR baseline,
read Latin registration letters and digits with pretrained OCR, inspect crops and
results, and save observations for later retrieval. Burmese text is outside this MVP.

## Setup

Use Python 3.12 and uv from the repository root:

```bash
uv add streamlit paddleocr paddlepaddle
uv add --dev pytest ruff jiwer
uv sync
```

These commands update pyproject.toml and uv.lock and install the local source
package. Dependencies have not yet been resolved or smoke-tested together.
PaddleOCR model weights are separate from Python packages and may download on
first inference. Record the chosen OCR model and version after the crop experiment.

Pillow, NumPy, PyYAML, torch, and torchvision are already project dependencies.
SQLite, pathlib, logging, and dataclasses are in the Python standard library.
Pillow is sufficient for initial cropping; add OpenCV only when image processing
or video requires it, checking for an existing transitive OpenCV distribution first.

## Layout

- apps/: Streamlit interface.
- src/parking_monitoring/: reusable inference, configuration, results, and storage.
- configs/mvp.yaml: proposed local paths and settings; paths resolve from repo root.
- scripts/: command-line prediction and evaluation entry points.
- tests/: placeholders for crop, pipeline, and persistence checks.
- data/mvp/: local images and manually transcribed development/test labels.
- artifacts/mvp/: generated crops, database, and inference output; ignored by Git.
- docs/reports/mvp-001/: measured results and selected failure examples.

The UI and CLI should call the same Python pipeline. Keep rendering and
persistence outside the core inference function.

## Build sequence

1. Verify detector checkpoint/config loading and OCR on representative crops.
2. Implement structured results and the image pipeline.
3. Implement the upload interface, persistence, and history.
4. Evaluate on held-out images and document failures and runtime.

## Completion checklist

- [ ] Detector and OCR smoke tests pass on the local environment.
- [ ] Multiple plates, no detections, and unreadable crops have explicit outcomes.
- [ ] Original-resolution crops and display overlays can be inspected.
- [ ] Observations survive an application restart.
- [ ] Held-out exact-match accuracy, character error rate, and latency are reported.
- [ ] Setup instructions and permitted example images allow another person to run it.

See [architecture](architecture.md), [evaluation protocol](evaluation-protocol.md),
and [OCR decision](decisions/003-ocr-selection.md).

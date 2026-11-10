# Detector — independent-review handoff

Date: 2026-11-10. D1.3's first arbitrary-image path is verified; D1.4 non-square
mapping and safe cropping are pending. Independent-review findings are not recorded.

## Responsibility and interface

`src/parking_monitoring/detector.py` implements:

```python
class RTDETRDetector:
    def __init__(self, config: DetectorConfig) -> None: ...
    def predict(self, image: Image.Image) -> list[Detection]: ...
```

The constructor initializes reusable resources; predict accepts a nonempty Pillow
image and returns Python-valued Detection objects from existing schemas.py. Boxes
are floating-point original-image xyxy coordinates; scores must strictly exceed
the configured threshold. An empty list is a successful no-detection result.
FileNotFoundError reports missing resources, ValueError invalid inputs/settings,
and RuntimeError initialization/inference failures with retained causes.

The CLI/caller owns image-file loading and result drawing/saving. Cropping owns
subsequent rounding and clamping; OCR, storage and UI are separate components.

## Decisions and implementation

- Reuse the existing baseline YAML builder, EMA-loading path and postprocessor.
  Keep checkpoint0019.pth in its configured runs location, without duplication.
- Validate settings and local input files, require a single class without COCO
  remapping, disable separate PResNet pretrained loading, and restore ema.module
  with strict parameter matching. Move model/postprocessor to the configured
  device and evaluation mode once.
- Construct only Compose from the merged validation-transform dictionary; never
  construct cfg.val_dataloader. Current transforms resize to 640 x 640 and produce
  scaled float32 RGB pixels. Keep the original source image intact.
- Under torch.inference_mode, preprocess a separate RGB copy, batch the tensor,
  run the model and pass original [width, height] to the existing postprocessor.
  Its returned boxes are already in original pixels; do not rescale them again.
- Validate result shapes, finite/ordered boxes, single-class labels and score
  ranges; filter scores and construct Detection values. Outside-image bounds
  are retained for the next crop step to handle.
- The private backend helper uses the existing source checkout and upstream
  package name src; it checks the imported module origin for shadowing. This
  source-tree/layout dependency has not been tested as an installed distribution.

## Tests and runtime evidence

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_detector.py -q -p no:cacheprovider
69 passed in 1.53s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider
Final run: 149 passed in 1.99s (80 configuration + 69 detector cases)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check --no-cache tests/test_detector.py
Final run: All checks passed!
```

All commands exited 0 in their final runs. Initial Ruff findings were four SIM117
test-style issues and were fixed; no pytest failures or detector fixes occurred.
Environment: Python 3.12.9, PyTorch 2.14.1, torchvision 0.29.1, pytest 9.2.0.
Tests use controlled backend components/mocked torch.load and a tiny actual
PyTorch module for strict synthetic parameter loading. They verify wrapper
contracts, not real-model accuracy or the vendored preprocessing implementation.

Separate user-run evidence from the supplied Python-session commands:

- RTDETRDetector(load_config().detector) completed and the user reported the
  supplied EMA-initialization success message. Current configuration selects CPU,
  configs/training/baseline-001-eval.yml and the existing epoch-19 checkpoint.
- Prediction on data/mvp/images/image291_jpg.rf.fbd83e8ba4fbf26dfb8e92d3e43d2c21.jpg
  (480 x 480) returned box (287.6219787597656, 223.14540100097656,
  392.65081787109375, 332.2291564941406), score 0.8843116760253906.
- Agent directly opened artifacts/mvp/results/detector-check-de6537eb/overlay.png
  and confirmed the rectangle encloses the visible plate and the label matches.
  Metadata confirms 480 x 480 RGB. The inference itself was not agent-rerun.
- Checkpoint SHA-256 was independently read by the agent and matches the saved
  baseline manifest: a48b7e760bc273ed8ba52a6f749b0944d2e420e2393d71917fd01df99030d871.

## Limitations and learning gaps

- One square development-image smoke test is not an accuracy benchmark. Real
  wide/tall mapping, multiple/no-plate image cases, startup/warm latency and
  broader quality measurements remain unverified. No OCR or crops are implemented.
- The standalone prediction CLI, storage/UI integration and portability are
  pending. Unscoped vendored tests were not run. The overlay and checkpoint are
  local ignored artifacts. This image is development-exposed, not untouched holdout.
- User explained architecture versus weights and model reuse, and proposed a
  correct ratio approach but needed guidance on separate axis scaling. User wrote
  a scaffold against a supplied contract; Codex corrected naming/indentation and
  implemented internals/tests. Guided runtime execution is practical usage evidence,
  not proof of independent implementation mastery. See learning/q-and-a.md.

Please review interface/responsibility boundaries, backend/config assumptions,
EMA restoration, preprocessing/coordinate semantics, failure handling and whether
the evidence supports the stated completion. Return actionable findings within
the still-image MVP scope. Remaining Day 1 goals are not marked complete.

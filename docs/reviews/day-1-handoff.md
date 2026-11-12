# Day 1 — independent-review handoff

Date: 2026-11-11. Work session spans November 10-11; focused hours were not measured.
All five technical Day 1 acceptance criteria are verified. The user subsequently
reported finishing the review; no findings/report were supplied and none are
invented. All planned tasks are checked. The user then accepted Day 1 closeout
and requested a prompt to start Day 2 in a new chat; no Day 2 work started here.

## Result and public interfaces

Local still images now pass through real EMA detector initialization, inference,
original-coordinate postprocessing and safe original-pixel crop extraction.
Native crops, overlays and evidence files are saved for the selected examples.

```python
load_config(
    path: str | Path = "configs/mvp.yaml",
    *, project_root: str | Path | None = None,
) -> AppConfig

RTDETRDetector(config: DetectorConfig)
RTDETRDetector.predict(image: Image.Image) -> list[Detection]

crop_plate(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    padding_pixels: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]]
```

All approved public interfaces and schemas.py contracts are preserved.

## Architecture and implementation

- Configuration reads explicit YAML, validates required fields/types/ranges and
  resolves paths against the project root. It returns frozen settings containers.
  Resource availability/runtime compatibility belong to components. Loading
  configuration performs no model initialization/downloads/output creation.
- The detector reuses the vendored RT-DETR YAML builder and postprocessor. It
  disables separate PResNet pretraining, strictly restores ema.module, and loads
  once per instance in evaluation mode. It builds validation transforms without
  constructing a COCO loader or using annotations.
- Prediction preserves the source image, preprocesses an RGB copy, runs under
  inference mode and supplies original [width, height] to the postprocessor.
  Finite/ordered boxes, labels and score ranges are checked; score > threshold
  filtering returns Python-valued Detection objects. Empty results are valid.
- Crop code in pipeline.py requires finite native numeric tuple coordinates and
  nonnegative integer padding, rejects no-overlap boxes before padding, rounds
  outward and clamps to each image dimension. It preserves original pixels/mode
  and returns the exact integer bounds; right/bottom are exclusive. Integer
  padding/floor/ceil arithmetic avoids float overflow while preserving semantics.
- User developed a guarded manual main through guided practice. Codex refined
  it to process both selected inputs with one detector, close files, pass padding,
  skip empty results and save first-crop PNG/metadata in fresh directories. This
  remains a two-example manual inspection runner; scripts/predict_image.py and
  the full detection/OCR pipeline are still placeholders/later work.
- Errors distinguish invalid configuration/settings/image/box input, missing files
  and technical initialization/inference failures. Detector ValueError contracts
  were preserved with documented local Ruff TRY004 exceptions rather than silently
  changed. No new dependencies or production features were added.

## Tests and actual final results

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider
240 passed in 1.74s; exit 0; no warnings or skips

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check --no-cache src/parking_monitoring/config.py src/parking_monitoring/detector.py src/parking_monitoring/pipeline.py tests/test_config.py tests/test_detector.py tests/test_cropping.py
All checks passed!; exit 0
```

Totals: 80 configuration, 69 controlled detector, 91 crop cases. Unit detector
tests use a tiny PyTorch module and controlled backend; they are separate from
the real runtime evidence below. Initial broad lint had six findings; import/type
annotation style was fixed and exception-contract preferences documented. No
pytest failures occurred in this finishing turn. Source/test patch checks pass.

Environment: Python 3.12.9, PyTorch 2.14.1, torchvision 0.29.1, pytest 9.2.0,
PyYAML 6.0.3. Current settings select CPU, detector threshold 0.6, crop padding 0,
baseline-001-eval.yml and checkpoint0019.pth. Checkpoint SHA-256 again matches
the saved baseline manifest: a48b7e760bc273ed8ba52a6f749b0944d2e420e2393d71917fd01df99030d871.

## Real runtime and visual evidence

The agent ran actual pipeline.main() via a preserved verification script, with
only Image.Image.show suppressed to avoid desktop popups. No model, checkpoint
loading, prediction or crop behavior was mocked. One real detector processed both
inputs, without an annotation file or dataset loader.

| Input | Original | Crop bounds | Native crop | Detector score |
| --- | --- | --- | --- | --- |
| pic_286, wide | 2048 x 1536 | (374, 330, 1818, 1213) | 1444 x 883 | 0.8614094853401184 |
| image037, tall | 361 x 480 | (157, 248, 224, 285) | 67 x 37 | 0.8702967166900635 |

Each returned one detection and reproduced the user's earlier reported bounds.
Independent source byte-row comparisons establish exact native crop pixels;
original mode, bounds, source hashes and source-file preservation pass. Existing
draw_prediction generated original-size overlays. The agent opened both overlays
and crops and confirmed full plate/character coverage. The wide box includes
frame/background; no localization-tightness or OCR-quality metric is claimed.

Artifacts under artifacts/mvp/results/:

- wide-check-4811d7a2/{crop.png,overlay.png,result.json}.
- tall-check-d5a0d005/{crop.png,overlay.png,result.json}.
- non-square-verification-z10zeeuq/{report.json,stdout.txt,verify_non_square.py}.
- Earlier square-image evidence: detector-check-de6537eb/overlay.png and
  crop-check-t1fcbitw/{crop.png,report.json,verify_crop.py}.

The final real command was:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python /tmp/parking-non-square-di05z91o.py
Exit 0
```

The script copy is rerunnable from the repository root using the same environment
prefix; each run creates fresh output folders. Report includes actual settings,
code/input/checkpoint/output hashes, package versions and visual-inspection notes.
Images/checkpoint/results are local ignored artifacts and are not bundled.

## Limits, provenance and remaining work

- These are development smoke checks, including an already-used baseline validation
  source, not held-out evaluation. No accuracy/CER, calibrated acceptance coverage,
  real multi/no-plate image benchmark, startup/warm latency or portability claim.
- OCR, structured full-pipeline assembly, database persistence, production CLI,
  Streamlit and later evaluation are not implemented by Day 1. Pipeline/storage
  test files remain placeholders. Unscoped vendored tests were not run.
- User supplied interfaces and iterated the manual driver with prompts; Codex
  implemented core internals/tests and final resource/evidence refinements. Original
  interview answers remain verbatim in learning/q-and-a.md. Original resolution
  and discrete pixels were identified; geometry arithmetic needed full guidance.
  Imports, calls and early-return reasoning are developing with support; independent
  implementation, geometry and test-design mastery are not established.
- D1.1-D1.4 technical goals are verified; D1.5 review completion is user-reported.
  Day 1 sign-off is accepted; Day 2 will begin in a new chat. No merge/commit occurred.

Please independently review boundaries/interfaces, backend/config assumptions,
EMA restoration, geometry/error handling, test adequacy and evidence provenance.
Return actionable findings within the agreed still-image MVP scope. Module-level
details are in configuration-handoff.md, detector-handoff.md and cropping-handoff.md.

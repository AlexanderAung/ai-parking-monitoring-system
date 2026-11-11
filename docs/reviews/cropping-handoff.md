# Cropping — independent-review handoff

Date: 2026-11-11, updated after final verification. Cropping is unit-tested and
verified on square/wide/tall real sources, including agent-run real detection
and saved crop/overlay inspection for both non-square cases. D1.4 technical
criteria are verified. External review is pending; see day-1-handoff.md.

## Responsibility and interface

In src/parking_monitoring/pipeline.py:

```python
def crop_plate(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    padding_pixels: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]]: ...
```

One original Pillow image and original-pixel xyxy box produce an independent
crop and exact integer (left, top, right, bottom) bounds. Left/top are inclusive;
right/bottom are exclusive, matching existing PlateResult.crop_bbox_xyxy.
The crop function has no new result schema, file writing, OCR call or full-pipeline API.
The caller passes padding from configuration and handles ValueError per its policy.
The user-authored manual main below the helper was refined to process the two
selected sources with one detector instance, close images, pass padding and save
the first crop plus result.json in fresh folders. This is an inspection caller,
not full detection-to-OCR orchestration or the standalone prediction CLI.

## Approved decisions and implementation

- Require a nonempty usable Pillow image, a four-item tuple of finite native
  int/float coordinates (excluding bool), ordered positive-area bounds, and
  a nonnegative integer padding value (excluding bool).
- Reject no-overlap/edge-only boxes before padding; padding cannot rescue them.
  Partial overlap is valid. Round outward, add padding and clamp to each image
  dimension. For integer p, floor(x - p) = floor(x) - p (and likewise ceil), so
  integer arithmetic preserves the approved padding/rounding result without
  float overflow or precision loss. Require nonempty final bounds.
- Pillow crop extracts original pixels and preserves source mode/pixel values.
  Expected source-read/crop failures become descriptive ValueError with cause.
- Numbered comments map the eight user-approved implementation items to code.
  The user wrote the reviewed scaffold; Codex implemented internals and tests.

## Actual tests and evidence

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_cropping.py -q -p no:cacheprovider
Initial targeted run: 91 passed in 0.13s with one pytest deprecation warning.

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider
Final scoped run after fixing that test input: 240 passed in 2.17s; no warnings.

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check --no-cache src/parking_monitoring/pipeline.py tests/test_cropping.py
All checks passed!
```

All commands exited 0. The warning was enumerate used as parametrize input;
converting it to a list fixed it. No production defects or logic changes were
needed. Final totals: 80 configuration + 69 controlled detector + 91 crop cases.
The crop suite uses explicit bounds and coordinate-coded pixel oracles, covers
mode/source independence, fractional/extreme/invalid/outside geometry and errors.
It does not implement a new batch interface; empty detector results are already
covered by the detector tests.

The results above describe the initial crop verification. The latest final
scoped rerun after manual-runner refinements is 240 passed in 1.74s; focused Ruff
passes on config/detector/pipeline and all three test files. Exact commands and
initial lint findings/resolution are recorded in the progress log.

Real extraction used image291_jpg.rf.fbd83e8ba4fbf26dfb8e92d3e43d2c21.jpg
(480 x 480 RGB) and the user's previously reported box
(287.6219787597656, 223.14540100097656, 392.65081787109375, 332.2291564941406).
Padding 0 produced (287, 223, 393, 333), a 106 x 110 RGB crop. Every pixel matches
the source region; source mode/pixels and file hash are unchanged; saved PNG
round trip is lossless. The agent opened the crop and confirmed the plate is visible.

Evidence under artifacts/mvp/results/crop-check-t1fcbitw/: crop.png, report.json,
and verify_crop.py. The script is rerunnable from the project root and makes a new
output folder. The agent ran it initially from /tmp/parking-crop-smoke-44fm8r4w.py.
No model inference or annotation loader was run during this extraction check.
The source and crop hashes are in the report and progress log. These are local
ignored artifacts; the source is development-exposed, not untouched holdout.

Subsequent agent-run real CPU verification processed the selected wide and tall
images through actual main(). Both native crops and original-size overlays were
opened by the agent and contain the full visible plate. Independent byte-row
pixel checks and source hashes pass. Wide bounds (374, 330, 1818, 1213) produce
1444 x 883; tall bounds (157, 248, 224, 285) produce 67 x 37. Viewer popups alone
were suppressed; model inference and file writes were real. Evidence:
artifacts/mvp/results/non-square-verification-z10zeeuq/{report.json,stdout.txt,verify_non_square.py}.
Saved files are in wide-check-4811d7a2/ and tall-check-d5a0d005/ under the results root.

## Limits and learning evidence

- Real wide/tall smoke inspection now passes; synthetic unit checks and actual
  model execution remain distinct evidence. No broader accuracy metric is established.
- No OCR, full detection-to-OCR orchestration, standalone CLI, storage or UI
  integration was added. Unscoped vendored tests were not run.
- User identified original resolution and discrete pixel selection but needed
  guidance on postprocessing ownership, returning actual bounds and floor/ceil/
  clamping arithmetic. No solution was supplied to the applied exercise; the
  worked example was Codex-guided. Original answers are preserved in Q&A.
- Scaffold writing was supported by supplied contracts; generated implementation,
  tests and annotations do not establish independent user mastery.

Please review responsibility/interface choices, finite/type validation,
overlap-before-padding semantics, integer rounding/clamping, source preservation
and whether the tests/evidence establish the stated behavior. Return actionable
findings within the agreed MVP scope. Day 1 completion has not been accepted.

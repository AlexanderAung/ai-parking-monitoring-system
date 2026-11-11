# Proposed MVP architecture

Status: planned architecture; the current MVP files are scaffolding.

```mermaid
flowchart LR
    IN([Input: image file]) --> UP[Streamlit upload]

    subgraph PIPE["Python pipeline"]
        direction LR
        D[RT-DETR detection] --> C[Crop plates]
        C --> O[PaddleOCR]
        O --> R[Prepare results and status]
    end

    UP --> D
    R --> VIEW[Streamlit results view]
    VIEW --> OUT([Output: annotated image, plate crops, text and status])

    R --> SAVE[Storage layer]
    SAVE --> DB[(SQLite metadata)]
    SAVE --> FILES[Crop files]
```

1. The image file enters through Streamlit upload. 
2. The Python pipeline groups the detection, cropping, OCR, and result-preparation components. It returns structured data to the results view, which presents the annotated image, plate crops, text, and status to the user. 
3. Storage branches from the structured results.

The detector returns boxes in original-image coordinates. The pipeline validates
and clamps coordinates before cropping. Visualization draws on a separate copy.
Load models once and reuse them across images.

Results should preserve raw and normalized OCR text, separate detection and OCR
scores, bounding boxes, status, source ID, timing, and model versions. Scores are
model outputs, not calibrated probabilities of a correct plate reading.

Storage saves observation metadata in SQLite and crop references on disk. UI
reruns must not automatically create duplicate observations. The prediction CLI
can call the same pipeline; it is omitted from this diagram to keep the primary
user journey clear. Later API or video entry points can also reuse the pipeline;
tracking and event logic remain future work.

## Configuration boundary — agreed 2026-11-10

The app or prediction CLI loads YAML once and passes relevant settings to the
detector, OCR, pipeline/cropping, and storage components. Those components do not
read the YAML themselves.

Configuration owns required fields, types, supported values, numeric ranges and
path resolution against the project root. It returns structured settings or a
clear configuration error, without initializing/downloading models, creating
output directories, or opening the database.

Each component validates the resources it needs when initializing. A missing OCR
model therefore does not prevent detector-only work. Resolving a path does not
establish that the resource exists or can load.

Frozen dataclasses for component settings and the grouped AppConfig were agreed
on 2026-11-10. The user's field/signature skeleton was reviewed against the
supplied contract; decorator spelling and missing class docstrings were corrected.
The internal loader is implemented against that interface. After the review
guide, the user explicitly instructed test rewriting/execution. The scoped
project suite passes 80 configuration cases, including the actual MVP YAML from
another working directory. Other module tests are still placeholders; app/CLI
integration and model initialization remain unverified. The independent-review
[handoff](reviews/configuration-handoff.md) is prepared for the user.

The loader requires exactly the fields declared by the settings dataclasses and
does not substitute defaults for missing YAML fields. Thresholds must be finite
numbers in [0, 1]; crop padding is an integer >= 0 and CPU threads an integer >= 1.
Quoted booleans and numbers are rejected instead of silently converted. The OCR
engine is restricted to paddleocr; device, language and model-version fields
are nonempty strings whose backend compatibility belongs to the corresponding
component. Loading settings does not establish hardware or model compatibility.

The default root is derived from config.py's source-checkout location. The
keyword-only project_root override accepts an explicit alternative (relative
overrides use the working directory). Relative YAML paths and paths inside YAML
use that root, even if the YAML file is elsewhere. Absolute paths are preserved
subject to user-home expansion and canonical resolution. Path resolution does
not check that models or output directories exist.

## Detector boundary — agreed 2026-11-10

The user approved this design after discussing checkpoint placement, image input,
preprocessing/postprocessing and output locations:

```python
class RTDETRDetector:
    def __init__(self, config: DetectorConfig) -> None:
        """Validate detector resources and load the model and postprocessor."""

    def predict(self, image: Image.Image) -> list[Detection]:
        """Return plate boxes in original-image pixels and detection scores."""
```

Image.Image denotes a Pillow image. The CLI opens the user's image path and
passes the loaded image to the reusable detector; a future Streamlit caller can
pass an uploaded image through the same interface. The detector loads the model
once, preprocesses each image, runs inference, postprocesses with original image
dimensions, and applies the configured score threshold. Empty successful results
return []; initialization/inference failures raise errors. Error types and
threshold equality behavior are specified in the reviewed skeleton contract below.

Use the existing epoch-19 checkpoint at the configured runs/baseline-001 path;
no copy into models/rtdetr is needed. The file's SHA-256 was checked on 2026-11-10
and matches the baseline artifact manifest. The user subsequently reported a
successful real EMA initialization using the provided terminal sequence;
actual image inference remains unverified. Reuse the validation model-loading and
postprocessing logic without constructing a COCO dataset or loader. The inspected
validation transforms resize to 640 x 640 and convert to scaled float32 pixels;
retain the original image for future cropping.

The detector returns existing schemas.Detection objects; it does not save or
draw outputs. The caller can save inspection JSON/overlays in a run subfolder of
the configured artifacts/mvp/results directory. Crops belong to the separate
cropping step and configured artifacts/mvp/crops location. The detector internals
are implemented and have 69 passing unit cases using controlled model substitutes;
real checkpoint initialization has a user-reported success; image inference
has produced one user-reported real-image result. User and agent inspected the
saved overlay and confirmed the box encloses the plate on that 480 x 480 image;
real wide/tall coordinate inspection and cropping remain pending D1.4.
The CLI remains a placeholder. The three-question interview and user-written
interface review are complete. The coordinate example required
guidance on separate x/y scale factors. Original-size postprocessing must happen
exactly once; returned pixel boxes must not be scaled a second time.

### Detector skeleton contract — reviewed 2026-11-10

Keep the agreed two-method interface. Initialization checks detector resources,
constructs the model/postprocessor, and restores EMA weights once. Prediction
accepts a usable Pillow image, preserves original pixels, explicitly runs the
preprocessing/forward/postprocessing sequence, and returns Python-valued Detection
objects. The architecture YAML alone does not execute these image-processing steps.

Use score > config.score_threshold, matching the existing validation script;
an empty filtered result is valid. Boxes are floating-point original-image
coordinates; rounding/clamping remains the separate crop responsibility.
Standard exceptions: FileNotFoundError for missing detector files,
ValueError for invalid settings/image input, RuntimeError for initialization,
checkpoint/device compatibility or inference/postprocessing failures. The user's
skeleton matches this contract after correcting the class-name capitalization,
predict docstring indentation and minor spelling/spacing. The corrected class
imported successfully before internal implementation. The two public method
signatures and documented error/threshold contracts are preserved in the
implemented module. No CLI is implemented yet.

### Detector internals — implemented, runtime verification pending

Initialization validates DetectorConfig values and input files, imports the
existing vendored backend, requires a single class without COCO remapping, turns
off separate PResNet backbone pretraining, and restores ema.module with strict
state-dictionary matching. Model and postprocessor move to the selected device
and evaluation mode. A private helper locates the existing vendored source tree;
it follows the validation script's import-root convention and checks module origin.

The preprocessing Compose is built from a copy of the merged validation-transform
settings, without accessing cfg.val_dataloader or any annotation/image dataset.
For the current YAML, those transforms resize to 640 x 640 and produce scaled
float32 RGB tensors. Prediction checks the resulting tensor against the model's
configured eval_spatial_size, runs under torch.inference_mode, and passes
[[original_width, original_height]] to the existing postprocessor exactly once.

Prediction uses a separate RGB copy, checks output structure and finite/ordered
boxes, valid single-class labels and score ranges, filters with score > threshold,
and returns existing Detection objects containing Python floats. Out-of-image
box bounds remain for the separate crop step to clamp. Exceptions retain their
causes; an empty successful prediction is distinct from a technical failure.

Unit tests exercise the constructor and predict method with a tiny PyTorch model,
controlled YAML/transform/postprocessor substitutes and a mocked torch.load.
They establish the wrapper's EMA selection/strict matching, settings/files/errors,
model reuse/evaluation/inference mode, preprocessing result contract, original
width/height handoff, score filtering and Python-valued results. Synthetic wide/tall
cases verify argument order and that returned original-pixel boxes are not scaled
again. They do not verify the actual vendored transform/postprocessor behavior.

Final scoped run: 149 passed in 1.99s (80 configuration and 69 detector cases).
After those unit tests, the user ran the supplied default-config constructor
sequence and reported: "Detector initialized; EMA checkpoint loaded successfully."
This supplies user-run initialization evidence for checkpoint restoration and
backend construction. The agent has not independently rerun that initialization.
The user then ran predict on the supplied 480 x 480 development image without
annotations and reported one Detection with finite ordered coordinates and a
score of 0.8843116760253906. This is separate real-path execution evidence, not
an agent rerun or quality benchmark. The saved overlay was then opened by the
agent: it is 480 x 480 RGB, the box encloses the visible plate, and the label
shows 0.884 consistently with the reported score. This completes the first-image
smoke inspection; non-square coordinate inspection is pending D1.4. No actual
weights or annotation datasets were loaded in the earlier
unit-test run; the later constructor/prediction reports are separate evidence.

The detector's [independent-review handoff](reviews/detector-handoff.md) is ready.
No external review outcome is recorded. Source code/tests were not changed or
rerun for the overlay inspection; no broader accuracy measurement is claimed.

## Cropping boundary — agreed 2026-11-11

Implement one public crop function in the existing pipeline.py module; full
detection/OCR orchestration remains later work. No new result schema is needed.

```python
def crop_plate(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    padding_pixels: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]]: ...
```

Inputs are the original Pillow image, one original-pixel detector box and an
explicit nonnegative integer padding value. The caller passes the configured
crop_padding_pixels; the function does not load configuration itself. Return
the cropped image and exact integer xyxy bounds for later PlateResult use.

Require finite ordered coordinates and a usable image. Reject boxes with no
positive-area overlap with the original image before padding, even if padding
could reach into the image. Expand by padding, floor left/top, ceil right/bottom,
then clamp to image dimensions and require nonempty final bounds. Left/top are
inclusive; right/bottom are exclusive, matching schemas.py.

Invalid/unusable inputs raise descriptive ValueError; the caller decides failure
handling. Extract original-resolution pixels and preserve the source image;
display, file saving and OCR remain caller responsibilities. With zero padding,
the previously reported box rounds to (287, 223, 393, 333), a 106 x 110 crop.
That initially expected arithmetic is now verified by real extraction: the saved
106 x 110 RGB crop matches the source pixels exactly and has been visually inspected.

The user accepted the proposal with "Agree, let's go". The three-question crop
interview and user-written scaffold review are complete. A docstring/body
indentation mismatch was corrected and AST parsing passed. After an exact
implementation-scope list, the user replied "approve". Internal implementation
was implemented, and the user subsequently requested annotations mapping every
approved implementation item to its code, followed by tests and real-output checks.
Day 1 continues as a work session on this date; Day 2 has not started.

### Crop internals — implemented and tested 2026-11-11

The function requires a nonempty Pillow image, a four-item tuple of finite Python
int/float coordinates (excluding bool), ordered positive-area bounds, and
nonnegative integer padding (excluding bool). It rejects an unpadded box with no
positive-area overlap, calculates outward-rounded padded bounds, clamps each axis
against its own image dimension and requires a nonempty region. Pillow crop is
used to return a separate image and exact integer bounds while preserving source
mode/pixel values. Expected Pillow read/crop failures become ValueError with cause.

Because padding p is an integer, floor(x - p) = floor(x) - p and
ceil(x + p) = ceil(x) + p. The implementation uses these equivalent integer
operations to avoid floating-point overflow/precision loss during padding;
the approved outward-padding/rounding result is preserved. Python integers are
finite without conversion; only float coordinates need an isfinite check.

Numbered comments above the relevant blocks map the eight approved implementation
items to the code. There were no functional changes during test development.
The 91 crop cases verify explicit bounds, coordinate-coded original pixels,
source/crop independence, modes/palette preservation, partial/outside/reversed/
non-finite boxes, padding (including very large integers) and readable errors.
The final scoped suite passes 240 cases in 2.17s; Ruff passes.

A separate verification script applied the function to the user's recorded
detection on the real image291 source. Bounds are (287, 223, 393, 333), with a
106 x 110 RGB crop. Every crop pixel was compared against the original region;
source pixels/mode and source-file hash remained unchanged, and a saved PNG
round trip was lossless. The agent opened the crop and confirmed the visible
plate is contained. Evidence is under artifacts/mvp/results/crop-check-t1fcbitw/.
No model was rerun; this uses the earlier user-run detection on the same image.

Afterward, the user ran the selected wide image (2048 x 1536), reported crop
bounds (374, 330, 1818, 1213), size 1444 x 883, and confirmed full-plate coverage.
This is user-observed smoke evidence; no saved wide output was agent-inspected.
The user then supplied tall bounds (157, 248, 224, 285), size 67 x 37, and confirmed
visual coverage. The agent subsequently reran both selected inputs with the real
CPU model and obtained the same bounds, saved both crops/overlays and directly
inspected full-plate coverage. D1.4 is now verified.
Full pipeline orchestration remains later work. The cropping independent-review
[handoff](reviews/cropping-handoff.md) is prepared; no external review is recorded.

### Manual runner and final Day 1 verification — 2026-11-11

The user developed a guarded manual main in pipeline.py through guided practice.
At their request to finish saving/verifying both cases, it was refined to retain
main() with no required arguments, process the two preselected source paths with
one detector instance, close source/crop images, pass configured padding, skip
empty detections, save the first crop and result.json in a fresh folder, and show
the crop on normal manual invocation. This is a two-example inspection runner;
the standalone prediction CLI and full detection/OCR orchestration remain pending.
The public crop/detector/config interfaces were preserved.

The agent executed this actual main with only Image.Image.show suppressed to
avoid desktop popups. Model loading, forward passes, preprocessing/postprocessing,
cropping and file writes were real. The verification script compared saved crops
against independent source byte-row slices, checked original-size bounds, selected
source hashes and unchanged source files, then reused draw_prediction to produce
overlays. Native crops and original-size overlays were opened by the agent.

Evidence: artifacts/mvp/results/non-square-verification-z10zeeuq/{report.json,
stdout.txt,verify_non_square.py}; wide-check-4811d7a2/{crop.png,overlay.png,result.json}
and tall-check-d5a0d005/{crop.png,overlay.png,result.json}. Both cases match the
user's earlier bounds. The checkpoint hash again matches the baseline manifest.
Final scoped suite: 240 passed in 1.74s. Focused Ruff passes; intentional ValueError
contracts have local TRY004 explanations rather than changing approved exceptions.
All technical Day 1 acceptance criteria are verified; independent review and
day-completion agreement remain pending. No general accuracy/latency claim is made.

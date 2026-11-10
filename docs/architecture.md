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

# OCR selection for the MVP

Status: proposed; requires a local crop benchmark.

The MVP extracts Latin registration letters and digits. Burmese recognition is
not required. PaddleOCR with English recognition is the initial candidate.

Before selecting a model, evaluate 20–50 representative plate crops, including
small, blurred, angled, and multiline examples. Compare full-plate exact matches,
character errors, and CPU latency. Record the exact model and package versions.

Recognition-only inference may suit clean single-line crops; multiline plates may
require text detection and reading-order handling inside the crop. Choose based
on observed failures. Retain raw OCR output and avoid unvalidated substitutions
such as changing every O to 0.

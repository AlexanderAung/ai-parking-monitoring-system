# MVP evaluation protocol (planned)

This protocol covers plate recognition. Existing detector results remain in the
baseline report and do not establish end-to-end OCR accuracy.

## Data

Use representative images with verified plate boxes and text. Record source IDs,
plate transcriptions, legibility, and development/test split membership in
data/mvp/labels.jsonl. Define the record schema before labeling. Preserve raw
transcriptions and document normalization rules. Include multi-plate and no-plate
images. Keep related frames and the same capture session in one split.

Tune thresholds and preprocessing on development data; reserve test data for
final evaluation. Report dataset counts and supported plate formats.

## Measurements

- OCR on manual crops: isolate recognition quality.
- OCR on predicted crops: inspect localization effects.
- Full-image pipeline: match predictions to ground truth one-to-one by box overlap,
  with the matching IoU threshold declared before evaluation. Count missed plates
  as failures and report unmatched detections separately.
- Full-plate exact-match rate and character error rate, with normalization stated.
- Accepted-read accuracy and coverage, with thresholds and denominators stated.
- Warm end-to-end latency and separate model startup time, on named hardware.

Keep unreadable ground truth separate from readable-plate text metrics and report
its count and system outcomes. Preserve no-plate and unreadable output examples.

## Evidence

Record checkpoint hash, OCR model/version, dependency lockfile, configuration,
dataset version, metric definitions, and representative failures under
docs/reports/mvp-001/. No MVP measurements have been made yet.

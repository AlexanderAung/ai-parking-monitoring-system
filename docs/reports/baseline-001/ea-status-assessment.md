# Detector error-analysis status assessment

Date: 2026-11-09. Inspected revision: `6e5dd82`.
Reference: [unchanged baseline report](report.md).
Scope: Phase 1 repository assessment; interview pending. Separate from daytime MVP implementation. No training, inference, dataset edits, or experiment execution performed.

## Verified evidence and status

- Baseline evaluation and size breakdown are complete as saved artifacts. Running `.venv/bin/python scripts/extract_coco_metrics.py` succeeded and reproduced AP50 0.9323, AP75 0.7714, and AP50:95 0.6565. Small/medium/large AP50:95 reproduced as 0.4189/0.7143/0.6686. This reads saved results; it is not fresh inference.
- The epoch-19 checkpoint, evaluated annotation JSON, and saved `eval.pth` exist and match their recorded SHA-256 hashes. The training notebook, log, resolved configuration, original annotation files, selected training annotation files, and audit summary/issues also match their manifest hashes. Checkpoint loading into the model was not rerun.
- Validation contains 2,048 image records and 2,195 boxes. Every referenced image file exists; decoding and image-content identity were not reverified. All boxes use category 0 and `iscrowd=0`; no orphan image references were found. Duplicate category definitions with ID 0 remain as documented.
- Under the saved evaluator's inclusive area ranges, counts are small 389, medium 1,527, large 280. One box has area exactly 9,216 and belongs to both medium and large evaluator ranges; these counts are not a disjoint partition.
- Local visualization directories contain 2,048 prediction images, 2,048 size-labelled ground-truth images, and 2,048 comparisons. Their presence does not establish exhaustive human review. Existing [review notes](../../../runs/baseline-001/eval-epoch19/visual-checks/review.md) document sampled annotation concerns and loose medium/small predictions.
- `scripts/predict_validation.py` restores EMA weights, uses the configured validation loader, and maps boxes to original image sizes. Its default drawing threshold is score > 0.6, distinct from AP evaluation.
- `scripts/export_validation_predictions_json.py` is implemented and reuses that inference pipeline. It writes image IDs, category IDs, XYWH boxes, and scores without an additional confidence cutoff. It is suitable in principle for EA, but was only inspected, not executed or validated against baseline metrics. It writes over its chosen output path and supplies no provenance manifest.
- No full validation prediction export was located, including searches of ignored local JSON/JSONL, CSV, pickle, NumPy, and Parquet artifacts. Saved `eval.pth` contains precision, score and recall arrays plus a `matched` dictionary, but no full detection-box table. It supports aggregate analysis; it is not a replacement for prediction records for visual error attribution.
- Image metadata contains original names and optional `user_tags`; 82 images have nonempty tags, including `False-positive` and `False-negative`. Their provenance is unverified and they must not be counted as baseline error labels. No systematic environmental-condition annotation scheme was found.

## Partial or missing work

Qualitative EA and annotation inspection are partial. Quantified FP/FN, localization and duplicate analysis, a reviewed case ledger, annotation-defect prevalence, condition-specific performance, and TIDE or equivalent decomposition were not found. `docs/templates/error-analysis.md` is an unfilled template; `docs/experiments/experiments.md` is empty.

Training evidence specifies RT-DETRv2 ResNet-18, one class, 10,000 training images, 640-square input, batch size 8, AdamW, EMA, and 50 epochs on one T4. Historical training YAML includes do not resolve as a ready-to-run local configuration. Exact training reproduction remains limited by unpinned historical code/environment and an unset training seed. The active local evaluation configuration is separate.

Training selection excluded multi-box images and relative-area extremes. This is a plausible coverage concern, not a demonstrated cause of small-plate failures. The existing structural training-data audit does not establish semantic annotation quality. Cross-split source independence, representative parking-camera evaluation, and independent test performance remain unverified.

## Interpretation and next gate

Small plates are the weakest measured size group. Annotation problems are documented in selected examples. Neither establishes the dominant error mechanism or the best training intervention. First resolve the scope of prior human review, target-domain priorities, metadata provenance, and available nightly time/compute through the interview. Then agree on a bounded analysis methodology before selecting one controlled experiment. Do not infer blur/lighting causes, overfitting, or benefits from higher resolution from the present evidence alone.

Initial Git status showed untracked `AGENTS.md`, `docs/learning/`, `docs/plans/`, and `docs/progress/`; these were preserved. This assessment is the only added project file. No baseline artifacts were overwritten.

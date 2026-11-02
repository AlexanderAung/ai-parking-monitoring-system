# AI Parking Monitoring System

An object-detection research project centered on fine-tuning RT-DETRv2 to locate license plates. The repository currently provides dataset auditing/selection utilities and a PyTorch RT-DETRv2 training/evaluation implementation. It does not yet implement a complete parking-lot monitoring application.

## Current pipeline

```text
Roboflow license-plate images + COCO annotations
        → audit and inspect data
        → filter/sample a baseline dataset
        → RT-DETRv2 license-plate detector
        → COCO bounding-box evaluation
```

The detector predicts license-plate bounding boxes. Parking-space occupancy, vehicle tracking, plate reading/OCR, a serving API, and a user interface are not present in the current code.

## Dataset

The source is the [Roboflow License Plate Recognition dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/13), represented as COCO detection annotations with one `license_plate` class. Dataset preparation utilities are in `src/parking_monitoring/data/`:

- `audit.py` checks image integrity, annotation validity, image/annotation pairing, duplicate files, and suspicious box sizes; it writes `issues.csv`, `summary.json`, and `distributions.png`.
- `select_dataset.py` excludes ERROR-level issues, orphan images, images with multiple boxes, and relative-box-area outliers (P5–P95). It leaves the raw data unchanged.
- `select_for_baseline.py` proportionally samples five relative-area strata with seed 42.
- `scripts/visualize_coco.py` draws COCO annotations for visual inspection.

The checked-in audit report describes 98,798 images, 102,844 annotations, 98,658 image/annotation pairs, 140 orphan images, zero audit errors, and 2,278 warnings. These are report figures for the audited source data, not model results.

Dataset state is not fully reproducible from the current scripts without configuration edits: the selection script currently targets 100 images in `data/baseline_test`, while `docs/decisions/002-dataset-selection.md` describes an approximately 86k candidate pool and a 10k `baseline_v1_0`. The training YAML also points to Kaggle-mounted paths. Treat those paths and dataset-version descriptions as environment-specific/documented state and verify them against the data you use.

## Model and configuration

The detector implementation is in [thrid-party/RT-DETR/rtdetrv2_pytorch](thrid-party/RT-DETR/rtdetrv2_pytorch/). The active local [evaluation config](configs/training/baseline-001-eval.yml) includes the [data config](configs/data/baseline-v1-eval.yml) and shared model/runtime settings. It uses one class, a ResNet-18 variant-d backbone, three decoder layers, 640 × 640 inputs, and EMA weights.

The original training run used batch size 8 and 50 epochs. Historical Kaggle configs and commands are retained as evidence, not as ready-to-run local training instructions. See the [technical appendix](docs/reports/baseline-001/technical-appendix.md#training-configuration-and-run-history).

## Recorded baseline

Checkpoint index 19 (20 completed epochs) was selected by validation AP and re-evaluated locally on 2,048 images containing 2,195 annotated boxes.

| Metric | Overall | Small | Medium | Large |
|---|---:|---:|---:|---:|
| AP50 | 0.9323 | 0.7363 | 0.9802 | 0.9487 |
| AP75 | 0.7714 | 0.4311 | 0.8620 | 0.7577 |
| AP50:95 | 0.6565 | 0.4189 | 0.7143 | 0.6686 |

AR@100 averaged over IoU 0.50–0.95 is 0.7357 (0.736 in the console summary). Size groups use COCO annotation-area ranges in original-image pixels squared.

The [baseline report](docs/reports/baseline-001/report.md) documents the category-ID mismatch and its resolution, annotation-quality limitations, and sampled prediction review. [Selected comparisons and dataset attribution](docs/reports/baseline-001/figures/qualitative/README.md) provide visual evidence. Reviewed examples showed no systematic box shifts or stretching; this is not proof for every image.

## Reproduce the metrics table

Use Python 3.12 or newer as specified by the project. From the repository root:

```bash
uv sync --locked
.venv/bin/python scripts/extract_coco_metrics.py
```

This extracts the table from the bundled evaluation output without running inference. The [evaluation record](runs/baseline-001/eval-epoch19/evaluation-config.md) lists the original local environment versions. Load only trusted saved evaluation files.

## Rerun evaluation

**Checkpoint and validation images are not bundled. Metrics can be extracted from the saved evaluation output; rerunning inference requires those external inputs.**

Supply the checkpoint at `runs/baseline-001/checkpoints/checkpoint0019.pth` and the 2,048 validation images at `data/raw/valid/`. Use the bundled `data/raw/annotations/kaggle_val_annotations.json`, which preserves the category mapping used during training-time evaluation. Verify the checkpoint and annotation hashes using the [artifact manifest](docs/reports/baseline-001/artifact-manifest.json). No checkpoint download or full image-content manifest is provided.

```bash
.venv/bin/python thrid-party/RT-DETR/rtdetrv2_pytorch/tools/train.py \
  -c configs/training/baseline-001-eval.yml \
  -r runs/baseline-001/checkpoints/checkpoint0019.pth \
  --test-only \
  --device cpu
```

This evaluates the configured validation split, not an independent test split, and overwrites evaluation output in the configured directory. [Prediction](scripts/predict_validation.py), [ground-truth visualization](scripts/visualize_coco_sizes.py), and [comparison](scripts/compare_validation_images.py) scripts can regenerate the full local visual review outputs once the required inputs are present.

## Limitations and next work

- Annotation defects and domain differences limit interpretation of the baseline scores. Structured annotation auditing and quantitative error analysis remain future work.
- Historical training-code/environment details are incomplete and training was not explicitly seeded. Recreating the metrics table is distinct from reproducing the original training run.
- Dataset selection scripts still require path and sampling-setting review before reuse.
- Independent test performance, representative CCTV evaluation, latency, tracking, OCR, and parking occupancy are not established by this baseline.

## Repository map

```text
thrid-party/RT-DETR/rtdetrv2_pytorch/  Detector implementation
configs/                            Project configs
src/parking_monitoring/data/         Dataset audit and selection utilities
scripts/                            Metric extraction and visualizations
docs/reports/baseline-001/           Baseline report and selected evidence
docs/templates/                     Templates, not measured experiment results
runs/baseline-001/eval-epoch19/       Saved evaluation, metrics, and config snapshots
```

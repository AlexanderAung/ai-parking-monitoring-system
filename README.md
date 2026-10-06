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

The main implementation is `RT-DETR/rtdetrv2_pytorch/`. It contains the RT-DETR/RT-DETRv2 model, COCO dataset/evaluator, training solver, and export utilities. The project-specific configuration is `RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/license_plate.yml`, which includes `configs/dataset/license_plate.yml` and shared model, optimizer, dataloader, and runtime YAML files.

The license-plate config sets one class and a ResNet-18 variant backbone (`PResNet`, depth 18), a three-level `HybridEncoder`, and an RT-DETRv2 transformer decoder with three layers. It requests pretrained backbone weights, trains for 50 epochs, and uses batch size 8. Evaluation uses the COCO bounding-box evaluator. The dataset config currently names train/validation directories and annotation files under `/kaggle/input/datasets/aungkhantbwar/baseline/`; update these paths for another environment.

## Setup, training, and evaluation

Use Python 3.12 or newer as specified in the root `pyproject.toml`. The root dependencies include PyTorch, torchvision, COCO evaluation packages, PyYAML, and TensorBoard; RT-DETRv2 also provides `requirements.txt`.

From the repository root, install the root project dependencies and enter the RT-DETRv2 implementation:

```bash
uv sync
cd RT-DETR/rtdetrv2_pytorch
```

After editing the dataset paths in `configs/dataset/license_plate.yml`, train with:

```bash
python tools/train.py -c configs/rtdetrv2/license_plate.yml --seed 0
```

Evaluate a saved checkpoint against the configured validation set with:

```bash
python tools/train.py -c configs/rtdetrv2/license_plate.yml \
  -r /path/to/checkpoint.pth --test-only
```

The script supports mixed precision (`--use-amp`), resume (`-r` without `--test-only`), and fine-tuning (`-t`). For multi-GPU training, use `torchrun` as documented in `RT-DETR/rtdetrv2_pytorch/README.md`. Checkpoints and logs are written under the configured output directory. The repository does not provide a project-specific end-to-end inference command for the parking application; the upstream RT-DETRv2 tools include export utilities.

## Recorded baseline

`docs/experiments/baseline_error_analysis.md` records a baseline evaluation using the best checkpoint at epoch 19 on a validation set of 2,048 images:

| Metric | Score |
| --- | ---: |
| COCO mAP@[0.50:0.95] | 0.656 |
| AP50 | 0.932 |
| AP75 | 0.771 |
| AP (small / medium / large) | 0.389 / 0.771 / 0.681 |

These are repository-recorded values; the checkpoint and complete training logs are not included here, so the run cannot be independently reproduced from this repository alone.

## Limitations and next components

- The implemented task is license-plate detection, not parking-space occupancy monitoring.
- Dataset paths are machine-specific; the preparation scripts use hard-coded relative paths and sampling defaults that differ from the older dataset design note.
- The experiment record has metrics but no saved checkpoint or detailed error analysis in the repository.
- No explicit implementation or finalized roadmap for downstream parking features was found. Natural follow-on work—still unimplemented—is to connect detections to video/camera input, add parking-space occupancy logic, and define any required plate OCR or monitoring interface.

## Repository map

```text
RT-DETR/rtdetrv2_pytorch/        Main detector, configs, train/eval/export tools
src/parking_monitoring/data/     Dataset audit and selection utilities
scripts/                         COCO annotation visualization helper
docs/                            Dataset decisions and baseline experiment notes
```

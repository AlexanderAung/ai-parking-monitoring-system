| Item | Record |
|---|---|
| Checkpoint | a48b7e760bc273ed8ba52a6f749b0944d2e420e2393d71917fd01df99030d871  runs/baseline-001/checkpoints/checkpoint0019.pth |
| Validation data | fc3b9de67de3b999ddb6ae66e86eec3ff12ed4e2763de977780c751457d11f0d  data/raw/annotations/kaggle_val_annotations.json|
| Class mapping | One detection class: license plate. num_classes: 1. Original annotation category ID 1 was converted to 0 in the Kaggle annotations. Model predictions and evaluation annotations both use ID 0. remap_mscoco_category: False. The evaluated Kaggle annotation file retains two category definitions sharing ID 0|
| Model/input | RT-DETRv2 with a ResNet-18 variant-d backbone and three decoder layers. Validation images are directly resized to 640 × 640, converted to float32, and pixel values scaled from 0–255 to 0–1. No letterboxing or channel mean/std normalization is configured.|
| Runtime | Evaluation ran locally on CPU using the checkpoint’s EMA weights. Mixed precision was disabled (use_amp: False); inference used float32. |
| Evaluation | Bounding-box evaluation on all 2,048 validation images, matching category IDs. IoU thresholds: 0.50–0.95 in increments of 0.05. AP uses 101 recall points from 0 to 1. Detection limits: 1, 10, and 100 per image; the reported AP table uses 100.|
| Postprocessing | Class scores are calculated using sigmoid. The top 300 predictions per image are retained, with no fixed confidence-score cutoff and no non-maximum suppression (NMS). Predicted boxes are converted from normalized center coordinates to corner coordinates and scaled to the original image dimensions. COCO evaluation then applies its detection limits; the AP table uses 100.|
| Reproduction | Evaluation command and environment versions are recorded below. Configuration snapshots are retained with the results. Metrics are extracted using scripts/extract_coco_metrics.py. |


| Group | Annotation area, pixels² |
|---|---:|
| All | 0–10,000,000,000 |
| Small | 0–1,024 |
| Medium | 1,024–9,216 |
| Large | 9,216–10,000,000,000 |

| Component | Version |
|---|---|
| Python | 3.12.9 |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| faster-coco-eval — active evaluator | 1.8.0 |
| pycocotools | 2.0.11 |
| numpy | 2.5.3 |
| pillow | 12.3.0 |
| PyYAML | 6.0.3 |
~~~
.venv/bin/python thrid-party/RT-DETR/rtdetrv2_pytorch/tools/train.py \
  -c configs/training/baseline-001-eval.yml \
  -r runs/baseline-001/checkpoints/checkpoint0019.pth \
  --test-only \
  --device cpu
~~~


The command runs from the repository root and overwrites the configured evaluation output. Checkpoint and validation images are not bundled. Metrics can be extracted from the saved evaluation output; rerunning inference requires those external inputs. See the [reproduction instructions](../../../docs/reports/baseline-001/report.md#reproduce-the-metrics-table).

The active config now uses `runs/baseline-001/eval-epoch19/tensorboard` for `summary_dir`. The frozen run snapshot preserves the original `runs/baseline-001/evaluation-local/tensorboard`. This output-location-only correction does not change model, preprocessing, or evaluation settings. Frozen snapshots have not been edited.

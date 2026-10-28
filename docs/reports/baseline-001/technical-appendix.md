# Baseline-001: Technical Appendix

[← Read the portfolio report](report.md)

This appendix preserves the technical record behind the license-plate detection baseline: dataset preparation, training settings, evaluation details, and reproduction evidence. Results describe validation performance from one training run.

## Contents

- [Experiment metadata](#experiment-metadata)
- [Experiment scope and model choice](#experiment-scope-and-model-choice)
- [Dataset and audit details](#dataset-and-audit-details)
- [Model specification](#model-specification)
- [Training configuration and run history](#training-configuration-and-run-history)
- [Evaluation protocol](#evaluation-protocol)
- [Full validation results](#full-validation-results)
- [Prediction review status](#prediction-review-status)
- [Reproduction and artifacts](#reproduction-and-artifacts)
- [References](#references)

## Experiment metadata

| Field | Value |
|---|---|
| Author | Alexander Aung Khant Bwar |
| Experiment ID | `baseline-001` |
| Training date | September 26, 2026 |
| Report prepared | October 27, 2026 |
| Project purpose | Computer vision portfolio project |
| Task | Single-class license-plate bounding-box detection |
| Model | RT-DETRv2 with a ResNet-18 variant-d (`PResNet`) backbone |
| Dataset | Roboflow License Plate Recognition v13; derived training subset `baseline_v1_0` |
| Platform | Kaggle, one NVIDIA T4 GPU |
| Primary evidence | [Notebook](../../../notebooks/baseline-training.ipynb), [epoch log](log.txt), [configuration snapshot](resolved-training-config.json), and [checkpoint](../../../runs/baseline-001/checkpoints/checkpoint0019.pth) |

## Experiment scope and model choice

### Experimental question

What license-plate detection performance does the selected RT-DETRv2 configuration achieve on the prepared dataset using the available Kaggle T4 resources, and what limitations should guide subsequent experiments?

### Model-selection rationale

RT-DETRv2 was selected for its focus on real-time object detection, aligning with the planned use of CCTV video in the parking project.
This is a suitability rationale; actual processing speed on the intended deployment hardware has not been established. The architecture and its real-time focus are described in the [RT-DETRv2 paper](https://arxiv.org/abs/2407.17140).

The backbone was changed to ResNet-18 after a larger configuration encountered a CUDA out-of-memory error on the Kaggle T4 GPU. The exact failed configuration and its full settings were not recorded. This supports the smaller configuration as a feasible starting point under the attempted settings, without establishing that every larger configuration would be infeasible.

### Inputs, outputs, and intended use

- **Current experiment input:** still images, resized to 640 × 640 for the model.
- **Current experiment output:** bounding boxes, detection scores, and a license-plate class prediction.
- **Intended future pipeline:** CCTV frame → plate detection → plate crop → OCR → plate text.
- **Outside this experiment:** OCR evaluation, tracking, entry/exit recording, occupancy estimation, video integration, and deployment benchmarking.

### Acceptance criteria

No numerical accuracy or speed threshold was predefined. The purpose was to establish the starting performance reference for subsequent experiments. Checkpoint selection used the highest validation AP@[0.50:0.95].

## Dataset and audit details

### Source and annotation format

The source is [Roboflow License Plate Recognition, version 13](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/13), exported in COCO format. Dataset usage and redistribution license details have not been verified. Dataset, pretrained-weight, and code licenses should be recorded before redistribution; later demonstrations should account for the privacy of plate imagery. Camera, geography, lighting, and vehicle coverage were not systematically documented.

The source training data was already augmented, including geometric and appearance transformations. These are source-dataset augmentations, separate from the augmentations applied during model training. The original train/validation/test partition was preserved. Independence of augmented source images across splits has not been verified.

### Dataset sizes

| Dataset | Images | Bounding boxes | Role |
|---|---:|---:|---|
| Original training split | 98,798 | 102,844 | Source for training-data preparation |
| Filtered candidate pool | 86,085 | 86,085 | Single-plate candidates after area filtering |
| Baseline training subset | 10,000 | 10,000 | Model training |
| Original validation split | 2,048 | 2,195 | Per-epoch evaluation and checkpoint selection |
| Original test split | 1,020 | 1,085 | Available locally; no test evaluation recorded |

The filtered pool and baseline subset are stored under `data/processed/selected_v0_1/` and `data/processed/baseline_v1_0/`. The baseline image IDs are a subset of the local original training image IDs.

### Automated audit

The project's audit script was run before selecting the training subset. The audit measures and reports findings without modifying the raw data. It checks image decoding and dimensions, image/annotation pairing, basic class and box validity, relative box sizes, and exact file duplicates within the configured image folder.

| Audit finding | Count |
|---|---:|
| Error-level findings | 0 |
| Warning-level findings | 2,278 |
| Tiny-box warnings, relative area below 0.0005 | 593 |
| Very-large-box warnings, relative area above 0.90 | 1,172 |
| Out-of-bounds box warnings | 513 |
| Images without associated annotations | 140 |
| Byte-identical duplicate files detected in the audited folder | 0 |

Warning counts are findings and do not necessarily represent distinct images. Zero structural errors does not establish annotation correctness. Small/large boxes can be legitimate examples; the warning thresholds are review heuristics.

Audit limitations include the lack of perceptual duplicate checks and cross-split checks. The current COCO audit code hardcodes the orphan-annotation count to zero, so that output is not evidence that all annotation references were validated.

Full evidence: [summary](../../../artifacts/data-audit/roboflow-v13-train/audit_report/summary.json), [issues](../../../artifacts/data-audit/roboflow-v13-train/audit_report/issues.csv), and [distribution plots](../../../artifacts/data-audit/roboflow-v13-train/audit_report/distributions.png).

### Filtering and sampling

1. Exclude images associated with error-level audit findings.
2. Exclude images without an associated bounding-box annotation.
3. Exclude images containing multiple bounding boxes.
4. Calculate each remaining box's relative area: `(box_width × box_height) / (image_width × image_height)`.
5. Retain images between the remaining population's 5th and 95th relative-area percentiles, inclusive.
6. Divide the candidate pool into five equal-width relative-area bins and sample proportionally using seed 42.
7. Save the 10,000-image training subset separately from the raw data.

| Stage | Images retained | Boxes retained |
|---|---:|---:|
| Original training data | 98,798 | 102,844 |
| After removing 140 unannotated images and 3,007 multiple-box images | 95,651 | 95,651 |
| After inclusive P5–P95 relative-area filtering | 86,085 | 86,085 |
| Proportionally sampled baseline | 10,000 | 10,000 |

Reconstructed single-box thresholds are **P5 = 0.0046979818** and **P95 = 0.1171004911**. They differ from the raw audit percentiles because they are calculated after the single-box restriction.

The sampling script's current 100-image setting was used for inspection, and must not be mistaken for the completed run's 10,000-image training size.

### Visual inspection

Visually inspected sampled images and observed very small and very large plate instances. No manual annotation corrections were made. No per-image review record or quantified annotation-error estimate was recorded. Observations of plate sizes concern ground-truth data inspection, not model prediction errors.

### Class mapping and split independence

Local annotations use category ID 1 for plate boxes, and the category metadata also contains an unused ID 0 entry. The saved run configuration specifies one model class with `remap_mscoco_category: false`. 

Cross-split exact/perceptual duplicate checks and grouping by source image, camera, vehicle, or video were not recorded. Split independence remains unverified.

## Model specification

| Component | Recorded configuration |
|---|---|
| Architecture | `RTDETR` using `RTDETRTransformerv2` |
| Backbone | `PResNet`, depth 18, variant `d` |
| Backbone feature channels | 128, 256, 512 |
| Encoder | `HybridEncoder`, hidden dimension 256, one encoder layer |
| Feature strides | 8, 16, 32 |
| Decoder | Three layers, hidden dimension 256 |
| Object queries | 300 |
| Denoising queries | 100 |
| Model classes | 1 |
| Frozen backbone stages | None configured (`freeze_at: -1`) |
| Frozen normalization | Disabled (`freeze_norm: false`) |
| Trainable parameters | 20,083,028, as recorded in the epoch log |

The initial fine-tuning command references `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`. The resumed run explicitly restores a project checkpoint, including model, EMA, optimizer, and scheduler states. The log also records loading the ResNet-18 variant-d pretrained backbone during model construction before checkpoint restoration.

The complete printed configuration is preserved in [resolved-training-config.json](resolved-training-config.json). Which individual hyperparameters were deliberately tuned versus retained from defaults was not separately recorded.

## Training configuration and run history

### Environment

Training used one Kaggle T4 GPU. Two-GPU execution failed; the run notes associate this with a community-reported bug. The larger-backbone CUDA out-of-memory failure is a feasibility attempt.

### Configuration

| Setting | Recorded value |
|---|---|
| Planned / completed epochs | 50 / 50 |
| Training batch size | 8 on one GPU |
| Validation batch size | 16 |
| Input / evaluation resolution | 640 × 640 |
| Optimizer | AdamW |
| Default learning rate | 0.0001 |
| Selected backbone parameter-group learning rate | 0.00001 |
| Betas / epsilon | (0.9, 0.999) / 1e-8 |
| Default weight decay | 0.0001 |
| Selected encoder/decoder normalization-group weight decay | 0 |
| Scheduler | MultiStepLR, milestone 1000, gamma 0.1 |
| Warmup | Linear, 2,000 optimizer steps |
| Mixed precision | Disabled (`use_amp: false`) |
| Gradient clipping | Maximum norm 0.1 |
| EMA | Enabled, decay 0.9999, warmup 2,000 |
| Data-loader workers | 4 per loader |
| Training shuffle / drop last | Enabled / enabled |
| Validation shuffle / drop last | Disabled / disabled |
| Gradient accumulation | No accumulation setting recorded |
| Training random seed | Not explicitly set (`seed: null`) |
| Dataset-sampling seed | 42; separate from training randomness |
| Checkpoint frequency | Every epoch |
| Best-checkpoint criterion | Highest validation AP@[0.50:0.95] |
| Early stopping | No early stop; all 50 epochs completed |

The scheduler milestone lies outside the 50-epoch run, so no milestone decay is scheduled during this experiment. The saved checkpoint corroborates the optimizer group learning rates, weight decay, scheduler milestone, and warmup duration.

### Preprocessing and augmentation

Training applies photometric distortion (`p=0.5`), zoom-out with zero fill, IoU-based cropping (`p=0.8`), bounding-box sanitization with minimum size 1, horizontal flipping, direct resize to 640 × 640, and another sanitization pass. Images are converted to float32 and scaled, and boxes are converted to normalized center/width/height format. The configuration does not list channel-wise mean/std normalization or aspect-ratio-preserving letterboxing.

The augmentation policy stops photometric distortion, zoom-out, and IoU cropping at epoch index 40. It does not list horizontal flipping among the stopped operations. Batch multi-scale resizing is disabled (`scales: null`). Probabilities or parameters omitted from the printed configuration depend on implementation defaults and are not specified in this record.

Validation applies resize to 640 × 640 followed by float32 conversion and scaling. It has no random augmentations in the recorded transform list.

### Losses and run history

The criterion uses varifocal classification loss, bounding-box regression, and generalized-IoU loss, with configured weights 1, 5, and 2. Auxiliary decoder, denoising, and encoder losses are also present in the training log. Hungarian matching uses classification, box, and GIoU costs of 2, 5, and 2.

The notebook contains an initial fine-tuning command and a later resume command from `checkpoint0004.pth`. Its retained console output shows the resumed execution starting at epoch index 5. The separate epoch log contains entries 0–49, establishing the complete recorded 50-epoch history. The interruption reason and total wall-clock training duration were not retained. The saved epoch-19 checkpoint records the timestamp `2026-09-26T08:20:31.328762`; its timezone is unspecified.

## Evaluation protocol

The saved configuration evaluates bounding boxes using `CocoEvaluator` against `valid/valid` and `val_annotations.json`. The epoch log field `test_coco_eval_bbox` is an implementation name for this validation evaluation. It does not establish use of the separate test split.

Checkpoint selection maximizes the first COCO bounding-box statistic, AP@[0.50:0.95]. EMA is enabled; the retained EMA checkpoint state and the available solver's evaluation path support evaluation of the EMA model. Exact historical code/evaluator package versions were not captured.

The postprocessor retains the top 300 scored class/query predictions per image. The inspected implementation uses sigmoid scores without an NMS stage or a fixed confidence cutoff in its evaluation postprocessor. COCO evaluation summarizes AP across IoU thresholds 0.50–0.95 and reports AP50, AP75, and small/medium/large object categories. Standard COCO area ranges use box area in pixels, whereas dataset filtering used relative box area; these size categories are different concepts.

No fixed-threshold precision, F1, detector latency, FPS, or end-to-end OCR/video measurements were recorded for this run. Training iteration timings must not be interpreted as inference FPS.

## Full validation results

### Validation results

Scores use the **0–1 scale**. Table headings count completed epochs; the corresponding zero-based log indices are 19 and 49. AP is average precision; AR is average recall. Values are extracted from [log.txt](log.txt) and preserved in [metrics.json](metrics.json).

| Metric | Best: epoch 20 | Final: epoch 50 |
|---|---:|---:|
| AP@[0.50:0.95] | **0.6565** | 0.6374 |
| AP50 | 0.9323 | 0.9305 |
| AP75 | 0.7714 | 0.7593 |
| AP small | 0.4189 | 0.3886 |
| AP medium | 0.7143 | 0.6988 |
| AP large | 0.6686 | 0.6790 |
| AR, maximum 1 detection/image | 0.6799 | 0.6731 |
| AR, maximum 10 detections/image | 0.7233 | 0.7199 |
| AR, maximum 100 detections/image | 0.7357 | 0.7282 |

The notebook and logs contain no separate test evaluation.

### Training behavior

See the [training curve in the main report](report.md#results-and-interpretation).

The figure numbers completed epochs from 1 to 50; the selected checkpoint's logged index 19 therefore appears at completed epoch 20. Total training loss decreases from 14.4176 at logged epoch 0 to 3.4114 at logged epoch 49. The final validation AP is 0.0190 below the best value. Continued training did not improve the selected primary metric over its earlier maximum. These measurements alone do not establish the cause of the subsequent variation or prove overfitting. Validation loss was not included in the epoch log used for this figure.

### Conditions and variability

At the selected checkpoint, small-object AP is lower than medium- and large-object AP. This identifies a useful evaluation direction, but does not establish a specific error mechanism. Lighting, blur, occlusion, camera angle, and geography were not evaluated as annotated subgroups in this experiment; their subset counts and metrics are unavailable.

One completed baseline training history is documented. No independent-seed repetition, confidence interval, or run-to-run variability estimate was recorded. The unseeded training setup also limits exact repeatability.

## Prediction review status

Ground-truth sample inspection identified very small and large plate instances, and no labels were corrected. This is a dataset observation rather than an analysis of model mistakes. No reviewed prediction set or quantified false-positive, missed-detection, or localization-error breakdown was recorded.

Prediction-level analysis is pending in [error_analysis.md](error_analysis.md). Explanations of failure causes need reviewed examples; aggregate metrics identify questions to investigate. The [main report](report.md#limitations-and-next-experiment) sets out the proposed next steps.

## Reproduction and artifacts

### Current reproduction status

The notebook, full epoch log, local dataset annotations, selected checkpoint, and extracted configuration support inspection of the reported experiment. Exact regeneration is incomplete because the historical training revision, full environment, uploaded label conversion, and training seed are not established.

The notebook clones the repository without pinning a commit. Historical commands use an `RT-DETR/` directory, whereas the inspected repository stores that implementation under `thrid-party/RT-DETR/`. The copied configuration files under `configs/` also contain relative includes referring to their original upstream directory layout. They must not be treated as a verified runnable reproduction setup in their current location.

### Dataset reconstruction

The recorded sequence is `audit.py` → `select_dataset.py` → `select_for_baseline.py`. Before rerunning, configure input/output paths for each script's working directory, set the training sample target to 10,000 rather than the current inspection target of 100, preserve five bins and seed 42, and verify the selected annotation manifest. The existing script defaults use differing relative directory layouts, so no unverified one-command reconstruction is claimed here.

### Original notebook training commands

**Historical commands:** these are transcribed from the notebook and preserve its Kaggle layout. They document the run and have not been verified as runnable commands in the current local checkout.

Initial fine-tuning:

```bash
python /kaggle/working/ai-parking-monitoring-system/RT-DETR/rtdetrv2_pytorch/tools/train.py \
  -c /kaggle/working/ai-parking-monitoring-system/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/license_plate.yml \
  -t /kaggle/input/datasets/aungkhantbwar/weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --device cuda
```

Recorded resume:

```bash
python /kaggle/working/ai-parking-monitoring-system/RT-DETR/rtdetrv2_pytorch/tools/train.py \
  -c /kaggle/working/ai-parking-monitoring-system/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/license_plate.yml \
  -r /kaggle/working/output/license_plate/checkpoint0004.pth \
  --device cuda
```

No standalone test-evaluation command appears in the notebook. To establish an independent test result, explicitly configure a loader for the original test images and annotations, restore the frozen selected checkpoint, record preprocessing/postprocessing, and save the evaluation output. In this implementation, `--test-only` still evaluates the configured validation loader; the flag alone does not select the test dataset.

### Artifact inventory

Relative links below point to files available in the local workspace. Local availability does not establish that a file is committed or downloadable from the public repository. The Kaggle reference is an external notebook link.

| Artifact | Location |
|---|---|
| Source notebook | [baseline-training.ipynb](../../../notebooks/baseline-training.ipynb) |
| Public notebook reference | [Kaggle](https://www.kaggle.com/code/aungkhantbwar/rt-detr-baseline-training) |
| Full 50-epoch metrics log | [log.txt](log.txt) |
| Printed runtime/configuration snapshot | [resolved-training-config.json](resolved-training-config.json) |
| Extracted best/final metrics | [metrics.json](metrics.json) |
| Local evidence sizes and SHA-256 hashes | [artifact-manifest.json](artifact-manifest.json) |
| Selected epoch-19 checkpoint | [checkpoint0019.pth](../../../runs/baseline-001/checkpoints/checkpoint0019.pth) |
| Training curves | [training-curves.png](figures/training-curves.png) |
| Baseline training annotations | [annotations.json](../../../data/processed/baseline_v1_0/annotations.json) |
| Candidate annotations | [annotations.json](../../../data/processed/selected_v0_1/annotations.json) |
| Original validation annotations | [valid_annotations.json](../../../data/raw/annotations/valid_annotations.json) |
| Original test annotations | [test_annotations.json](../../../data/raw/annotations/test_annotations.json) |
| Dataset preparation note | [002-dataset-selection.md](../../decisions/002-dataset-selection.md) |

Hashes identify the inspected local artifacts, not all source-image contents or the Kaggle upload. The JSON configuration snapshot retains historical includes and paths and is not a directly executable training configuration.

## References

- Lv et al., [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140).
- [Official RT-DETR implementation and model configurations](https://github.com/lyuwenyu/RT-DETR).
- [Roboflow License Plate Recognition dataset, version 13](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/13).
- [Project Kaggle notebook](https://www.kaggle.com/code/aungkhantbwar/rt-detr-baseline-training).

[← Return to the portfolio report](report.md)

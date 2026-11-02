# RT-DETR License Plate Detection — Error Analysis Template, NOT REAL REPORT!


**Author:** Alexander Aung Khant Bwar
**Date:** 2026-10-31  
**Model:** RT-DETR  
**Dataset:** License Plate Validation Set  
**Status:** Analysis Complete

## Baseline Metrics
- mAP@[0.50:0.95]: 0.6565
- AP50: 0.9323
- AP75: 0.7714
- Small AP: 0.4189
- Medium AP: 0.7143
- Large AP: 0.6686

Source: the `test_coco_eval_bbox` values at epoch index 19 in [log.txt](log.txt).

## 1. Executive Summary

The RT-DETR model performs well on medium and large 
license plates but struggles with small license plates.

### Key Findings
- Overall AP50: 0.82
- Small-plate AP50: 0.39
- Main failure: Missed small license plates
- Secondary failure: Inaccurate bounding boxes
- Likely contributing factor: Insufficient spatial
  resolution for distant license plates

### Recommendation
Investigate higher inference resolution before
retraining or modifying the model architecture.

---

## 2. Objective

Identify why RT-DETR performs poorly on small license
plates and determine the most promising improvements.

Research questions:
1. Which detection errors occur most frequently?
2. Which errors have the greatest AP impact?
3. What conditions are associated with failures?
4. What evidence supports potential improvements?

---

## 3. Dataset and Evaluation Setup

| Item | Value |
|---|---|
| Validation images | 1,200 |
| Ground-truth plates | 1,600 |
| Small plates | 420 |
| Medium plates | 780 |
| Large plates | 400 |
| Baseline input size | 640 × 640 |
| Model | RT-DETR |

Size groups are based on original-image plate height:
- Small: < 24 pixels
- Medium: 24–63 pixels
- Large: ≥ 64 pixels

Evaluation uses IoU thresholds from 0.50 to 0.95,
with consistent annotation and prediction coordinates.

---

## 4. Baseline Model Performance

| Metric | Overall | Small | Medium | Large |
|---|---:|---:|---:|---:|
| AP50 | 0.82 | 0.39 | 0.84 | 0.94 |
| AP75 | 0.57 | 0.15 | 0.57 | 0.83 |
| AP50:95 | 0.52 | 0.18 | 0.51 | 0.71 |

### Observations

1. Small plates have substantially lower AP.
2. The large AP50-to-AP75 drop suggests localization
   difficulties, especially for small plates.
3. Aggregate metrics conceal poor small-object
   performance.

These metrics identify the problem but do not yet
establish its root cause.

---

## 5. Evaluation Sanity Checks

Completed checks:

- [x] Verified prediction coordinate format
- [x] Verified bounding-box scaling
- [x] Checked class-ID mapping
- [x] Checked ground-truth annotation format
- [x] Confirmed consistent IoU evaluation settings
- [x] Confirmed the model checkpoint
- [x] Verified confidence-filtering behavior

No evaluation implementation errors were identified.

---

## 6. Detection Error Classification

TIDE was used to analyze the six detection-error
categories.

### Estimated Error Impact

| Error Type | Oracle ΔAP50 |
|---|---:|
| Missed detections (MISS) | +0.13 |
| Localization (LOC) | +0.07 |
| Background (BKG) | +0.04 |
| Duplicate (DUP) | +0.01 |
| Classification (CLS) | +0.00 |
| Both (CLS + LOC) | +0.00 |

Note: These are hypothetical TIDE oracle improvements,
not guaranteed gains from real model changes. They
must not be added together.

### Interpretation

Missed detections have the largest estimated impact.

Localization errors are the second priority.

Because this is single-class detection, classification
errors have little relevance in this example.

---

## 7. Qualitative Failure Inspection

A random sample of 100 small ground-truth plates was
manually inspected at a confidence threshold of 0.30.

| Outcome | Count |
|---|---:|
| No nearby prediction | 51 |
| Poorly localized prediction | 29 |
| Correctly localized prediction | 20 |

These are manual audit groups, not formal TIDE counts.

### Example A — Missed Small Plate

**Image:** `examples/missed_001.jpg`

- Plate height: 12 pixels
- Ground-truth box exists
- No nearby detection above threshold
- Possible explanation: Insufficient visual detail

### Example B — Localization Failure

**Image:** `examples/localization_014.jpg`

- Plate height: 18 pixels
- Prediction confidence: 0.83
- IoU with ground truth: 0.36
- Possible explanation: Small coordinate errors
  have a large effect on IoU for tiny objects

### Example C — Background False Positive

**Image:** `examples/background_008.jpg`

- A vehicle grille was detected as a plate
- Confidence: 0.76
- No matching ground-truth plate

## 8. Root-Cause Hypotheses

| ID | Hypothesis | Priority |
|---|---|---|
| H1 | Input resolution is insufficient | High |
| H2 | Small plates are underrepresented | High |
| H3 | Annotation quality is inconsistent | Medium |
| H4 | Confidence filtering hides detections | Medium |
| H5 | Motion blur contributes to failures | Medium |

These are hypotheses, not confirmed causes.

---

## 9. Controlled Experiments

### Experiment 1 — Confidence Threshold

**Hypothesis:** Some small plates are predicted with
low confidence and removed by operational filtering.

**Method:**
- Use the same model and saved predictions
- Evaluate thresholds: 0.05, 0.15, 0.30
- Compare precision, recall, and false positives
- Keep dataset and IoU criteria fixed

**Illustrative result:**
Lower thresholds improved operating-point recall
but increased false positives.

**Conclusion:**
Filtering contributes to some missed detections,
but does not fully explain poor small-object AP.

### Experiment 2 — Inference Resolution

**Hypothesis:** Higher inference resolution improves
recognition and localization of small plates.

**Method:**
- Keep checkpoint and dataset fixed
- Compare input sizes of 640 and 960
- Preserve consistent coordinate transformations
- Measure small AP50 and inference latency

| Metric | 640 | 960 |
|---|---:|---:|
| Small AP50 | 0.39 | 0.53 |
| Inference latency | 20 ms | 33 ms |

**Conclusion:**
Higher resolution substantially improved small-plate
AP50 in this example, suggesting that resolution
contributes to the observed failures.

However, higher latency creates a deployment tradeoff.

---

## 10. Recommendations

### Priority 1 — Validate Input Resolution

Evaluate 640, 800, and 960 input resolutions using
the same validation set and checkpoint.

Select a resolution that balances detection quality
and deployment latency.

### Priority 2 — Audit Small-Plate Annotations

Inspect randomly sampled small plates for missing,
inaccurate, or inconsistent bounding boxes.

### Priority 3 — Analyze Training Data Distribution

Compare small-plate frequency and visual conditions
between training and validation datasets.

Do not initiate retraining until these investigations
provide sufficient evidence.

---

## 11. Limitations

- Error-impact estimates are counterfactual.
- Manual inspection covers only a sample.
- Correlation does not prove root cause.
- Additional experiments are necessary to confirm
  generalization beyond this validation dataset.
- Findings may depend on camera and image conditions.

---

## 12. Final Conclusion

RT-DETR's primary weakness is detecting small
license plates.

Error analysis identifies missed detections and
localization failures as the main problems.

The resolution experiment provides evidence that
limited spatial detail contributes to these errors.

The next engineering step is to validate resolution
tradeoffs and inspect annotation quality before
considering retraining.

---

## 13. Supporting Artifacts

- `metrics/baseline_metrics.json`
- `metrics/tide_results.json`
- `plots/ap_by_size.png`
- `plots/error_distribution.png`
- `plots/confidence_vs_recall.png`
- `examples/missed/`
- `examples/localization/`
- `examples/background/`
- `experiments/resolution_comparison.csv`

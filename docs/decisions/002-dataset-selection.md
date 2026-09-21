# Dataset selection 

# Baseline Dataset v1.0

## Purpose

Create a clean and reproducible **10,000-image baseline dataset** for the first RT-DETRv2 training experiment.

The baseline is intended to establish an initial model performance reference before experimenting with larger datasets or different data-selection strategies.

---

## Dataset Pipeline

The dataset is produced in three stages:

```text
Raw dataset
    │
    ▼
audit.py
    │
    ├── Detect data-quality problems
    └── Generate audit report
    │
    ▼
select_dataset.py
    │
    ├── Remove invalid/error images
    ├── Remove orphan images
    ├── Remove images with >1 bounding box
    └── Filter by bbox relative-area P5–P95
    │
    ▼
~86k candidate images
    │
    ▼
select_for_baseline.py
    │
    └── Stratified random sampling
    │
    ▼
10,000-image baseline dataset
```

---

## Stage 1 — Dataset Audit

`audit.py` was used to inspect the raw training dataset before selection.

The audit checks include:

* Image integrity and decodability
* Image dimensions and extreme aspect ratios
* Annotation validity
* Invalid class IDs
* Invalid bounding boxes
* Bounding boxes outside image boundaries
* Empty annotations
* Very small bounding boxes
* Suspiciously large bounding boxes
* Duplicate images
* Image/annotation pairing

The audit produces:

```text
audit_report/
├── issues.csv
├── summary.json
└── distributions.png
```

`issues.csv` contains individual data-quality issues, while `summary.json` provides aggregate dataset statistics.

Warnings are not automatically treated as reasons for removal. Only structural/data-integrity errors are excluded during dataset selection.

---

## Stage 2 — Candidate Dataset Selection

`select_dataset.py` creates a cleaned candidate dataset from the raw dataset.

The raw dataset is **never modified**.

The following images are excluded:

1. Images with `ERROR`-level audit issues.
2. Orphan images without a corresponding annotation.
3. Images containing more than one bounding box.

After structural filtering, bounding-box relative area is calculated:

```text
relative_bbox_area =
    bbox_width × bbox_height
    -------------------------
       image_width × image_height
```

The 5th and 95th percentiles of the relative-area distribution are then calculated.

Images outside this range are excluded.

This produces the candidate dataset:

```text
data/selected_v0_1/
├── images/
└── annotations.json
```

The purpose of this stage is to remove clear structural problems and extreme bounding-box cases while retaining a large candidate pool for later experiments.

---

## Stage 3 — Baseline Dataset Sampling

The candidate dataset contains approximately **86,000 images** after filtering.

`select_for_baseline.py` creates the baseline dataset by selecting **10,000 images** from this candidate pool.

### Sampling strategy

The sampling uses **proportional stratified random sampling based on bounding-box relative area**.

The candidate images are divided into five relative-area strata. Images are then randomly sampled from each stratum in proportion to its size.

A fixed random seed is used:

```text
random_seed = 42
```

This makes the selection reproducible. Running the same script against the same candidate dataset produces the same 10,000-image sample.

The final dataset is:

```text
data/baseline_v1_0/
├── images/
└── annotations.json
```

---

## Why 10,000 Images?

The first experiment is intended to establish a baseline rather than maximize final model performance.

Using 10,000 images provides a dataset large enough for an initial object-detection experiment while keeping training time low enough to allow multiple experiments within the available GPU budget.

Larger datasets can be evaluated after the baseline has been established.

---

## Dataset Selection Principles

The selection process follows these principles:

* **Keep the raw dataset immutable.**
* **Separate auditing from dataset selection.**
* **Do not automatically remove every audit warning.**
* **Remove structural data-quality errors.**
* **Use reproducible sampling rather than manual image selection.**
* **Keep dataset versions so experiments can be reproduced.**

The baseline therefore represents a controlled starting point for evaluating RT-DETRv2 performance.

---

## Dataset Version

| Dataset         | Purpose              | Approx. size |
| --------------- | -------------------- | -----------: |
| Raw             | Original dataset     |         ~99k |
| `selected_v0_1` | Clean candidate pool |         ~86k |
| `baseline_v1_0` | RT-DETRv2 baseline   |          10k |

---

## Reproducibility

The baseline dataset can be regenerated from the raw dataset using:

```text
audit.py
    ↓
select_dataset.py
    ↓
select_for_baseline.py
```

with the documented filtering criteria and random seed.

This keeps the dataset-selection process explicit and reproducible rather than relying on manually selected images.

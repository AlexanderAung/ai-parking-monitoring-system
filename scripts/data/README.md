# Dataset preparation tools

These standalone tools prepare training data. All configured paths resolve from
the repository root using the script location, regardless of the working directory.
Review the configuration constants in each script before running it.

| Tool | Input | Output |
| --- | --- | --- |
| `audit.py` | `data/raw/train/` and `data/raw/annotations/train_annotations.json` | `artifacts/data-audit/local/audit_report/` |
| `issues_checker.py` | Local audit issues, raw training images, and COCO annotations | `artifacts/data-audit/local/audit_inspection/` |
| `select_dataset.py` | Raw training data and local audit issues | `data/selected_v0_1/` |
| `select_for_baseline.py` | `data/selected_v0_1/` | `data/baseline_test/` |

For YOLO-format auditing, set `ANNO_FORMAT = "yolo"` and supply labels in
`data/raw/labels/`, or change `ANNO_DIR` to the desired location.

From the repository root, run tools as needed:

```bash
uv run python scripts/data/audit.py
uv run python scripts/data/issues_checker.py
uv run python scripts/data/select_dataset.py
uv run python scripts/data/select_for_baseline.py
```

Auditing regenerates local report files, and issue visualization can replace
matching local output images. Historical reports under
`artifacts/data-audit/roboflow-v13-train/` are separate evidence.
The selection scripts retain their existing overwrite protections. The sampler
currently targets **100 images**, not the historical 10,000-image baseline.

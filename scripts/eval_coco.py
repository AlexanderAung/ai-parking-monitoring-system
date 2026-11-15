"""Run standard COCO bbox evaluation on all saved predictions."""

import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "runs/baseline-001/eval-epoch19/fresh-evaluation/predictions.json"
GT_PATH = ROOT / "data/raw/annotations/kaggle_val_annotations.json"
OUTPUT_DIR = ROOT / "runs/baseline-001/eval-epoch19/error-analysis"
METRIC_NAMES = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


def main():
    """Evaluate unfiltered predictions, then save and print the 12 COCO stats."""
    with GT_PATH.open(encoding="utf-8") as file:
        ground_truth = json.load(file)
    ground_truth["categories"] = [{"id": 0, "name": "license_plate"}]

    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_predictions = coco_gt.loadRes(str(PREDICTIONS_PATH))

    evaluator = COCOeval(coco_gt, coco_predictions, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "coco_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

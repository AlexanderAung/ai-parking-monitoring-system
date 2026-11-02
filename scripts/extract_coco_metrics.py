import numpy as np
import torch

result = torch.load(
    "runs/baseline-001/eval-epoch19/eval.pth",
    map_location="cpu",
    weights_only=False,
)
params = result["params"]
max_dets_index = params.maxDets.index(100)

print("| Metric | Overall | Small | Medium | Large |")
print("|---|---:|---:|---:|---:|")

for name, iou in [("AP50", 0.50), ("AP75", 0.75), ("AP50:95", None)]:
    precision = result["precision"]

    if iou is not None:
        precision = precision[np.isclose(params.iouThrs, iou)]

    cells = []
    for area in ["all", "small", "medium", "large"]:
        area_index = params.areaRngLbl.index(area)
        values = precision[:, :, :, area_index, max_dets_index]
        valid = values[values > -1]
        cells.append(f"{valid.mean():.4f}" if valid.size else "N/A")

    print("| " + " | ".join([name, *cells]) + " |")

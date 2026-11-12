import argparse 
from pathlib import Path 
import torch 
from predict_validation import load_pipeline, predict_batch 
import json 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export validation predictions in COCO JSON format"
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to the model YAML configuration.",
    )

    parser.add_argument(
        "-r", "--resume",
        type=Path,
        required=True,
        help="Path to the checkpoint containing EMA weights.",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path for the output predictions JSON.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for inference, such as cpu or cuda.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images processed per batch.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    return args

def to_coco_records(image_id, prediction):
    records = []
    boxes = prediction["boxes"].tolist()
    scores = prediction["scores"].tolist()
    labels = prediction["labels"].tolist()

    for box, score, label in zip(boxes, scores, labels):
        left, top, right, bottom = box

        records.append({
            "image_id": int(image_id),
            "category_id": int(label),
            "bbox": [left, top, right - left, bottom - top],
            "score": float(score),
        })

    return records


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, postprocessor, loader = load_pipeline(
        config= args.config, 
        checkpoint = args.resume, 
        device = device, 
        batch_size = args.batch_size, 
    )
    print("Model and EMA weights loadedl")
    print(f"Validation images: {len(loader.dataset)}")

    results = [ ]

    for samples, targets in loader: 
        predictions = predict_batch(
            model, postprocessor, samples, targets, device
        )

        for target, prediction in zip(targets, predictions):
            image_id = target["image_id"].item()

            records = to_coco_records(image_id, prediction)
            results.extend(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding = "utf-8") as file: 
        json.dump(results, file, indent = 2 , allow_nan=False)
    print(f"Saved {len(results)} detections to {args.output}")

   
if __name__ == "__main__": 
    main()




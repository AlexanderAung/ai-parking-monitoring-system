import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from collections import Counter
import torch
import yaml

from predict_validation import load_pipeline, predict_batch


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

def to_coco_records(
    image_id: int,
    prediction: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    """Validate one image's predictions and return COCO detection records.

    Input boxes must be floating-point original-image XYXY coordinates.
    Scores must be floating-point values in [0, 1], and labels must be
    integer category IDs equal to 0 for this baseline.

    Return every prediction as original-image XYWH, without filtering,
    clipping, or rescaling. Properly shaped empty tensors return [].

    Raise ValueError for an invalid image ID or malformed predictions.
    """
    if isinstance(image_id, bool) or not isinstance(image_id, int):
        raise ValueError("image_id must be an integer")

    for name in ("boxes", "scores", "labels"):
        if not isinstance(prediction.get(name), torch.Tensor):
            raise ValueError(f"Image {image_id}: {name} must be a torch.Tensor")

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Image {image_id}: boxes must have shape (N, 4)")

    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError(f"Image {image_id}: scores and labels must have shape (N,)")

    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(f"Image {image_id}: box, score, and label counts differ")

    if not boxes.is_floating_point() or not scores.is_floating_point():
        raise ValueError(f"Image {image_id}: boxes and scores must be floating-point")

    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError(f"Image {image_id}: labels must have an integer dtype")

    records: list[dict[str, object]] = []

    for index, (box, score, label) in enumerate(
        zip(boxes.tolist(), scores.tolist(), labels.tolist(), strict=True)
    ):
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        context = f"Image {image_id}, prediction {index}"

        if not all(
            math.isfinite(value)
            for value in (left, top, right, bottom, width, height, score)
        ):
            raise ValueError(f"{context}: coordinates and score must be finite")

        if width <= 0 or height <= 0:
            raise ValueError(f"{context}: box width and height must be positive")

        if not 0 <= score <= 1:
            raise ValueError(f"{context}: score must be in [0, 1]")

        if label != 0:
            raise ValueError(f"{context}: baseline category_id must be 0")

        records.append({
            "image_id": image_id,
            "category_id": label,
            "bbox": [left, top, width, height],
            "score": score,
        })

    return records

def file_identity(path: Path) -> dict[str, str | int]:
    """Return a local file's absolute path, byte count, and SHA-256.

    Read incrementally to avoid loading the entire file into memory.
    Propagate file-access errors rather than returning incomplete identity.
    """
    resolved_path = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    size_bytes = 0

    with resolved_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break

            digest.update(chunk)
            size_bytes += len(chunk)

    return {
        "path": str(resolved_path),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }

def configuration_identities(config: Path) -> list[dict[str, str | int]]:
    """Identify the entry YAML and all recursively included YAML files.

    Resolve relative includes against the containing YAML's directory.
    Record each file once. Raise ValueError for invalid configuration
    structure or circular includes; propagate parsing and file errors.
    """
    identities: list[dict[str, str | int]] = []
    visited: set[Path] = set()
    active: set[Path] = set()

    def visit(path: Path) -> None:
        resolved_path = path.expanduser().resolve(strict=True)

        if resolved_path in active:
            raise ValueError(f"Circular configuration include: {resolved_path}")

        if resolved_path in visited:
            return

        active.add(resolved_path)

        with resolved_path.open("r", encoding="utf-8") as file:
            settings = yaml.safe_load(file)

        if settings is None:
            settings = {}

        if not isinstance(settings, dict):
            raise ValueError(
                f"Configuration must contain a mapping: {resolved_path}"
            )

        includes = settings.get("__include__", [])
        if not isinstance(includes, list) or any(
            not isinstance(item, str) for item in includes
        ):
            raise ValueError(
                f"__include__ must be a list of paths: {resolved_path}"
            )

        identities.append(file_identity(resolved_path))

        for included_file in includes:
            included_path = Path(included_file).expanduser()
            if not included_path.is_absolute():
                included_path = resolved_path.parent / included_path
            visit(included_path)

        active.remove(resolved_path)
        visited.add(resolved_path)

    visit(config)
    return identities

def collect_run_provenance(
    args: argparse.Namespace,
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> dict[str, object]:
    """Collect input, source, environment, and inference provenance.

    Call after pipeline initialization and before prediction begins.
    Return JSON-compatible metadata without saving files or running inference.
    Propagate file, package-metadata, and Git errors.
    """
    project_root = Path(__file__).resolve().parent.parent
    dataset = loader.dataset

    def git_output(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments],
            text=True,
        ).rstrip("\n")

    source_paths = (
        project_root / "scripts/export_validation_predictions_json.py",
        project_root / "scripts/predict_validation.py",
        project_root
        / "thrid-party/RT-DETR/rtdetrv2_pytorch"
        / "src/zoo/rtdetr/rtdetr_postprocessor.py",
    )
    packages = (
        "torch",
        "torchvision",
        "faster-coco-eval",
        "pycocotools",
        "numpy",
        "pillow",
        "PyYAML",
    )

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "checkpoint": file_identity(args.resume),
        "weight_selection": "ema.module",
        "annotations": file_identity(Path(dataset.ann_file)),
        "images_directory": str(
            Path(dataset.img_folder).resolve(strict=True)
        ),
        "image_content_hashes_recorded": False,
        "expected_image_count": len(dataset.coco.imgs),
        "configuration": {
            "entry_path": str(args.config.expanduser().resolve(strict=True)),
            "files": configuration_identities(args.config),
            "backbone_pretrained_override": False,
        },
        "source": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status": git_output(
                "status", "--porcelain=v1", "--untracked-files=normal"
            ).splitlines(),
            "files": [file_identity(path) for path in source_paths],
        },
        "environment": {
            "python": sys.version,
            "packages": {name: version(name) for name in packages},
        },
        "inference": {
            "requested_device": str(torch.device(args.device)),
            "parameter_devices": sorted({
                str(parameter.device) for parameter in model.parameters()
            }),
            "parameter_dtypes": sorted({
                str(parameter.dtype) for parameter in model.parameters()
            }),
            "model_training": model.training,
            "postprocessor_training": postprocessor.training,
            "batch_size": loader.batch_size,
            "num_workers": loader.num_workers,
            "shuffle": loader.shuffle,
            "drop_last": loader.drop_last,
            "num_classes": postprocessor.num_classes,
            "num_top_queries": postprocessor.num_top_queries,
            "use_focal_loss": postprocessor.use_focal_loss,
            "remap_mscoco_category": postprocessor.remap_mscoco_category,
            "autocast_enabled": False,
            "confidence_cutoff": None,
            "nms_applied": False,
            "export_box_format": "xywh",
            "coordinate_space": "original_image_pixels",
            "original_size_order": ["width", "height"],
        },
    }

def prediction_output_paths(output: Path) -> tuple[Path, Path]:
    """Return prediction and manifest paths, rejecting existing destinations."""
    prediction_path = output.expanduser().absolute()
    manifest_path = prediction_path.with_name("prediction-manifest.json")

    if prediction_path == manifest_path:
        raise ValueError("Prediction and manifest paths must be different")

    for path in (prediction_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")

    return prediction_path, manifest_path


def write_prediction_artifacts(
    output: Path,
    predictions: list[dict[str, object]],
    processed_counts: dict[int, int],
    provenance: dict[str, object],
) -> tuple[Path, Path]:
    """Save validated predictions and their processing/provenance manifest.

    The caller must first verify complete coverage of expected image IDs.
    Reject inconsistent counts, changed input/source files, and existing
    destinations. Propagate write errors; partial files may remain on failure.
    """
    prediction_path, manifest_path = prediction_output_paths(output)

    if len(processed_counts) != provenance["expected_image_count"]:
        raise ValueError("Processed image count differs from the expected count")

    if any(count < 0 for count in processed_counts.values()):
        raise ValueError("Prediction counts must be nonnegative")

    actual_counts = Counter(record["image_id"] for record in predictions)
    expected_nonempty_counts = {
        image_id: count
        for image_id, count in processed_counts.items()
        if count > 0
    }
    if actual_counts != expected_nonempty_counts:
        raise ValueError("Prediction records disagree with per-image counts")

    recorded_files = [
        provenance["checkpoint"],
        provenance["annotations"],
        *provenance["configuration"]["files"],
        *provenance["source"]["files"],
    ]
    for recorded in recorded_files:
        current = file_identity(Path(recorded["path"]))
        if current != recorded:
            raise RuntimeError(
                f"Input or source file changed during the run: {recorded['path']}"
            )

    # Validate provenance serialization before creating output files.
    json.dumps(provenance, allow_nan=False)

    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    with prediction_path.open("x", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2, allow_nan=False)
        file.write("\n")

    manifest = {
        "schema_version": 1,
        "export_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_image_count": len(processed_counts),
        "prediction_count": len(predictions),
        "processed_images": [
            {
                "image_id": image_id,
                "prediction_count": count,
            }
            for image_id, count in sorted(processed_counts.items())
        ],
        "predictions": file_identity(prediction_path),
        "provenance": provenance,
    }

    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")

    return prediction_path, manifest_path

def main():
    """Export complete validation predictions with processing provenance."""
    args = parse_args()

    # Fail early if either output destination is already occupied.
    args.output, _ = prediction_output_paths(args.output)
    args.config = args.config.expanduser().resolve(strict=True)
    args.resume = args.resume.expanduser().resolve(strict=True)
    device = torch.device(args.device)

    model, postprocessor, loader = load_pipeline(
        config=args.config,
        checkpoint=args.resume,
        device=device,
        batch_size=args.batch_size,
    )

    dataset = loader.dataset
    expected_ids = set(dataset.coco.imgs)

    if not expected_ids:
        raise ValueError("Validation annotations contain no images")

    if (
        len(dataset.ids) != len(expected_ids)
        or set(dataset.ids) != expected_ids
    ):
        raise ValueError(
            "Validation dataset IDs do not match the annotation image IDs"
        )

    total_images = len(expected_ids)
    results: list[dict[str, object]] = []
    processed_counts: dict[int, int] = {}

    print("Model and EMA weights loaded", flush=True)
    print(f"Expected validation images: {total_images}", flush=True)
    print("Collecting run provenance...", flush=True)

    provenance = collect_run_provenance(
        args, model, postprocessor, loader
    )

    for samples, targets in loader:
        predictions = predict_batch(
            model, postprocessor, samples, targets, device
        )

        if not (len(samples) == len(targets) == len(predictions)):
            raise RuntimeError(
                "Batch image, target, and prediction counts differ"
            )

        for target, prediction in zip(targets, predictions, strict=True):
            image_id = target["image_id"].item()

            if image_id not in expected_ids:
                raise ValueError(f"Unexpected validation image ID: {image_id}")

            if image_id in processed_counts:
                raise ValueError(f"Image ID processed more than once: {image_id}")

            records = to_coco_records(image_id, prediction)
            results.extend(records)
            processed_counts[image_id] = len(records)

        processed = len(processed_counts)
        if (
            processed == len(targets)
            or processed % 100 == 0
            or processed == total_images
        ):
            print(
                f"Processed {processed}/{total_images} images; "
                f"collected {len(results)} predictions",
                flush=True,
            )

    missing_ids = expected_ids - set(processed_counts)
    if missing_ids:
        raise RuntimeError(
            f"Missing {len(missing_ids)} validation images; "
            f"first missing IDs: {sorted(missing_ids)[:10]}"
        )

    prediction_path, manifest_path = write_prediction_artifacts(
        output=args.output,
        predictions=results,
        processed_counts=processed_counts,
        provenance=provenance,
    )

    print(
        f"Completed {len(processed_counts)} images; "
        f"exported {len(results)} detections",
        flush=True,
    )
    print(f"Predictions: {prediction_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

if __name__ == "__main__": 
    main()




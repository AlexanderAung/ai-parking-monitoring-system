"""
audit.py — Data audit for object-detection training data (RT-DETR baseline)

Audit rule: an auditor MEASURES and REPORTS. It never silently "fixes" data.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image

# ===================== CONFIG  =====================
IMAGE_DIR = Path("../../data/raw/train")            # folder with images (recursive OK)
ANNO_FORMAT = "coco"                           # "yolo" (.txt per image) or "coco" (single .json)
ANNO_DIR = Path("/path/to/labels")             # used when ANNO_FORMAT == "yolo"
COCO_JSON = Path("../../data/raw/annotations/train_annotations.json")  # used when ANNO_FORMAT == "coco"
CLASS_NAMES = ["unsued", "license_plate"]                # index = class id; extend if multi-class
OUTPUT_DIR = Path("./audit_report")
NUM_WORKERS = 16
MIN_IMG_SIDE = 32
TINY_BOX_MAX_REL_AREA = 5e-4   # boxes < 0.05% of image area = probably unlearnable
GIANT_BOX_MIN_REL_AREA = 0.90  # a "plate" covering 90% of the frame is suspicious
EPS = 1e-6
# =========================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEV_ERROR, SEV_WARN = "ERROR", "WARN"

@dataclass
class Issue:
    file: str; check: str; severity: str; detail: str

@dataclass
class AuditState:
    n_images: int = 0
    n_annos: int = 0
    issues: list = field(default_factory=list)
    img_w: list = field(default_factory=list)
    img_h: list = field(default_factory=list)
    boxes_per_img: list = field(default_factory=list)
    box_rel_area: list = field(default_factory=list)
    box_abs_w: list = field(default_factory=list)
    box_abs_h: list = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    n_empty_labels: int = 0
    n_boxes_clipped: int = 0
    def add(self, file, check, severity, detail):
        self.issues.append(Issue(file, check, severity, detail))

# ----------------------------- loaders -----------------------------
def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p

def load_yolo_anno(txt_path: Path):
    """Return list of (cls, cx, cy, w, h) normalized, or raise ValueError."""
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"line {ln}: expected 5 fields, got {len(parts)}")
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            boxes.append((cls, cx, cy, w, h))
    return boxes

def load_coco_index(coco_path: Path):
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    id2file = {im["id"]: im["file_name"] for im in data["images"]}
    index = defaultdict(list)
    for ann in data["annotations"]:
        index[id2file[ann["image_id"]]].append((ann["category_id"], *ann["bbox"]))
    return index

# ----------------------------- checks -----------------------------
def check_one_image(img_path: Path, state: AuditState):
    """
    Check structure 
    Check truncated files 
    """
    rel = str(img_path)
    try:
        with Image.open(img_path) as im:
            im.verify()          # cheap structural check
        with Image.open(img_path) as im:
            im.load()            # full decode — catches truncated files
            w, h = im.size
            if im.mode not in ("RGB", "L"):
                state.add(rel, "image_mode", SEV_WARN, f"mode={im.mode}")
    except Exception as e:
        state.add(rel, "image_corrupt", SEV_ERROR, f"{type(e).__name__}: {e}")
        return
    state.img_w.append(w); state.img_h.append(h)
    if min(w, h) < MIN_IMG_SIDE:
        state.add(rel, "image_tiny", SEV_WARN, f"{w}x{h}")
    if w / max(h, 1) > 5 or h / max(w, 1) > 5:
        state.add(rel, "image_extreme_aspect", SEV_WARN, f"{w}x{h}")

def hash_file(p: Path) -> str:
    m = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()

def audit_boxes_for_image(rel, W, H, boxes, is_normalized, state: AuditState):
    if not boxes:
        state.n_empty_labels += 1
        state.add(rel, "label_empty", SEV_WARN, "no objects")
        state.boxes_per_img.append(0)
        return
    state.boxes_per_img.append(len(boxes))
    n_clipped_this = 0
    for cls, x, y, w, h in boxes:
        if cls < 0 or cls >= len(CLASS_NAMES):
            state.add(rel, "class_id_invalid", SEV_ERROR, f"cls={cls}"); continue
        state.class_counts[CLASS_NAMES[cls]] += 1
        clipped = False
        if is_normalized:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                state.add(rel, "box_center_oob", SEV_ERROR, f"cx,cy=({x:.3f},{y:.3f})")
                continue
            if not (EPS < w <= 1.0 + EPS and EPS < h <= 1.0 + EPS):
                state.add(rel, "box_size_invalid", SEV_ERROR, f"w,h=({w:.4f},{h:.4f})")
                continue

            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
            clipped = x1 < -EPS or y1 < -EPS or x2 > 1 + EPS or y2 > 1 + EPS

            aw, ah, rel_area = w * W, h * H, w * h
        else:
            if w <= 0 or h <= 0:
                state.add(rel, "box_size_invalid", SEV_ERROR, f"w,h=({w},{h})")
                continue

            clipped = x < 0 or y < 0 or x + w > W or y + h > H

            aw, ah = w, h
            rel_area = (w * h) / max(W * H, 1)
        if clipped:
            state.n_boxes_clipped += 1
            state.add(rel, "box_out_of_bounds", SEV_WARN, "box exceeds image bounds")

        
        state.box_rel_area.append(rel_area)
        state.box_abs_w.append(aw); state.box_abs_h.append(ah)
        if rel_area < TINY_BOX_MAX_REL_AREA:
            state.add(rel, "box_tiny", SEV_WARN, f"rel_area={rel_area:.2e} ({aw:.0f}x{ah:.0f}px)")
        if rel_area > GIANT_BOX_MIN_REL_AREA:
            state.add(rel, "box_suspiciously_large", SEV_WARN, f"rel_area={rel_area:.2f}")

# ----------------------------- report -----------------------------
def pct(part, whole): return f"{100.0 * part / max(whole, 1):.2f}%"

def print_summary(state, n_pairs, n_img_orphan, n_ann_orphan, dup_groups):
    err = sum(1 for i in state.issues if i.severity == SEV_ERROR)
    warn = len(state.issues) - err
    print("\n" + "=" * 70 + "\nAUDIT SUMMARY\n" + "=" * 70)
    print(f"images found        : {state.n_images}")
    print(f"annotations found   : {state.n_annos}")
    print(f"image+label pairs   : {n_pairs}")
    print(f"orphan images       : {n_img_orphan}")
    print(f"orphan labels       : {n_ann_orphan}")
    print(f"duplicate images    : {sum(len(g) for g in dup_groups)} files in {len(dup_groups)} groups")
    print(f"empty label files   : {state.n_empty_labels}")
    print(f"ISSUES: {err} ERROR, {warn} WARN")
    if state.box_rel_area:
        a = np.array(state.box_rel_area)
        print("\nbox rel area: min={:.2e} p1={:.2e} p50={:.2e} p99={:.2e} max={:.2e}".format(
            a.min(), *np.percentile(a, [1, 50, 99]), a.max()))
        print(f"tiny (<{TINY_BOX_MAX_REL_AREA:.0e}): {pct((a < TINY_BOX_MAX_REL_AREA).sum(), len(a))}")
    if state.box_abs_w:
        w = np.array(state.box_abs_w); h = np.array(state.box_abs_h)
        print(f"box abs W(px): p1={np.percentile(w,1):.0f} p50={np.percentile(w,50):.0f} p99={np.percentile(w,99):.0f}")
        print(f"box abs H(px): p1={np.percentile(h,1):.0f} p50={np.percentile(h,50):.0f} p99={np.percentile(h,99):.0f}")
    print("\nper-class box counts:")
    for name, c in state.class_counts.most_common():
        print(f"  {name:20s}: {c}")
    print("=" * 70)

def write_artifacts(state, n_pairs, n_img_orphan, n_ann_orphan, dup_groups):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "issues.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["file", "check", "severity", "detail"])
        for i in state.issues:
            wr.writerow([i.file, i.check, i.severity, i.detail])
    summary = {
        "images": state.n_images, "annotations": state.n_annos, "pairs": n_pairs,
        "orphan_images": n_img_orphan, "orphan_labels": n_ann_orphan,
        "duplicate_files": sum(len(g) for g in dup_groups),
        "empty_labels": state.n_empty_labels,
        "class_counts": dict(state.class_counts),
        "n_errors": sum(1 for i in state.issues if i.severity == SEV_ERROR),
        "n_warnings": sum(1 for i in state.issues if i.severity == SEV_WARN),
    }
    if state.box_rel_area:
        a = np.array(state.box_rel_area)
        summary["box_rel_area_percentiles"] = {str(p): float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
        summary["tiny_box_fraction"] = float((a < TINY_BOX_MAX_REL_AREA).mean())
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        if state.box_rel_area:
            axes[0,0].hist(np.log10(np.array(state.box_rel_area) + 1e-8), bins=60)
            axes[0,0].axvline(np.log10(TINY_BOX_MAX_REL_AREA), color="r", ls="--")
            axes[0,0].set_title("log10(box relative area)")
        if state.img_w:
            axes[0,1].scatter(state.img_w, state.img_h, s=1, alpha=0.2)
            axes[0,1].set_title("image W vs H")
        if state.boxes_per_img:
            bpc = Counter(state.boxes_per_img)
            axes[1,0].bar(bpc.keys(), bpc.values()); axes[1,0].set_title("boxes per image")
        if state.class_counts:
            axes[1,1].bar(state.class_counts.keys(), state.class_counts.values())
            axes[1,1].set_title("boxes per class"); axes[1,1].tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / "distributions.png", dpi=110)
    except Exception as e:
        print(f"(plotting skipped: {e})", file=sys.stderr)
    print(f"\nreport written to {OUTPUT_DIR}/  (issues.csv, summary.json, distributions.png)")

# ----------------------------- main -----------------------------
def main():
    if not IMAGE_DIR.is_dir():
        sys.exit(f"IMAGE_DIR does not exist: {IMAGE_DIR}")
    state = AuditState()
    images = list(iter_images(IMAGE_DIR))
    state.n_images = len(images)

    # First print line
    print(f"[1/5] {len(images)} images; checking integrity...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        list(ex.map(lambda p: check_one_image(p, state), images))

    # Second print line check the format, and the orphan img & anno counts
    print("[2/5] pairing images with annotations...")
    anno_map = {}
    if ANNO_FORMAT == "yolo":
        for img in images:
            txt = ANNO_DIR / f"{img.stem}.txt"
            if txt.exists():
                try:
                    anno_map[img] = ("yolo", load_yolo_anno(txt))
                except ValueError as e:
                    state.add(str(txt), "anno_parse", SEV_ERROR, str(e))
        state.n_annos = len(list(ANNO_DIR.rglob("*.txt")))
        img_keys = {im.stem for im in images}
        anno_keys = {im.stem for im in anno_map}
        n_img_orphan = len(img_keys - anno_keys)
        n_ann_orphan = sum(1 for t in ANNO_DIR.rglob("*.txt") if t.stem not in img_keys)
    elif ANNO_FORMAT == "coco":
        # find orphan images and orphan annotations
        coco_index = load_coco_index(COCO_JSON)
        names = {im.name: im for im in images}
        for fname, boxes in coco_index.items():
            if fname in names:
                anno_map[names[fname]] = ("coco", boxes)
        state.n_annos = sum(len(boxes) for boxes in coco_index.values())
        n_img_orphan = len(images) - len(anno_map)
        n_ann_orphan = 0
    else:
        sys.exit(f"unknown ANNO_FORMAT: {ANNO_FORMAT}")
    print(f"      pairs={len(anno_map)}  orphan_images={n_img_orphan}  orphan_labels={n_ann_orphan}")

    # Third print line check anno boxes 
    print("[3/5] validating boxes...")
    for img, (fmt, boxes) in anno_map.items():
        with Image.open(img) as im:
            W, H = im.size
        audit_boxes_for_image(str(img), W, H, boxes, fmt == "yolo", state)

    # Forth print line 
    print("[4/5] hashing for exact duplicates...")
    hashes = defaultdict(list)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for img, digest in zip(images, ex.map(hash_file, images)):
            hashes[digest].append(img)
    dup_groups = [g for g in hashes.values() if len(g) > 1]
    for g in dup_groups:
        for extra in g[1:]:
            state.add(str(extra), "duplicate_exact", SEV_WARN, f"same as {g[0].name}")

    # Firth print line
    print("[5/5] writing report...")
    print_summary(state, len(anno_map), n_img_orphan, n_ann_orphan, dup_groups)
    write_artifacts(state, len(anno_map), n_img_orphan, n_ann_orphan, dup_groups)

if __name__ == "__main__":
    main()


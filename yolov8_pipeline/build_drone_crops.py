"""
Build a directory of tightly-cropped drone patches (run ONCE, setup step).

Extracts every labelled drone box from the training set and saves it as a small
image under ``dataset/drone_crops/``. These crops are consumed at training time
by the small-drone paste augmentation (see small_drone_paste.py), which pastes
them onto training images at a tiny apparent size to simulate distant drones.

This script does NOT modify the base dataset — it only reads labels/images and
writes new crop files into a separate directory.

Usage:
    python build_drone_crops.py
    python build_drone_crops.py --images dataset/images/train \
        --labels dataset/labels/train --out dataset/drone_crops \
        --classes 0 --min-box 12 --pad 0.1
"""

import os
import sys

from _conda_env import conda_python

_YOLOV8_PYTHON = conda_python("yolov8", "YOLOV8_PYTHON")
if os.path.exists(_YOLOV8_PYTHON) and os.path.realpath(
    sys.executable
) != os.path.realpath(_YOLOV8_PYTHON):
    os.execv(_YOLOV8_PYTHON, [_YOLOV8_PYTHON] + sys.argv)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _find_image(images_dir, stem):
    """Find the image file matching a label stem, trying common extensions."""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def build_drone_crops(
    images_dir="dataset/images/train",
    labels_dir="dataset/labels/train",
    out_dir="dataset/drone_crops",
    classes=(0,),
    min_box=12,
    pad=0.1,
):
    """Extract tight drone crops from labelled training images."""
    import cv2

    if not os.path.isdir(images_dir):
        print(f"ERROR: images dir not found: {images_dir}")
        return
    if not os.path.isdir(labels_dir):
        print(f"ERROR: labels dir not found: {labels_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    classes = set(int(c) for c in classes)

    print("\n" + "=" * 70)
    print("Building drone crops")
    print("=" * 70)
    print(f"  Images:   {images_dir}")
    print(f"  Labels:   {labels_dir}")
    print(f"  Output:   {out_dir}")
    print(f"  Classes:  {sorted(classes)}")
    print(f"  Min box:  {min_box}px   Padding: {pad:.0%}")
    print("=" * 70 + "\n")

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    saved = 0
    skipped_small = 0

    for lf in label_files:
        stem = os.path.splitext(lf)[0]
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            continue
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        with open(os.path.join(labels_dir, lf), "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id not in classes:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])

            # Add padding around the box, then convert to pixel xyxy.
            bw *= 1.0 + pad
            bh *= 1.0 + pad
            x1 = int(round((cx - bw / 2) * w))
            y1 = int(round((cy - bh / 2) * h))
            x2 = int(round((cx + bw / 2) * w))
            y2 = int(round((cy + bh / 2) * h))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < min_box or y2 - y1 < min_box:
                skipped_small += 1
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_name = f"{stem}_{idx}.png"
            cv2.imwrite(os.path.join(out_dir, out_name), crop)
            saved += 1

    print("=" * 70)
    print("Done")
    print(f"  Crops saved:        {saved}")
    print(f"  Skipped (too small):{skipped_small}")
    print(f"  Output directory:   {os.path.abspath(out_dir)}")
    print("=" * 70)
    if saved:
        print(
            "\nNext step: enable the small-drone paste augmentation during training:\n"
            "  python train_yolov8.py --small-drone-crops dataset/drone_crops"
        )
    else:
        print("\nNo crops were produced — check your image/label paths and classes.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract tight drone crops for paste augmentation (run once)."
    )
    parser.add_argument(
        "--images",
        type=str,
        default="dataset/images/train",
        help="Training images directory (default: dataset/images/train)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="dataset/labels/train",
        help="Training labels directory (default: dataset/labels/train)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="dataset/drone_crops",
        help="Output crops directory (default: dataset/drone_crops)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0],
        help="Class id(s) to extract (default: 0)",
    )
    parser.add_argument(
        "--min-box",
        type=int,
        default=12,
        help="Skip source boxes smaller than this many pixels (default: 12)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.1,
        help="Fractional padding added around each box (default: 0.1)",
    )

    args = parser.parse_args()
    build_drone_crops(
        images_dir=args.images,
        labels_dir=args.labels,
        out_dir=args.out,
        classes=args.classes,
        min_box=args.min_box,
        pad=args.pad,
    )

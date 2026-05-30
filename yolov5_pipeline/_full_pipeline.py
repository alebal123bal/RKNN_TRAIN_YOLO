"""
Full YOLOv5 pipeline: train → export → convert to RKNN.

Run from the workspace root:
    python yolov5_pipeline/_full_pipeline.py [options]

Artifacts are written to rknn_files/yolov5/ with the naming scheme:
    best_<name>_<img_size>.onnx
    best_<name>_<img_size>.rknn

Unlike the YOLOv8 pipeline, no ONNX output-fixing step is required: the
RockChip YOLOv5 fork already exports RKNN-ready outputs (this is the same
flow used for the RV1106 deployment).

The convert step is executed under the RKNN-Toolkit2 conda environment;
all other steps use the yolov5 conda environment.
"""

import argparse
import os
import shutil
import subprocess
import sys

from _conda_env import conda_python

_YOLOV5_PYTHON = conda_python("yolov5", "YOLOV5_PYTHON")
_RKNN_PYTHON = conda_python("RKNN-Toolkit2-rk3588s", "RKNN_PYTHON")
_RKNN_DIR = "rknn_files/yolov5"


def _run(cmd):
    display = " ".join(str(c) for c in cmd)
    print(f"\n{'='*70}")
    print(f">>> {display}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStep failed (exit code {result.returncode}). Aborting pipeline.")
        sys.exit(result.returncode)


def _artifact(stem, suffix):
    return os.path.join(_RKNN_DIR, f"{stem}{suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Full YOLOv5 pipeline: train → export → convert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Train parameters ──────────────────────────────────────────────────────
    train = parser.add_argument_group("train")
    train.add_argument(
        "--name",
        type=str,
        default="yolov5n",
        help="Run name under runs/train/ (also used in artifact names)",
    )
    train.add_argument("--img", type=int, default=640, help="Input image size")
    train.add_argument("--batch", type=int, default=16, help="Batch size")
    train.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train.add_argument(
        "--device", type=str, default="0", help="'0' for GPU, 'cpu' for CPU"
    )
    train.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Dataset data.yaml",
    )

    # ── Export parameters ─────────────────────────────────────────────────────
    export = parser.add_argument_group("export")
    export.add_argument("--opset", type=int, default=12, help="ONNX opset version")

    # ── Convert parameters ────────────────────────────────────────────────────
    conv = parser.add_argument_group("convert")
    conv.add_argument(
        "--platform",
        type=str,
        default="rv1106",
        choices=["rk3566", "rk3568", "rk3588", "rk3588s", "rv1103", "rv1106", "rk2118"],
        help="RKNN target platform",
    )
    conv.add_argument(
        "--calibration",
        type=str,
        default="yolov5_pipeline/setup_files/calibration_images.txt",
        help="Image list for INT8 quantisation calibration",
    )

    # ── Skip flags ────────────────────────────────────────────────────────────
    skip = parser.add_argument_group("skip flags (resume from a specific step)")
    skip.add_argument("--skip-train", action="store_true", help="Skip training")
    skip.add_argument("--skip-export", action="store_true", help="Skip export")
    skip.add_argument(
        "--skip-convert", action="store_true", help="Skip RKNN conversion"
    )

    args = parser.parse_args()

    # Derived artifact paths
    weights = f"runs/train/{args.name}/weights/best.pt"
    stem = f"best_{args.name}_{args.img}"
    onnx_out = _artifact(stem, ".onnx")
    rknn_out = _artifact(stem, ".rknn")

    os.makedirs(_RKNN_DIR, exist_ok=True)

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        _run(
            [
                _YOLOV5_PYTHON,
                "yolov5_pipeline/train_yolov5.py",
                "--img",
                str(args.img),
                "--batch",
                str(args.batch),
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--data",
                args.data,
                "--name",
                args.name,
            ]
        )
    else:
        print("Skipping training.")

    # ── Step 2: Export ────────────────────────────────────────────────────────
    if not args.skip_export:
        if not os.path.exists(weights):
            print(f"ERROR: weights not found at {weights}")
            print("Run without --skip-train, or train first.")
            sys.exit(1)

        _run(
            [
                _YOLOV5_PYTHON,
                "yolov5_pipeline/export_yolov5.py",
                "--weights",
                weights,
                "--img",
                str(args.img),
                "--opset",
                str(args.opset),
            ]
        )

        # export_yolov5.py writes <weights basename>.onnx (e.g. best.onnx);
        # rename to the versioned artifact name.
        default_onnx = os.path.join(
            _RKNN_DIR, os.path.basename(weights).replace(".pt", ".onnx")
        )
        if os.path.exists(default_onnx) and os.path.abspath(
            default_onnx
        ) != os.path.abspath(onnx_out):
            shutil.move(default_onnx, onnx_out)
            print(f"Renamed: {default_onnx} → {onnx_out}")
    else:
        print("Skipping export.")

    # ── Step 3: Convert ───────────────────────────────────────────────────────
    if not args.skip_convert:
        if not os.path.exists(onnx_out):
            print(f"ERROR: ONNX not found at {onnx_out}")
            print("Run without --skip-export, or export first.")
            sys.exit(1)

        if not os.path.exists(args.calibration):
            print(f"ERROR: calibration file not found at {args.calibration}")
            sys.exit(1)

        _run(
            [
                _RKNN_PYTHON,
                "yolov5_pipeline/convert_yolov5.py",
                onnx_out,
                args.calibration,
                rknn_out,
                "--platform",
                args.platform,
            ]
        )
    else:
        print("Skipping RKNN conversion.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Pipeline complete.")
    print(f"  ONNX:  {onnx_out}")
    print(f"  RKNN:  {rknn_out}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

"""
Full YOLOv8 pipeline: train → export → fix ONNX → convert to RKNN.

Run from the workspace root:
    python yolov8_pipeline/_full_pipeline.py [options]

Artifacts are written to rknn_files/yolov8/ with the naming scheme:
    best_<model>_<img_size>.onnx
    best_<model>_<img_size>_fixed.onnx
    best_<model>_<img_size>.rknn

The convert step is executed under the RKNN-Toolkit2 conda environment;
all other steps use the yolov8 conda environment.
"""

import argparse
import os
import shutil
import subprocess
import sys

from _conda_env import conda_python

_YOLOV8_PYTHON = conda_python("yolov8", "YOLOV8_PYTHON")
_RKNN_PYTHON = conda_python("RKNN-Toolkit2-rk3588s", "RKNN_PYTHON")
_RKNN_DIR = "rknn_files/yolov8"


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
        description="Full YOLOv8 pipeline: train → export → fix → convert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Train parameters ──────────────────────────────────────────────────────
    train = parser.add_argument_group("train")
    train.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model variant",
    )
    train.add_argument("--img", type=int, default=640, help="Input image size")
    train.add_argument("--batch", type=int, default=16, help="Batch size")
    train.add_argument("--epochs", type=int, default=150, help="Training epochs")
    train.add_argument(
        "--device", type=str, default="0", help="'0' for GPU, 'cpu' for CPU"
    )
    train.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Dataset data.yaml",
    )
    train.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name under runs/train/ (default: the model variant)",
    )
    train.add_argument(
        "--small-drone-crops",
        type=str,
        default=None,
        help="Directory of drone crops to enable distant-drone paste augmentation "
        "(build it once with build_drone_crops.py). Disabled if omitted.",
    )

    # ── Convert parameters ────────────────────────────────────────────────────
    conv = parser.add_argument_group("convert")
    conv.add_argument(
        "--platform",
        type=str,
        default="rk3588s",
        choices=["rk3566", "rk3568", "rk3588", "rk3588s", "rv1103", "rv1106", "rk2118"],
        help="RKNN target platform",
    )
    conv.add_argument(
        "--calibration",
        type=str,
        default="yolov8_pipeline/setup_files/calibration_images.txt",
        help="Image list for INT8 quantisation calibration",
    )

    # ── Activation ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Replace SiLU with ReLU before training (RK3588S NPU-friendly; requires training from scratch)",
    )

    # ── Skip flags ────────────────────────────────────────────────────────────
    skip = parser.add_argument_group("skip flags (resume from a specific step)")
    skip.add_argument("--skip-train", action="store_true", help="Skip training")
    skip.add_argument(
        "--skip-export", action="store_true", help="Skip export and ONNX fix"
    )
    skip.add_argument(
        "--skip-convert", action="store_true", help="Skip RKNN conversion"
    )

    args = parser.parse_args()

    # Derived artifact paths
    run_name = args.name or args.model
    weights = f"runs/train/{run_name}/weights/best.pt"
    stem = f"best_{args.model}_{args.img}"
    onnx_raw = _artifact(stem, ".onnx")
    onnx_fixed = _artifact(stem, "_fixed.onnx")
    rknn_out = _artifact(stem, ".rknn")

    os.makedirs(_RKNN_DIR, exist_ok=True)

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        train_cmd = [
            _YOLOV8_PYTHON,
            "yolov8_pipeline/train_yolov8.py",
            "--model",
            args.model,
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
        ]
        if args.name:
            train_cmd += ["--name", args.name]
        if args.small_drone_crops:
            train_cmd += ["--small-drone-crops", args.small_drone_crops]
        if args.relu:
            train_cmd.append("--relu")
        _run(train_cmd)
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
                _YOLOV8_PYTHON,
                "yolov8_pipeline/export_yolov8.py",
                "--weights",
                weights,
                "--model",
                args.model,
                "--img",
                str(args.img),
            ]
        )

        # export_yolov8.py writes best_<model>.onnx; rename to versioned name
        default_onnx = os.path.join(_RKNN_DIR, f"best_{args.model}.onnx")
        if os.path.exists(default_onnx) and os.path.abspath(
            default_onnx
        ) != os.path.abspath(onnx_raw):
            shutil.move(default_onnx, onnx_raw)
            print(f"Renamed: {default_onnx} → {onnx_raw}")

        # ── Step 2b: Fix ONNX ─────────────────────────────────────────────────
        if not os.path.exists(onnx_raw):
            print(f"ERROR: raw ONNX not found at {onnx_raw}")
            sys.exit(1)

        _run(
            [
                _YOLOV8_PYTHON,
                "yolov8_pipeline/fix_onnx_outputs.py",
                onnx_raw,
                onnx_fixed,
            ]
        )
    else:
        print("Skipping export and ONNX fix.")

    # ── Step 3: Convert ───────────────────────────────────────────────────────
    if not args.skip_convert:
        if not os.path.exists(onnx_fixed):
            print(f"ERROR: fixed ONNX not found at {onnx_fixed}")
            print("Run without --skip-export, or fix the ONNX first.")
            sys.exit(1)

        if not os.path.exists(args.calibration):
            print(f"ERROR: calibration file not found at {args.calibration}")
            sys.exit(1)

        _run(
            [
                _RKNN_PYTHON,
                "yolov8_pipeline/convert_yolov8.py",
                onnx_fixed,
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
    print(f"  ONNX (raw):   {onnx_raw}")
    print(f"  ONNX (fixed): {onnx_fixed}")
    print(f"  RKNN:         {rknn_out}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

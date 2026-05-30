"""
Export YOLOv5n model to ONNX format with RockChip RKNN modifications.
The --rknpu flag activates the RKNN model hack (separates detection heads,
replaces Focus layers) and saves RK_anchors.txt alongside the ONNX file.
"""

import os
import sys

from _conda_env import conda_python

_YOLOV5_PYTHON = conda_python("yolov5", "YOLOV5_PYTHON")
if os.path.exists(_YOLOV5_PYTHON) and os.path.realpath(
    sys.executable
) != os.path.realpath(_YOLOV5_PYTHON):
    os.execv(_YOLOV5_PYTHON, [_YOLOV5_PYTHON] + sys.argv)


def check_requirements(weights_path):
    if not os.path.exists("yolov5"):
        print("ERROR: yolov5 directory not found!")
        print("\nPlease run setup first:")
        print("  git submodule update --init")
        return False

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("\nPlease train the model first:")
        print("  python train.py")
        return False

    return True


def export_model(
    weights="runs/train/yolov5n/weights/best.pt",
    img_size=512,
    rknpu=True,
    opset=12,
):
    if not check_requirements(weights):
        return

    abs_weights = os.path.abspath(weights)

    print("\n" + "=" * 70)
    print("Exporting YOLOv5n to ONNX (RKNN-ready)")
    print("=" * 70)
    print(f"\nExport Configuration:")
    print(f"  Weights:  {abs_weights}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  RKNN hack: {'enabled' if rknpu else 'disabled'}")
    print(f"  Opset:    {opset}")

    print(f"\n{'='*70}")
    print("Running export...")
    print(f"{'='*70}\n")

    import runpy
    import torch

    # Monkey-patch: prevent torch from probing CUDA drivers before ONNX export.
    # On some driver/torch version combinations the CUDA probe triggers a
    # double-free in glibc's tcache allocator (SIGABRT / exit code 134).
    torch.cuda.is_available = lambda: False

    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")

    argv = [
        "export.py",
        "--weights",
        abs_weights,
        "--imgsz",
        str(img_size),
        "--include",
        "onnx",
        "--opset",
        str(opset),
    ]
    if rknpu:
        argv.append("--rknpu")

    orig_argv = sys.argv[:]
    orig_dir = os.getcwd()
    result = 1
    try:
        os.chdir("yolov5")
        sys.argv = argv
        runpy.run_path("export.py", run_name="__main__")
        result = 0
    except SystemExit as e:
        result = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception as exc:
        print(f"\nExport error: {exc}")
        result = 1
    finally:
        sys.argv = orig_argv
        os.chdir(orig_dir)

    output_dir = os.path.join(orig_dir, "rknn_files", "yolov5")
    os.makedirs(output_dir, exist_ok=True)

    onnx_src = abs_weights.replace(".pt", ".onnx")
    onnx_dst = os.path.join(output_dir, os.path.basename(onnx_src))
    anchors_src = "yolov5/RK_anchors.txt"
    anchors_dst = os.path.join(orig_dir, "yolov5_pipeline", "RK_anchors.txt")

    if result == 0:
        print(f"\n{'='*70}")
        print("Export completed successfully!")
        print(f"{'='*70}")
        if os.path.exists(onnx_src):
            os.replace(onnx_src, onnx_dst)
        print(f"\nONNX model saved to: {onnx_dst}")
        if rknpu and os.path.exists(anchors_src):
            os.replace(anchors_src, anchors_dst)
            print(f"RK anchors saved to:  {anchors_dst}")
        print("\nNext step: convert with RKNN Toolkit:")
        print(f"  rknn_toolkit2 convert {onnx_dst}")
    else:
        print(f"\n{'='*70}")
        print("Export failed! Check the error messages above.")
        print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export YOLOv5n to ONNX for RKNN deployment"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/yolov5n/weights/best.pt",
        help="Path to .pt weights (default: runs/train/yolov5n/weights/best.pt)",
    )
    parser.add_argument(
        "--img",
        type=int,
        default=512,
        help="Image size (default: 512, must match training)",
    )
    parser.add_argument(
        "--no-rknpu",
        action="store_true",
        help="Disable RKNN model hack (plain ONNX export)",
    )
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version (default: 12)"
    )

    args = parser.parse_args()

    export_model(
        weights=args.weights,
        img_size=args.img,
        rknpu=not args.no_rknpu,
        opset=args.opset,
    )

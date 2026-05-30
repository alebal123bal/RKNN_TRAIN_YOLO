"""
Export YOLOv8n model to ONNX format with Rockchip RKNN optimizations.

The 'rknn' export format (Rockchip fork) removes the post-processing head
and the DFL structure from the ONNX graph — both are hostile to NPU
quantization and slow on the RK NPU. Post-processing runs on CPU using the
helpers from RKNN Model Zoo.

Output: <weights_stem>.onnx, moved to rknn_files/yolov8/ when done.
"""

import os
import sys

from _conda_env import conda_python

_YOLOV8_PYTHON = conda_python("yolov8", "YOLOV8_PYTHON")
if os.path.exists(_YOLOV8_PYTHON) and os.path.realpath(
    sys.executable
) != os.path.realpath(_YOLOV8_PYTHON):
    os.execv(_YOLOV8_PYTHON, [_YOLOV8_PYTHON] + sys.argv)

# Ensure the local Rockchip fork takes precedence over any installed ultralytics.
sys.path.insert(0, os.path.abspath("yolov8"))


def check_requirements(weights_path):
    if not os.path.exists("yolov8"):
        print("ERROR: yolov8 directory not found!")
        print("\nPlease run setup first:")
        print("  git submodule update --init")
        return False

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("\nPlease train the model first:")
        print("  python train_yolov8.py")
        return False

    return True


def export_model(
    weights="runs/train/yolov8n/weights/best.pt",
    img_size=640,
    model="yolov8n",
):
    if not check_requirements(weights):
        return

    from ultralytics import YOLO

    abs_weights = os.path.abspath(weights)

    print("\n" + "=" * 70)
    print(f"Exporting {model} to ONNX (RKNN-ready)")
    print("=" * 70)
    print(f"\nExport Configuration:")
    print(f"  Weights:    {abs_weights}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Opset:      12 (fixed by Rockchip fork)")

    print(f"\n{'='*70}")
    print("Running export...")
    print(f"{'='*70}\n")

    result = 1
    try:
        yolo_model = YOLO(abs_weights)
        yolo_model.export(format="rknn", imgsz=img_size)
        result = 0
    except Exception as exc:
        print(f"\nExport error: {exc}")
        result = 1

    output_dir = os.path.abspath("rknn_files/yolov8")
    os.makedirs(output_dir, exist_ok=True)

    # The fork writes the ONNX alongside the .pt file
    onnx_src = abs_weights.replace(".pt", ".onnx")
    onnx_name = os.path.splitext(os.path.basename(abs_weights))[0] + f"_{model}.onnx"
    onnx_dst = os.path.join(output_dir, onnx_name)

    if result == 0:
        print(f"\n{'='*70}")
        print("Export completed successfully!")
        print(f"{'='*70}")
        if os.path.exists(onnx_src):
            os.replace(onnx_src, onnx_dst)
        print(f"\nONNX model saved to: {onnx_dst}")
        print("\nNext step: convert with RKNN Toolkit2:")
        print(f"  conda activate RKNN-Toolkit2")
        dataset = os.path.abspath("yolov8_pipeline/setup_files/calibration_images.txt")
        rknn_dst = onnx_dst.replace(".onnx", ".rknn")
        print(f"  python convert.py {onnx_dst} {dataset} {rknn_dst}")
    else:
        print(f"\n{'='*70}")
        print("Export failed! Check the error messages above.")
        print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX for RKNN deployment"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/yolov8n/weights/best.pt",
        help="Path to .pt weights (default: runs/train/yolov8n/weights/best.pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model variant (default: yolov8n)",
    )
    parser.add_argument(
        "--img",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )

    args = parser.parse_args()
    export_model(weights=args.weights, img_size=args.img, model=args.model)

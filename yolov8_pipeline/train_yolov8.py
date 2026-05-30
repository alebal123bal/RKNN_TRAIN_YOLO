"""
Train a YOLOv8 model for object detection.
Uses the Rockchip fork (yolov8/) — RKNN optimizations are applied at
export time only; training is identical to upstream ultralytics.
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


def check_requirements(data="dataset/data.yaml"):
    if not os.path.exists("yolov8"):
        print("ERROR: yolov8 directory not found!")
        print("\nPlease run setup first:")
        print("  git submodule update --init")
        print("  pip install -r yolov8/requirements.txt")
        return False

    if not os.path.exists(data):
        print(f"ERROR: dataset config not found at {data}")
        print("\nPlease provide a YOLO dataset (use --data to point to your data.yaml)")
        return False

    return True


def _patch_activations_relu(yolo_model):
    """Swap every nn.SiLU in the loaded model to nn.ReLU without touching the repo."""
    import torch.nn as nn

    replaced = 0
    for module in yolo_model.model.modules():
        for attr in ("act", "act1", "act2"):
            if isinstance(getattr(module, attr, None), nn.SiLU):
                setattr(module, attr, nn.ReLU(inplace=True))
                replaced += 1
    print(f"  Replaced {replaced} SiLU → ReLU instances.")


def train_model(
    img_size=640,
    batch_size=16,
    epochs=100,
    device="0",
    model="yolov8n",
    relu=False,
    data="dataset/data.yaml",
    name=None,
):
    if name is None:
        name = model
    if not check_requirements(data):
        return

    import torch.nn as nn
    from ultralytics import YOLO
    from ultralytics.nn.modules.conv import Conv

    if relu:
        # Redirect Conv.default_act so any Conv rebuilt from config uses ReLU.
        Conv.default_act = nn.ReLU(inplace=True)
        print("Activation override: SiLU → ReLU (NPU-friendly mode)")

    abs_data = os.path.abspath(data)
    abs_project = os.path.abspath("runs/train")

    print("\n" + "=" * 70)
    print(f"Training {model} Object Detection Model")
    print("=" * 70)
    print(f"\nTraining Configuration:")
    print(f"  Model:      {model}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs:     {epochs}")
    print(f"  Device:     {'GPU' if device != 'cpu' else 'CPU'}")
    print(f"  Dataset:    {data}")
    print(f"  Output:     runs/train/{name}")

    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")

    result = 1
    try:
        yolo_model = YOLO(f"{model}.pt")
        if relu:
            # Also patch any SiLU baked into the pickled pre-trained checkpoint.
            _patch_activations_relu(yolo_model)
        yolo_model.train(
            data=abs_data,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=abs_project,
            name=name,
            exist_ok=True,
            # Augmentation tuned for dark/indoor deployment environments.
            # High hsv_v simulates brightness variation (bright training → dark deploy).
            # High hsv_s covers desaturated/blueish scenes.
            # hsv_h slight increase handles cool/warm colour temperature shifts.
            hsv_h=0.03,  # hue shift  (default 0.015)
            hsv_s=0.8,  # saturation (default 0.7)
            hsv_v=0.6,  # value/brightness (default 0.4) — key for dark basement
        )
        result = 0
    except Exception as exc:
        print(f"\nTraining error: {exc}")
        result = 1

    weights_dir = os.path.join(abs_project, name, "weights")
    if result == 0:
        print(f"\n{'='*70}")
        print("Training completed successfully!")
        print(f"{'='*70}")
        print("\nModel weights saved to:")
        print(f"  Best: {os.path.join(weights_dir, 'best.pt')}")
        print(f"  Last: {os.path.join(weights_dir, 'last.pt')}")
        print("\nNext step: export with")
        print("  python export_yolov8.py")
    else:
        print(f"\n{'='*70}")
        print("Training failed! Check the error messages above.")
        print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model for object detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model variant (default: yolov8n)",
    )
    parser.add_argument(
        "--img", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' for GPU, 'cpu' for CPU (default: 0)",
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Replace SiLU activations with ReLU for RK3588S NPU friendliness (requires retraining from scratch)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to the dataset data.yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name under runs/train/ (default: the model variant)",
    )

    args = parser.parse_args()
    train_model(
        img_size=args.img,
        batch_size=args.batch,
        epochs=args.epochs,
        device=args.device,
        model=args.model,
        relu=args.relu,
        data=args.data,
        name=args.name,
    )

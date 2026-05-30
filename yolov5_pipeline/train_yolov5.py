"""
Train a YOLOv5 model for object detection.
To be launched from WSL terminal.
This uses the fork from RockChip: the SiLU have been replaced with ReLU for
compatibility with RKNN Toolkit.
"""

import os


def check_requirements(data="dataset/data.yaml"):
    """Check if forked YOLOv5 is set up"""
    if not os.path.exists("yolov5"):
        print("ERROR: yolov5 directory not found!")
        print("\nPlease run setup first:")
        print("  git submodule update --init")
        print("  pip install -r yolov5/requirements.txt")
        return False

    if not os.path.exists(data):
        print(f"ERROR: dataset config not found at {data}")
        print("\nPlease provide a YOLO dataset (use --data to point to your data.yaml)")
        return False

    return True


def train_model(
    img_size=640,
    batch_size=16,
    epochs=100,
    device="0",
    data="dataset/data.yaml",
    name="yolov5n",
):
    """
    Train YOLOv5n model

    Args:
        img_size: Image size for training (default: 640)
        batch_size: Batch size (default: 16)
        epochs: Number of training epochs (default: 100)
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        data: Path to the dataset data.yaml
        name: Run name under runs/train/
    """

    if not check_requirements(data):
        return

    print("\n" + "=" * 70)
    print("Training YOLOv5n Object Detection Model")
    print("=" * 70)

    print(f"\nTraining Configuration:")
    print(f"  Model: YOLOv5n (nano)")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {'GPU' if device != 'cpu' else 'CPU'}")
    print(f"  Dataset: {data}")
    print(f"  Output: runs/train/{name}")

    # Get yolov5 environment Python path
    from _conda_env import conda_python, wsl_ld_prefix

    yolov5_python = conda_python("yolov5", "YOLOV5_PYTHON")

    if not os.path.exists(yolov5_python):
        print(f"WARNING: yolov5 environment not found at {yolov5_python}")
        print("Using system Python instead...")
        yolov5_python = "python"

    # Build training command — use absolute paths to avoid issues with cd yolov5
    from _dataset import materialize_data_yaml

    abs_data, _data_tmp = materialize_data_yaml(data)
    abs_project = os.path.abspath("runs/train")
    cmd = (
        f"cd yolov5 && "
        f"{wsl_ld_prefix()}"
        f"YOLOv5_AUTOINSTALL=false "
        f'"{yolov5_python}" train.py '
        f"--img {img_size} "
        f"--batch {batch_size} "
        f"--epochs {epochs} "
        f"--data {abs_data} "
        f"--weights yolov5n.pt "
        f"--cache "
        f"--device {device} "
        f"--project {abs_project} "
        f"--name {name} "
        f"--exist-ok"
    )

    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")

    # Execute training
    result = os.system(cmd)

    if _data_tmp:
        os.remove(_data_tmp)

    if result == 0:
        print(f"\n{'='*70}")
        print("Training completed successfully!")
        print(f"{'='*70}")
        print("\nModel weights saved to:")
        print(f"  - Best: runs/train/{name}/weights/best.pt")
        print(f"  - Last: runs/train/{name}/weights/last.pt")
        print("\nTraining results:")
        print(f"  - Metrics: runs/train/{name}/results.png")
        print(f"  - Confusion matrix: runs/train/{name}/confusion_matrix.png")
        print("\nNext step: Run 'python test.py' to evaluate the model")
    else:
        print(f"\n{'='*70}")
        print("Training failed! Check the error messages above.")
        print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a YOLOv5 model for object detection"
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
        help="Device: 0 for GPU, cpu for CPU (default: 0)",
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
        default="yolov5n",
        help="Run name under runs/train/ (default: yolov5n)",
    )

    args = parser.parse_args()

    train_model(
        img_size=args.img,
        batch_size=args.batch,
        epochs=args.epochs,
        device=args.device,
        data=args.data,
        name=args.name,
    )

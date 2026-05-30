"""
Test a YOLOv5 model for object detection
"""

import os
import sys


def check_requirements(
    weights_path="runs/train/yolov5n/weights/best.pt", data="dataset/data.yaml"
):
    """Check if model weights exist"""
    if not os.path.exists("yolov5"):
        print("ERROR: yolov5 directory not found!")
        print("\nPlease run setup first:")
        print("  git submodule update --init")
        print("  pip install -r yolov5/requirements.txt")
        return False

    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at {weights_path}")
        print("\nPlease train the model first:")
        print("  python train.py")
        return False

    if not os.path.exists(data):
        print(f"ERROR: dataset config not found at {data}")
        return False

    return True


def validate_model(
    img_size=640,
    weights="runs/train/yolov5n/weights/best.pt",
    data="dataset/data.yaml",
):
    """
    Validate model on validation set

    Args:
        img_size: Image size for validation (default: 640)
        weights: Path to .pt weights
        data: Path to the dataset data.yaml
    """

    if not check_requirements(weights, data):
        return

    weights_path = weights
    data_yaml = data
    run_name = os.path.basename(os.path.dirname(os.path.dirname(weights_path))) + "_val"

    print("\n" + "=" * 70)
    print("Validating YOLOv5n Object Detection Model")
    print("=" * 70)

    print(f"\nValidation Configuration:")
    print(f"  Weights: {weights_path}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Output: runs/val/{run_name}")

    # Get yolov5 environment Python path
    from _conda_env import conda_python, wsl_ld_prefix

    yolov5_python = conda_python("yolov5", "YOLOV5_PYTHON")
    if not os.path.exists(yolov5_python):
        yolov5_python = "python"

    # Build validation command — use absolute paths so they resolve correctly from inside yolov5/
    from _dataset import materialize_data_yaml

    abs_weights = os.path.abspath(weights_path)
    abs_data, _data_tmp = materialize_data_yaml(data_yaml)
    abs_project = os.path.abspath("runs/val")
    cmd = (
        f"cd yolov5 && "
        f"{wsl_ld_prefix()}"
        f"YOLOv5_AUTOINSTALL=false "
        f'"{yolov5_python}" val.py '
        f"--weights {abs_weights} "
        f"--data {abs_data} "
        f"--img {img_size} "
        f"--project {abs_project} "
        f"--name {run_name} "
        f"--exist-ok"
    )

    print(f"\n{'='*70}")
    print("Running validation...")
    print(f"{'='*70}\n")

    result = os.system(cmd)

    if _data_tmp:
        os.remove(_data_tmp)

    if result == 0:
        print(f"\n{'='*70}")
        print("Validation completed!")
        print(f"{'='*70}")
        print(f"\nResults saved to: runs/val/{run_name}/")


def detect_on_images(
    source,
    img_size=640,
    conf=0.25,
    save_txt=False,
    weights="runs/train/yolov5n/weights/best.pt",
    data="dataset/data.yaml",
):
    """
    Run detection on images

    Args:
        source: Source path (image, folder, video, or webcam)
        img_size: Image size for inference (default: 640)
        conf: Confidence threshold (default: 0.25)
        save_txt: Save results as txt files (default: False)
        weights: Path to .pt weights
        data: Path to the dataset data.yaml
    """

    if not check_requirements(weights, data):
        return

    weights_path = weights

    # Resolve to absolute paths so they work correctly from inside yolov5/
    abs_weights = os.path.abspath(weights_path)
    abs_source = os.path.abspath(source) if source != "0" else "0"
    abs_project = os.path.abspath("runs")

    print("\n" + "=" * 70)
    print("YOLOv5n Object Detection")
    print("=" * 70)

    print(f"\nDetection Configuration:")
    print(f"  Weights: {weights_path}")
    print(f"  Source: {abs_source}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Confidence: {conf}")
    print(f"  Output: {abs_project}/detect/")

    # Get yolov5 environment Python path
    from _conda_env import conda_python, wsl_ld_prefix

    yolov5_python = conda_python("yolov5", "YOLOV5_PYTHON")
    if not os.path.exists(yolov5_python):
        yolov5_python = "python"

    # Build detection command
    save_txt_flag = "--save-txt" if save_txt else ""

    cmd = (
        f"cd yolov5 && "
        f"{wsl_ld_prefix()}"
        f"YOLOv5_AUTOINSTALL=false "
        f'"{yolov5_python}" detect.py '
        f"--weights {abs_weights} "
        f"--source {abs_source} "
        f"--img {img_size} "
        f"--conf {conf} "
        f"--project {abs_project} "
        f"--name detect "
        f"--exist-ok "
        f"{save_txt_flag}"
    )

    print(f"\n{'='*70}")
    print("Running detection...")
    print(f"{'='*70}\n")

    result = os.system(cmd)

    if result == 0:
        print(f"\n{'='*70}")
        print("Detection completed!")
        print(f"{'='*70}")
        print(f"\nResults saved to: {abs_project}/detect/")


def show_menu():
    """Display test menu"""
    print("\n" + "=" * 70)
    print("YOLOv5n Object Detection - Testing Menu")
    print("=" * 70)
    print("\nSelect test mode:")
    print("  1. Validate on validation set")
    print("  2. Detect on validation images")
    print("  3. Detect on single image")
    print("  4. Detect on video")
    print("  5. Detect on webcam (real-time)")
    print("  0. Exit")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test a YOLOv5 model for object detection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["val", "detect"],
        help="Test mode: val for validation, detect for detection",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source for detection (image path, folder, video, or 0 for webcam)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/yolov5n/weights/best.pt",
        help="Path to .pt weights (default: runs/train/yolov5n/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to the dataset data.yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--img", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="Save results as txt files"
    )

    args = parser.parse_args()

    # If arguments provided, run directly
    if args.mode == "val":
        validate_model(img_size=args.img, weights=args.weights, data=args.data)
    elif args.mode == "detect":
        if not args.source:
            print("ERROR: --source required for detect mode")
            print("Example: python test.py --mode detect --source path/to/images")
            sys.exit(1)
        detect_on_images(
            source=args.source,
            img_size=args.img,
            conf=args.conf,
            save_txt=args.save_txt,
            weights=args.weights,
            data=args.data,
        )
    else:
        # Interactive menu
        while True:
            show_menu()
            choice = input("\nEnter your choice (0-5): ").strip()

            if choice == "0":
                print("\nExiting...")
                break
            elif choice == "1":
                validate_model(weights=args.weights, data=args.data)
            elif choice == "2":
                detect_on_images(
                    source=args.source or "dataset/images/valid",
                    save_txt=True,
                    weights=args.weights,
                    data=args.data,
                )
            elif choice == "3":
                img_path = input("Enter image path: ").strip()
                if img_path:
                    detect_on_images(
                        source=img_path, weights=args.weights, data=args.data
                    )
            elif choice == "4":
                video_path = input("Enter video path: ").strip()
                if video_path:
                    detect_on_images(
                        source=video_path, weights=args.weights, data=args.data
                    )
            elif choice == "5":
                print("\nStarting webcam detection (press Ctrl+C to stop)...")
                detect_on_images(source="0", weights=args.weights, data=args.data)
            else:
                print("\nInvalid choice! Please enter 0-5.")

            if choice != "0":
                input("\nPress Enter to continue...")

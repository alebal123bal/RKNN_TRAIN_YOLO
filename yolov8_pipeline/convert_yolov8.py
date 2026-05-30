import argparse
import os
import sys
import tempfile

from rknn.api import RKNN

SUPPORTED_PLATFORMS = [
    "rk3566",
    "rk3568",
    "rk3588",
    "rk3588s",
    "rv1103",
    "rv1106",
    "rk2118",
]


def build_abs_dataset(dataset_path):
    """Return a dataset list file whose image paths are absolute.

    RKNN-Toolkit2 resolves the image paths inside the dataset file relative to
    that file's own directory. To allow the calibration list to use clean
    workspace-root-relative paths, rewrite it to a temporary file with each
    path resolved against the current working directory.
    """
    base_dir = os.path.dirname(os.path.abspath(dataset_path))
    cwd = os.getcwd()
    abs_lines = []
    for raw in open(dataset_path, "r"):
        line = raw.strip()
        if not line:
            continue
        if os.path.isabs(line):
            abs_lines.append(line)
            continue
        # Prefer resolving against the cwd (workspace root); fall back to the
        # dataset file's directory for backward compatibility.
        cand_cwd = os.path.join(cwd, line)
        cand_base = os.path.join(base_dir, line)
        abs_lines.append(cand_cwd if os.path.exists(cand_cwd) else cand_base)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="rknn_calib_"
    )
    tmp.write("\n".join(abs_lines) + "\n")
    tmp.close()
    return tmp.name


def parse_arg():
    parser = argparse.ArgumentParser(description="Convert ONNX model to RKNN format")
    parser.add_argument("model_path", help="Path to input ONNX model")
    parser.add_argument("dataset_path", help="Path to quantization dataset file")
    parser.add_argument("output_path", help="Path for output .rknn model")
    parser.add_argument(
        "--platform",
        type=str,
        default="rk3588s",
        choices=SUPPORTED_PLATFORMS,
        help="Target RKNN platform (default: rk3588s)",
    )
    args = parser.parse_args()
    return args.model_path, args.dataset_path, args.output_path, args.platform


if __name__ == "__main__":
    model_path, dataset_path, output_path, platform = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print(f"--> Config model  [platform: {platform}]")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
        optimization_level=3,  # maximum graph-level optimisation
    )
    print("done")

    # Load model
    print("--> Loading model")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    abs_dataset_path = build_abs_dataset(dataset_path)
    try:
        ret = rknn.build(do_quantization=True, dataset=abs_dataset_path)
    finally:
        os.remove(abs_dataset_path)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export rknn model
    print("--> Export rknn model")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    # Release
    rknn.release()

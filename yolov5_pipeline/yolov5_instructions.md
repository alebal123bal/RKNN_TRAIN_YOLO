# YOLOv5 Pipeline Instructions

All scripts must be run from the **workspace root** (`RKNN_TRAIN_YOLO/`).
The scripts use the `yolov5` conda environment (some re-exec themselves under it automatically).

This pipeline uses the **RockChip YOLOv5 fork**, in which the `SiLU` activations
are already replaced with `ReLU` for RKNN compatibility. Unlike the YOLOv8
pipeline, **no ONNX output-fixing step is required** — the fork's `--rknpu`
export already produces RKNN-ready outputs. This is the same flow used for the
**RV1106** deployment.

---

## Prerequisites

```bash
git submodule update --init
conda activate yolov5
pip install -r yolov5/requirements.txt
```

Make sure your dataset's `data.yaml` exists before training (default path: `dataset/data.yaml`, override with `--data`).

### Interpreter resolution

The scripts locate the `yolov5` and `RKNN-Toolkit2-rk3588s` Python interpreters automatically (via `CONDA_EXE` / `CONDA_PREFIX`, then common install locations) — no paths are hardcoded. If your environments live elsewhere, set override variables:

```bash
export YOLOV5_PYTHON=/path/to/envs/yolov5/bin/python
export RKNN_PYTHON=/path/to/envs/RKNN-Toolkit2-rk3588s/bin/python
```

---

## Step 1 — Train

```bash
conda activate yolov5
python yolov5_pipeline/train_yolov5.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--img` | `640` | Input image size |
| `--batch` | `16` | Batch size |
| `--epochs` | `100` | Number of training epochs |
| `--device` | `0` | `0` for GPU, `cpu` for CPU |
| `--data` | `dataset/data.yaml` | Path to the dataset `data.yaml` |
| `--name` | `yolov5n` | Run name under `runs/train/` |

Example with custom options:
```bash
conda activate yolov5
python yolov5_pipeline/train_yolov5.py --epochs 50 --batch 8 --device cpu
```

Weights are saved to `runs/train/<name>/weights/best.pt` (where `<name>` defaults to `yolov5n`).

---

## Step 2 — Export to ONNX (RKNN-optimised)

```bash
conda activate yolov5
python yolov5_pipeline/export_yolov5.py
```

The `--rknpu` model hack (enabled by default) separates the detection heads and
replaces the Focus layers, and saves `RK_anchors.txt` alongside the ONNX file.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `runs/train/yolov5n/weights/best.pt` | Path to `.pt` weights |
| `--img` | `512` | Input image size (must match training) |
| `--no-rknpu` | *(off)* | Disable the RKNN model hack (plain ONNX export) |
| `--opset` | `12` | ONNX opset version |

The ONNX file is moved to `rknn_files/yolov5/` and the anchors are written to
`yolov5_pipeline/RK_anchors.txt` on completion.

---

## Step 3 — Convert ONNX → RKNN

Requires the `RKNN-Toolkit2-rk3588s` conda environment.

```bash
conda activate RKNN-Toolkit2-rk3588s
python yolov5_pipeline/convert_yolov5.py \
    rknn_files/yolov5/best_yolov5n.onnx \
    yolov5_pipeline/setup_files/calibration_images.txt \
    rknn_files/yolov5/best_yolov5n.rknn \
    --platform rv1106
```

**Positional arguments:**

| Argument | Description |
|----------|-------------|
| `model_path` | Path to the input ONNX model |
| `dataset_path` | Path to the quantization dataset file — `yolov5_pipeline/setup_files/calibration_images.txt` |
| `output_path` | Path for the output `.rknn` model |

**Options:**

| Flag | Default | Choices |
|------|---------|---------|
| `--platform` | `rk3588s` | `rk3566`, `rk3568`, `rk3588`, `rk3588s`, `rv1103`, `rv1106`, `rk2118` |

> The calibration image paths inside `calibration_images.txt` are resolved
> against the **workspace root** at convert time, so the list stays portable.

---

## Full Pipeline (Steps 1–3 in one command)

Runs train → export → convert in sequence (no ONNX-fix step needed).
Artifacts are written to `rknn_files/yolov5/` with the naming scheme
`best_<name>_<img>.onnx` and `best_<name>_<img>.rknn`.

```bash
python yolov5_pipeline/_full_pipeline.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | `yolov5n` | Run name under `runs/train/` (also used in artifact names) |
| `--img` | `640` | Input image size |
| `--batch` | `16` | Batch size |
| `--epochs` | `100` | Training epochs |
| `--device` | `0` | `0` for GPU, `cpu` for CPU |
| `--data` | `dataset/data.yaml` | Path to the dataset `data.yaml` |
| `--opset` | `12` | ONNX opset version |
| `--platform` | `rv1106` | RKNN target platform |
| `--calibration` | `yolov5_pipeline/setup_files/calibration_images.txt` | INT8 calibration image list |
| `--skip-train` | *(off)* | Skip Step 1 |
| `--skip-export` | *(off)* | Skip Step 2 |
| `--skip-convert` | *(off)* | Skip Step 3 |

Example — resume from conversion only:
```bash
python yolov5_pipeline/_full_pipeline.py --skip-train --skip-export --platform rv1106
```

Example — full run with custom image size:
```bash
python yolov5_pipeline/_full_pipeline.py --img 640 --epochs 150
```

---

## Step 4 — Test / Validate

```bash
conda activate yolov5
python yolov5_pipeline/test_yolov5.py
```

Run validation against the dataset:
```bash
conda activate yolov5
python yolov5_pipeline/test_yolov5.py --mode val
```

Run detection on images or a folder:
```bash
conda activate yolov5
python yolov5_pipeline/test_yolov5.py --mode detect --source path/to/images
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | *(interactive menu)* | `val` or `detect` |
| `--source` | — | Image path, folder, video, or `0` for webcam (required for `detect`) |
| `--weights` | `runs/train/yolov5n/weights/best.pt` | Path to `.pt` weights |
| `--data` | `dataset/data.yaml` | Path to the dataset `data.yaml` |
| `--img` | `640` | Image size |
| `--conf` | `0.25` | Confidence threshold |
| `--save-txt` | `false` | Save results as `.txt` files |

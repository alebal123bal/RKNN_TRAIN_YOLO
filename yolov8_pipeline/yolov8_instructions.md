# YOLOv8 Pipeline Instructions

All scripts must be run from the **workspace root** (`RKNN_TRAIN_YOLO/`).
The scripts automatically re-exec themselves under the `yolov8` conda environment.

---

## Prerequisites

```bash
git submodule update --init
conda activate yolov8
pip install -r yolov8/requirements.txt
```

Make sure your dataset's `data.yaml` exists before training (default path: `dataset/data.yaml`, override with `--data`).

### Interpreter resolution

The scripts locate the `yolov8` and `RKNN-Toolkit2-rk3588s` Python interpreters automatically (via `CONDA_EXE` / `CONDA_PREFIX`, then common install locations) — no paths are hardcoded. If your environments live elsewhere, set override variables:

```bash
export YOLOV8_PYTHON=/path/to/envs/yolov8/bin/python
export RKNN_PYTHON=/path/to/envs/RKNN-Toolkit2-rk3588s/bin/python
```

---

## Step 1 — Train

```bash
conda activate yolov8
python yolov8_pipeline/train_yolov8.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolov8n` | Model variant: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` |
| `--img` | `640` | Input image size |
| `--batch` | `16` | Batch size |
| `--epochs` | `150` | Number of training epochs |
| `--device` | `0` | `0` for GPU, `cpu` for CPU |
| `--data` | `dataset/data.yaml` | Path to the dataset `data.yaml` |
| `--name` | *(model variant)* | Run name under `runs/train/` |
| `--small-drone-crops` | *(off)* | Directory of drone crops to enable distant-drone paste augmentation (build once with `build_drone_crops.py`) |
| `--relu` | *(off)* | Replace SiLU → ReLU for RK3588S NPU friendliness (requires training from scratch) |

Example with custom options:
```bash
conda activate yolov8
python yolov8_pipeline/train_yolov8.py --epochs 50 --batch 8 --device cpu
```

To train an NPU-optimised model with ReLU activations:
```bash
conda activate yolov8
python yolov8_pipeline/train_yolov8.py --relu
```

Weights are saved to `runs/train/<name>/weights/best.pt` (where `<name>` defaults to the model variant, e.g. `yolov8n`).

### Augmentation notes

The training defaults are tuned for a single-class drone detector that must cope with **distant/small drones** and **complex backgrounds**. In addition to the dark-deployment HSV settings, `train_yolov8.py` enables `copy_paste=0.5`, `scale=0.9`, `mosaic=1.0`, `hsv_s=0.4`, `degrees=10`, and a `cos_lr` schedule over the 150-epoch run. All of this is applied **at training time only** — the dataset files on disk are never modified.

Two optional helpers further target small drones and false positives — see [Step 1b](#step-1b--optional-augmentation-helpers).

---

## Step 1b — (Optional) Augmentation helpers

These address the two hardest failure modes: small/distant drones and false positives on complex backgrounds. All effects are training-time only; the base dataset is never altered.

### Build drone crops (run once)

Extract tightly-cropped drone patches from the labelled training set into `dataset/drone_crops/`. Run this **once** before enabling the paste augmentation.

```bash
conda activate yolov8
python yolov8_pipeline/build_drone_crops.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | `dataset/images/train` | Training images directory |
| `--labels` | `dataset/labels/train` | Training labels directory |
| `--out` | `dataset/drone_crops` | Output crops directory |
| `--classes` | `0` | Class id(s) to extract |
| `--min-box` | `12` | Skip source boxes smaller than this many pixels |
| `--pad` | `0.1` | Fractional padding added around each box |

### Small-drone paste augmentation

Once crops exist, enable the synthetic distant-drone paste augmentation during training by pointing `--small-drone-crops` at the crops directory. It pastes crops at a random 8–40 px size at training time and supervises the model on the matching boxes:

```bash
conda activate yolov8
python yolov8_pipeline/train_yolov8.py --small-drone-crops dataset/drone_crops
```

---

## Step 2 — Export to ONNX (RKNN-optimised)

```bash
conda activate yolov8
python yolov8_pipeline/export_yolov8.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `runs/train/yolov8n/weights/best.pt` | Path to `.pt` weights |
| `--model` | `yolov8n` | Model variant (used to name the output file) |
| `--img` | `640` | Input image size |

The ONNX file is moved to `rknn_files/yolov8/` on completion.

---

## Step 2b — Fix ONNX outputs for RKNN

The raw RKNN-format ONNX from Step 2 has 9 NCHW tensors that the NPU post-processor cannot use directly.
This script rewrites the graph to produce 3 combined NHWC tensors (one per stride) that match the C++ post-process layout.

```bash
conda activate yolov8
python yolov8_pipeline/fix_onnx_outputs.py \
    rknn_files/yolov8/best_yolov8n.onnx \
    rknn_files/yolov8/best_yolov8n_fixed.onnx
```

**Positional arguments (both optional, defaults shown above):**

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | `rknn_files/yolov8/best_yolov8n.onnx` | ONNX exported in Step 2 |
| `output` | `rknn_files/yolov8/best_yolov8n_fixed.onnx` | Fixed ONNX to feed into Step 3 |

---

## Step 3 — Convert ONNX → RKNN

Requires the `RKNN-Toolkit2-rk3588s` conda environment.

```bash
conda activate RKNN-Toolkit2-rk3588s
python yolov8_pipeline/convert_yolov8.py \
    rknn_files/yolov8/best_yolov8n_fixed.onnx \
    yolov8_pipeline/setup_files/calibration_images.txt \
    rknn_files/yolov8/best_yolov8n.rknn \
    --platform rk3588s
```

**Positional arguments:**

| Argument | Description |
|----------|-------------|
| `model_path` | Path to the input ONNX model |
| `dataset_path` | Path to the quantization dataset file — `yolov8_pipeline/setup_files/calibration_images.txt` |
| `output_path` | Path for the output `.rknn` model |

**Options:**

| Flag | Default | Choices |
|------|---------|---------|
| `--platform` | `rk3588s` | `rk3566`, `rk3568`, `rk3588`, `rk3588s`, `rv1103`, `rv1106`, `rk2118` |

---

## Full Pipeline (Steps 1–3 in one command)

Runs train → export → fix ONNX → convert in sequence. Artifacts are written to `rknn_files/yolov8/`.

```bash
python yolov8_pipeline/_full_pipeline.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolov8n` | Model variant: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` |
| `--img` | `640` | Input image size |
| `--batch` | `16` | Batch size |
| `--epochs` | `150` | Training epochs |
| `--device` | `0` | `0` for GPU, `cpu` for CPU |
| `--data` | `dataset/data.yaml` | Path to the dataset `data.yaml` |
| `--name` | *(model variant)* | Run name under `runs/train/` (also used to locate weights for export) |
| `--small-drone-crops` | *(off)* | Directory of drone crops to enable distant-drone paste augmentation (build once with `build_drone_crops.py`) |
| `--relu` | *(off)* | Replace SiLU → ReLU for RK3588S NPU friendliness |
| `--platform` | `rk3588s` | RKNN target platform |
| `--calibration` | `yolov8_pipeline/setup_files/calibration_images.txt` | INT8 calibration image list |
| `--skip-train` | *(off)* | Skip Step 1 |
| `--skip-export` | *(off)* | Skip Steps 2 and 2b |
| `--skip-convert` | *(off)* | Skip Step 3 |

Example — resume from conversion only:
```bash
python yolov8_pipeline/_full_pipeline.py --skip-train --skip-export --platform rk3588s
```

Example — full NPU-friendly run with ReLU:
```bash
python yolov8_pipeline/_full_pipeline.py --relu --epochs 150
```

Example — 960×960 ReLU run with small-drone paste augmentation (build the crops first with `build_drone_crops.py`):
```bash
python yolov8_pipeline/_full_pipeline.py \
  --img 960 --relu \
  --small-drone-crops dataset/drone_crops \
  --name yolov8n_960_relu
```

---

## Step 4 — Test / Validate

```bash
conda activate yolov8
python yolov8_pipeline/test_yolov8.py
```

Run validation against the dataset:
```bash
conda activate yolov8
python yolov8_pipeline/test_yolov8.py --mode val
```

Run detection on images or a folder:
```bash
conda activate yolov8
python yolov8_pipeline/test_yolov8.py --mode detect --source path/to/images
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | *(interactive menu)* | `val` or `detect` |
| `--source` | — | Image path, folder, video, or `0` for webcam (required for `detect`) |
| `--weights` | `runs/train/yolov8n/weights/best.pt` | Path to `.pt` weights |
| `--data` | `dataset/data.yaml` | Path to the dataset `data.yaml` |
| `--img` | `640` | Image size |
| `--conf` | `0.25` | Confidence threshold |
| `--save-txt` | `false` | Save results as `.txt` files |

# Mock Cats Dataset

This folder shows the required dataset structure for a single-class YOLO detector (class 0 = `cat`).
Copy this layout when building the real dataset and update `data.yaml` with the correct absolute path.

---

## Structure

```
_mock_cats_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── domestic_cat_0.jpg      # positive
│   │   ├── domestic_cat_1.jpg      # positive
│   │   ├── background_0.jpg        # generic negative
│   │   ├── background_1.jpg        # generic negative
│   │   ├── kitchen_0.jpg           # domain-specific negative
│   │   └── kitchen_1.jpg           # domain-specific negative
│   └── valid/
│       ├── domestic_cat_0.jpg      # positive
│       ├── background_2.jpg        # generic negative
│       └── kitchen_2.jpg           # domain-specific negative
└── labels/
    ├── train/
    │   ├── domestic_cat_0.txt      # bbox annotation
    │   ├── domestic_cat_1.txt      # bbox annotation
    │   ├── background_0.txt        # empty (no objects)
    │   ├── background_1.txt        # empty (no objects)
    │   ├── kitchen_0.txt           # empty (no objects)
    │   └── kitchen_1.txt           # empty (no objects)
    └── valid/
        ├── domestic_cat_0.txt      # bbox annotation
        ├── background_2.txt        # empty (no objects)
        └── kitchen_2.txt           # empty (no objects)
```

---

## Image categories

### Positives — `domestic_cat_*`
Images that contain a cat. Each has a matching `.txt` label file with one or more bounding boxes
in YOLO format: `<class> <x_center> <y_center> <width> <height>` (all values normalised 0–1).

### Generic negatives — `background_*`
Images with no cat and no domain-specific content (open sky, grass, generic indoor scenes, etc.).
Label files are **empty**.

### Domain-specific negatives — `kitchen_*`
Images of the kitchen — the actual environment where the detector will run.
Including these teaches the model to ignore familiar background clutter that would otherwise
cause false positives. Label files are **empty**.

---

## Label format

```
<class_id> <x_center> <y_center> <width> <height>
```
- All coordinates are normalised to [0, 1] relative to image width/height.
- `class_id` is always `0` (cat).
- Negative images have an **empty** `.txt` file (file must still exist).

## Naming convention

| Prefix          | Type                        | Label      |
|-----------------|-----------------------------|------------|
| `domestic_cat_` | Positive (cat present)      | bbox line  |
| `background_`   | Generic negative            | empty      |
| `kitchen_`      | Domain-specific negative    | empty      |

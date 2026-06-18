"""
Small-drone synthetic paste augmentation (Ultralytics training callback).

Problem this targets: the detector struggles on *distant / small* drones because
real training boxes are mostly medium/large. This augmentation pastes tightly
cropped drone patches onto training images at a tiny apparent size (8-40 px),
simulating far-away drones, and appends the matching ground-truth box so the
model is supervised on them.

How it works (training-time only — the dataset on disk is never modified):
    * ``small_drone_paste(trainer)`` is registered on the
      ``on_train_epoch_start`` event.
    * On the first epoch it injects a :class:`SmallDronePaste` transform into the
      dataset's augmentation pipeline (right before the final ``Format`` step) and
      resets the dataloader so worker processes pick up the change.
    * The transform runs per image as it is loaded: with probability ``p`` it
      pastes one or more random crops at a random small size and position, and
      appends a ``class 0`` box for each. Because it lives in the transform
      pipeline the pasted boxes flow naturally into the label tensors.

Wiring it up (see also train_yolov8.py --small-drone-crops):

    from small_drone_paste import small_drone_paste, configure
    configure(crops_dir="dataset/drone_crops")
    model.add_callback("on_train_epoch_start", small_drone_paste)

Build the crops directory once beforehand with build_drone_crops.py.
"""

import os
import random
import glob

import numpy as np

# Default configuration; override via configure().
_CONFIG = {
    "crops_dir": "dataset/drone_crops",
    "paste_prob": 0.5,  # per-image probability of pasting at all
    "max_instances": 3,  # up to this many crops pasted on a chosen image
    "min_size": 8,  # smallest pasted drone size in pixels
    "max_size": 40,  # largest pasted drone size in pixels
    "white_thresh": 245,  # pixels >= this on all channels treated as background
}


def configure(**kwargs):
    """Override paste settings (crops_dir, paste_prob, max_instances, sizes...)."""
    unknown = set(kwargs) - set(_CONFIG)
    if unknown:
        raise KeyError(f"Unknown small_drone_paste config keys: {sorted(unknown)}")
    _CONFIG.update(kwargs)
    return dict(_CONFIG)


def _load_crops(crops_dir):
    """Load all crop images (with alpha if present) from a directory."""
    import cv2

    paths = []
    for ext in ("png", "jpg", "jpeg", "bmp", "webp"):
        paths.extend(glob.glob(os.path.join(crops_dir, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(crops_dir, f"*.{ext.upper()}")))
    crops = []
    for p in sorted(set(paths)):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # keep alpha channel if any
        if img is not None:
            crops.append(img)
    return crops


class SmallDronePaste:
    """
    Pipeline transform that pastes tiny drone crops onto a training image.

    Operates on the Ultralytics ``labels`` dict (keys ``img``, ``cls``,
    ``instances``) and is inserted immediately before the ``Format`` transform,
    so the pasted boxes are converted/normalized by the existing machinery.
    """

    def __init__(
        self,
        crops,
        paste_prob=0.5,
        max_instances=3,
        min_size=8,
        max_size=40,
        white_thresh=245,
    ):
        self.crops = crops
        self.paste_prob = paste_prob
        self.max_instances = max_instances
        self.min_size = min_size
        self.max_size = max_size
        self.white_thresh = white_thresh

    def _crop_rgb_and_mask(self, crop, size):
        """Resize a crop to size x size and return (bgr uint8, alpha mask 0..1)."""
        import cv2

        if crop.ndim == 3 and crop.shape[2] == 4:
            bgr = crop[:, :, :3]
            alpha = crop[:, :, 3].astype(np.float32) / 255.0
        else:
            bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            # Treat near-white background as transparent.
            near_white = np.all(bgr >= self.white_thresh, axis=2)
            alpha = (~near_white).astype(np.float32)

        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
        return bgr, alpha[:, :, None]

    def __call__(self, labels):
        if not self.crops or random.random() > self.paste_prob:
            return labels

        # Lazy import so the module is importable without ultralytics on PATH.
        from ultralytics.utils.instance import Instances

        img = labels["img"]
        cls = labels["cls"]
        h, w = img.shape[:2]

        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        boxes = instances.bboxes  # (N, 4) pixel xyxy

        new_boxes = []
        n = random.randint(1, self.max_instances)
        max_sz = min(self.max_size, w - 1, h - 1)
        if max_sz < self.min_size:
            # Image smaller than our paste size; nothing sensible to do.
            labels["instances"] = instances
            return labels

        for _ in range(n):
            crop = random.choice(self.crops)
            size = random.randint(self.min_size, max_sz)
            x1 = random.randint(0, w - size)
            y1 = random.randint(0, h - size)
            x2, y2 = x1 + size, y1 + size

            patch, alpha = self._crop_rgb_and_mask(crop, size)
            roi = img[y1:y2, x1:x2].astype(np.float32)
            blended = roi * (1.0 - alpha) + patch.astype(np.float32) * alpha
            img[y1:y2, x1:x2] = blended.astype(img.dtype)

            new_boxes.append([x1, y1, x2, y2])

        if new_boxes:
            new_boxes = np.asarray(new_boxes, dtype=np.float32)
            all_boxes = np.concatenate([boxes, new_boxes], axis=0)
            new_cls = np.zeros((len(new_boxes), 1), dtype=cls.dtype)  # class 0 = drone
            labels["cls"] = np.concatenate([cls, new_cls], axis=0)
            # Empty segments (detection task) — sized to match box count.
            segments = np.zeros((len(all_boxes), 0, 2), dtype=np.float32)
            labels["instances"] = Instances(
                all_boxes,
                segments=segments,
                keypoints=None,
                bbox_format="xyxy",
                normalized=False,
            )
        else:
            labels["instances"] = instances

        labels["img"] = img
        return labels


def small_drone_paste(trainer):
    """
    ``on_train_epoch_start`` callback: ensure the paste transform is installed.

    Idempotent — only inserts the transform once, then refreshes the dataloader
    so its workers use the updated pipeline.
    """
    dataset = getattr(getattr(trainer, "train_loader", None), "dataset", None)
    transforms = getattr(dataset, "transforms", None)
    if transforms is None or not hasattr(transforms, "transforms"):
        return

    if any(isinstance(t, SmallDronePaste) for t in transforms.transforms):
        return  # already installed

    crops = _load_crops(_CONFIG["crops_dir"])
    if not crops:
        if getattr(trainer, "epoch", 0) == 0:
            print(
                f"[small_drone_paste] No crops found in '{_CONFIG['crops_dir']}'. "
                "Run build_drone_crops.py first; skipping paste augmentation."
            )
        return

    transform = SmallDronePaste(
        crops,
        paste_prob=_CONFIG["paste_prob"],
        max_instances=_CONFIG["max_instances"],
        min_size=_CONFIG["min_size"],
        max_size=_CONFIG["max_size"],
        white_thresh=_CONFIG["white_thresh"],
    )
    # Insert just before the final Format transform.
    insert_at = max(0, len(transforms.transforms) - 1)
    transforms.insert(insert_at, transform)

    # Refresh workers so the modified pipeline takes effect.
    loader = trainer.train_loader
    if hasattr(loader, "reset"):
        loader.reset()

    print(
        f"[small_drone_paste] Installed paste augmentation with {len(crops)} crops "
        f"(size {_CONFIG['min_size']}-{_CONFIG['max_size']}px, "
        f"p={_CONFIG['paste_prob']}, up to {_CONFIG['max_instances']}/image)."
    )

"""Helper to make a YOLO data.yaml resolvable by the YOLOv5 fork.

YOLOv5 resolves relative ``train``/``val`` paths against its own repo root
(``yolov5/``), not against the data.yaml's directory (which is how the
ultralytics YOLOv8 fork behaves). To keep the committed data.yaml portable
(no absolute paths), this helper writes a temporary copy with an absolute
``path:`` key injected, pointing at the original yaml's directory.
"""

import os
import tempfile


def materialize_data_yaml(data_path):
    """Return ``(yaml_path, tmp_to_cleanup)``.

    If the data.yaml already has an absolute ``path:``, the original file is
    returned unchanged and ``tmp_to_cleanup`` is ``None``. Otherwise a temp
    copy is written with an absolute ``path:`` and its path is returned as
    both values (delete it once YOLOv5 has finished).
    """
    abs_data = os.path.abspath(data_path)
    data_dir = os.path.dirname(abs_data)

    with open(abs_data) as f:
        lines = f.read().splitlines()

    out = []
    path_set = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("path:"):
            value = stripped[len("path:") :].split("#", 1)[0].strip().strip("\"'")
            if os.path.isabs(value):
                return abs_data, None  # already absolute; nothing to do
            resolved = (
                os.path.abspath(os.path.join(data_dir, value)) if value else data_dir
            )
            out.append(f"path: {resolved}")
            path_set = True
        else:
            out.append(line)

    if not path_set:
        out.insert(0, f"path: {data_dir}")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="yolov5_data_"
    )
    tmp.write("\n".join(out) + "\n")
    tmp.close()
    return tmp.name, tmp.name

"""Helper to locate a conda environment's Python interpreter portably.

Avoids hardcoding absolute paths (and usernames). Resolution order:
  1. An explicit environment-variable override, if provided and set.
  2. The active conda installation, located via CONDA_EXE / CONDA_PREFIX.
  3. Common install locations under the user's home directory.
Falls back to the current interpreter if nothing matches.
"""

import os
import sys


def conda_python(env_name, override_env_var=None):
    # 1. Explicit override (e.g. YOLOV8_PYTHON=/path/to/python)
    if override_env_var:
        override = os.environ.get(override_env_var)
        if override:
            return override

    rel = os.path.join("envs", env_name, "bin", "python")
    candidates = []

    # 2. Locate the conda base from the active installation
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        # <base>/bin/conda → <base>
        candidates.append(os.path.dirname(os.path.dirname(conda_exe)))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        # If already inside an env (…/envs/<x>), step up to the base
        if os.path.basename(os.path.dirname(conda_prefix)) == "envs":
            candidates.append(os.path.dirname(os.path.dirname(conda_prefix)))
        candidates.append(conda_prefix)

    # 3. Common install locations
    for name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        candidates.append(os.path.expanduser(os.path.join("~", name)))

    for base in candidates:
        path = os.path.join(base, rel)
        if os.path.exists(path):
            return path

    # Fallback: current interpreter
    return sys.executable

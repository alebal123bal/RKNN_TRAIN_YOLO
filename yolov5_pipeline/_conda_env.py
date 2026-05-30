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
    # 1. Explicit override (e.g. YOLOV5_PYTHON=/path/to/python)
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


def wsl_ld_prefix():
    """Return a shell prefix that exposes the WSL CUDA driver stubs.

    Under WSL, the NVIDIA driver libraries (e.g. ``libcuda.so``) live in
    ``/usr/lib/wsl/lib`` but are not on the default loader path, which breaks
    torch's bundled cuDNN. Returns a ``LD_LIBRARY_PATH=... `` prefix when that
    directory exists, or an empty string otherwise (no-op off WSL).
    """
    wsl_lib = "/usr/lib/wsl/lib"
    if os.path.isdir(wsl_lib):
        return f"LD_LIBRARY_PATH={wsl_lib}:$LD_LIBRARY_PATH "
    return ""

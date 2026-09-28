"""
edgelens.hardware.detector
---------------------------
Detects board identity and the AI software stack (JetPack/L4T, CUDA,
TensorRT, key Python packages) without depending on any AGPL-licensed
code (notably, this does NOT import jetson-stats — it reads the same
underlying /proc, /sys and /etc sources directly).

NOTE ON VALIDATION: the is_jetson()/detect_board()/detect_jetpack()/
detect_cuda() functions read real Jetson system files
(/proc/device-tree/model, /etc/nv_tegra_release, /usr/local/cuda/version.json)
per NVIDIA's documented layout. They have not been exercised against
physical Jetson hardware by the author of this scaffold — please
validate on your board and file an issue with your `edgelens doctor`
output if anything looks wrong.
"""

import json
import os
import platform
import subprocess
import sys


def _read_file(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def _run(cmd, timeout=5):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() or out.stderr.strip()
    except Exception:
        return None


def is_jetson() -> bool:
    """Best-effort Jetson detection. True only on real Jetson hardware."""
    model = _read_file("/proc/device-tree/model")
    if model and ("jetson" in model.lower() or "tegra" in model.lower()):
        return True
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    return False


def detect_board():
    model = _read_file("/proc/device-tree/model")
    l4t_raw = _read_file("/etc/nv_tegra_release")
    return {
        "is_jetson": is_jetson(),
        "model": model or f"Non-Jetson host ({platform.node()})",
        "l4t_release_raw": l4t_raw,
    }


def detect_jetpack():
    """Parses /etc/nv_tegra_release, e.g. '# R36 (release), REVISION: 4.3, ...'"""
    raw = _read_file("/etc/nv_tegra_release")
    if not raw:
        return None
    return raw.replace("# ", "").strip()


def detect_cuda():
    version_json = _read_file("/usr/local/cuda/version.json")
    if version_json:
        try:
            data = json.loads(version_json)
            return data.get("cuda", {}).get("version")
        except Exception:
            pass
    nvcc_out = _run(["nvcc", "--version"])
    if nvcc_out:
        for line in nvcc_out.splitlines():
            if "release" in line.lower():
                return line.strip()
    return None


def detect_tensorrt():
    try:
        import tensorrt as trt  # type: ignore
        return trt.__version__
    except Exception:
        return None


import importlib.metadata as md
import importlib.util

# import name -> candidate pip distribution names
_PKG_DISTS = {
    "torch": ["torch"],
    "onnx": ["onnx"],
    "onnxruntime": ["onnxruntime", "onnxruntime-gpu"],
    "numpy": ["numpy"],
    "cv2": ["opencv-python", "opencv-python-headless", "opencv-contrib-python"],
    "pycuda": ["pycuda"],
}


def detect_python_packages():
    """Report installed versions WITHOUT importing the packages.

    Importing can be slow (torch) or crash on ABI mismatches (e.g. a
    NumPy-1.x-built onnxruntime under NumPy 2). Metadata lookup avoids both.
    """
    packages = {}
    for import_name, dists in _PKG_DISTS.items():
        version = None
        for dist in dists:
            try:
                version = md.version(dist)
                break
            except md.PackageNotFoundError:
                continue
        if version is None and importlib.util.find_spec(import_name) is not None:
            # installed without pip metadata (e.g. OpenCV from apt on Jetson)
            version = "installed (version unknown)"
        packages[import_name] = version
    return packages


def full_fingerprint():
    board = detect_board()
    software = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "jetpack_l4t": detect_jetpack(),
        "cuda": detect_cuda(),
        "tensorrt": detect_tensorrt(),
        "packages": detect_python_packages(),
    }
    return {"hardware": board, "software": software}

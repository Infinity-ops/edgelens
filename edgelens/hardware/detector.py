"""
edgelens.hardware.detector
---------------------------
Detects board identity and the AI software stack (JetPack/L4T, CUDA,
TensorRT, key Python packages) without depending on any AGPL-licensed
code (notably, this does NOT import jetson-stats — it reads the same
underlying /proc, /sys and /etc sources directly).

Validation: these readers (/proc/device-tree/model, /etc/nv_tegra_release,
/usr/local/cuda/version.json, dpkg/header/library TensorRT fallbacks) are
verified on a Jetson Nano (JetPack 4.6 / L4T R32.7.6). The compatible /
soc0 / versioned-CUDA / /usr/lib fallbacks come from an AGX Orin 32GB on a
custom carrier running a Yocto (OE4Tegra) rootfs. Other modules follow
NVIDIA's documented layout but are not yet verified — please open an issue
with your `edgelens doctor` output if anything looks wrong.
"""

import glob
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys

# import name -> candidate pip distribution names. Package versions are
# read from install metadata, NOT by importing the package — importing
# can be slow (torch) or crash outright on ABI mismatches (e.g. a
# NumPy-1.x-built onnxruntime wheel under NumPy 2, seen in the field).
_PKG_DISTS = {
    "torch": ["torch"],
    "onnx": ["onnx"],
    "onnxruntime": ["onnxruntime", "onnxruntime-gpu"],
    "numpy": ["numpy"],
    "cv2": ["opencv-python", "opencv-python-headless", "opencv-contrib-python"],
    "pycuda": ["pycuda"],
}


def _read_file(path):
    try:
        with open(path, "r") as f:
            # Device-tree strings end in a NUL byte, which .strip() keeps:
            # on a real Nano the board name was stored as "...Kit\u0000" in
            # every result JSON and printed glued to the shell prompt.
            return f.read().replace("\x00", "").strip()
    except Exception:
        return None


def _run(cmd, timeout=5):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() or out.stderr.strip()
    except Exception:
        return None


# Module part numbers in /proc/device-tree/compatible -> marketing name.
# The compatible string survives custom carrier boards and non-Ubuntu
# rootfs images (Yocto/meta-tegra), where the model string is the
# integrator's own ("Taipur Orin") and /etc/nv_tegra_release is absent.
_MODULE_NAMES = {
    "p3701-0000": "Jetson AGX Orin",
    "p3701-0004": "Jetson AGX Orin 32GB",
    "p3701-0005": "Jetson AGX Orin 64GB",
    "p3701-0008": "Jetson AGX Orin Industrial",
    "p3767-0000": "Jetson Orin NX 16GB",
    "p3767-0001": "Jetson Orin NX 8GB",
    "p3767-0003": "Jetson Orin Nano 8GB",
    "p3767-0004": "Jetson Orin Nano 4GB",
    "p3767-0005": "Jetson Orin Nano 8GB (devkit)",
    "p3448-0000": "Jetson Nano (devkit)",
    "p3448-0002": "Jetson Nano (eMMC)",
}


def _read_compatible():
    """/proc/device-tree/compatible: NUL-separated, most specific first."""
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            raw = f.read()
    except Exception:
        return []
    return [c for c in raw.decode("ascii", "replace").split("\x00") if c]


def detect_soc():
    """'tegra234' / 'tegra210' / ... from the compatible list, else None."""
    for entry in _read_compatible():
        m = re.match(r"nvidia,(tegra\d+)$", entry)
        if m:
            return m.group(1)
    return None


def detect_module():
    """Jetson module name from its part number, e.g. 'Jetson AGX Orin 32GB'."""
    for entry in _read_compatible():
        m = re.match(r"nvidia,(p\d{4}-\d{4})$", entry)
        if m and m.group(1) in _MODULE_NAMES:
            return _MODULE_NAMES[m.group(1)]
    return None


def is_jetson() -> bool:
    """Best-effort Jetson detection. True only on real Jetson hardware.

    Checks, in order: the board model string (NVIDIA devkits), the L4T
    release file (NVIDIA's Ubuntu rootfs), the device-tree compatible list
    and the SoC family in sysfs. The last two also catch custom carrier
    boards running Yocto/meta-tegra, seen on an AGX Orin 32GB whose model
    string is the integrator's own and whose rootfs has no nv_tegra_release.
    """
    model = _read_file("/proc/device-tree/model")
    if model and ("jetson" in model.lower() or "tegra" in model.lower()):
        return True
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    if detect_soc() is not None:
        return True
    family = _read_file("/sys/devices/soc0/family")
    if family and family.strip().lower() == "tegra":
        return True
    return False


def detect_board():
    model = _read_file("/proc/device-tree/model")
    l4t_raw = _read_file("/etc/nv_tegra_release")
    return {
        "is_jetson": is_jetson(),
        "model": model or f"Non-Jetson host ({platform.node()})",
        "module": detect_module(),
        "soc": detect_soc(),
        "l4t_release_raw": l4t_raw,
    }


def detect_jetpack():
    """Parses /etc/nv_tegra_release, e.g. '# R36 (release), REVISION: 4.3, ...'"""
    raw = _read_file("/etc/nv_tegra_release")
    if not raw:
        return None
    return raw.replace("# ", "").strip()


def _cuda_version_json_paths():
    """/usr/local/cuda first, then versioned dirs newest first. Yocto
    (meta-tegra) installs /usr/local/cuda-12.6 without the symlink."""
    paths = ["/usr/local/cuda/version.json"]
    versioned = glob.glob("/usr/local/cuda-*/version.json")

    def _key(path):
        m = re.search(r"cuda-(\d+)\.(\d+)", path)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    paths += sorted(versioned, key=_key, reverse=True)
    return paths


def detect_cuda():
    for path in _cuda_version_json_paths():
        version_json = _read_file(path)
        if not version_json:
            continue
        try:
            data = json.loads(version_json)
            version = data.get("cuda", {}).get("version")
            if version:
                return version
        except Exception:
            pass
    nvcc_out = _run(["nvcc", "--version"])
    if nvcc_out:
        for line in nvcc_out.splitlines():
            if "release" in line.lower():
                return line.strip()
    # Runtime-only installs (Yocto/meta-tegra images, containers) can lack
    # version.json and nvcc. Fall back to the libcudart file name, then to
    # the versioned install directory. Only reached when everything above
    # found nothing, so hosts that were detected before are unaffected.
    return _cuda_version_from_cudart() or _cuda_version_from_dir()


_CUDART_GLOBS = (
    "/usr/local/cuda*/lib64/libcudart.so.*",
    "/usr/local/cuda*/targets/*/lib/libcudart.so.*",
    "/usr/lib/aarch64-linux-gnu/libcudart.so.*",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.*",
    "/usr/lib/libcudart.so.*",
)


def _cuda_version_from_cudart():
    """'12.6.68' from libcudart.so.12.6.68 (longest version wins)."""
    best = None
    for pattern in _CUDART_GLOBS:
        for path in glob.glob(pattern):
            m = re.search(r"libcudart\.so\.(\d+(?:\.\d+)*)$", path)
            if m and (best is None or len(m.group(1)) > len(best)):
                best = m.group(1)
    # "12" alone (the soname link) says less than the directory name does
    return best if best and "." in best else None


def _cuda_version_from_dir():
    """'12.6' from /usr/local/cuda-12.6 (newest if several)."""
    found = []
    for path in glob.glob("/usr/local/cuda-*"):
        m = re.search(r"/cuda-(\d+)\.(\d+)$", path)
        if m and os.path.isdir(path):
            found.append((int(m.group(1)), int(m.group(2))))
    if not found:
        return None
    major, minor = max(found)
    return "%d.%d" % (major, minor)


# Where the apt-installed TensorRT C++ library and header live on Jetson.
_TRT_HEADER_CANDIDATES = (
    "/usr/include/aarch64-linux-gnu/NvInferVersion.h",   # JetPack 4.x / 5.x / 6.x
    "/usr/include/x86_64-linux-gnu/NvInferVersion.h",    # x86 apt installs
    "/usr/local/cuda/include/NvInferVersion.h",
    "/usr/include/NvInferVersion.h",                     # Yocto / meta-tegra
)
_TRT_LIB_GLOBS = (
    "/usr/lib/aarch64-linux-gnu/libnvinfer.so.*",
    "/usr/lib/x86_64-linux-gnu/libnvinfer.so.*",
    "/usr/lib/libnvinfer.so.*",                          # Yocto / meta-tegra
)
# apt package names that carry the TensorRT version, most specific first.
_TRT_DPKG_PACKAGES = ("tensorrt", "libnvinfer10", "libnvinfer8", "libnvinfer7")


def _trt_version_from_python():
    """Layer 1: import the Python bindings. Only works when the bindings
    were built for THIS interpreter's ABI (on JetPack 4.x that means the
    system python3.6 only — a Python 3.8 venv can never import them)."""
    try:
        import tensorrt as trt  # type: ignore
        return trt.__version__
    except Exception:
        return None


def _trt_version_from_dpkg():
    """Layer 2: ask dpkg — no import, works from any Python version."""
    for pkg in _TRT_DPKG_PACKAGES:
        try:
            out = subprocess.run(
                ["dpkg-query", "-W", "-f=${Status}|${Version}", pkg],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return None  # no dpkg on this host at all
        if out.returncode != 0 or "|" not in out.stdout:
            continue
        status, version = out.stdout.strip().split("|", 1)
        if not status.endswith(" installed"):  # skip half-installed / config-files
            continue
        # "8.2.1.8-1+cuda10.2" -> "8.2.1.8"
        version = version.split("-", 1)[0].strip()
        if version:
            return version
    return None


def _trt_version_from_header():
    """Layer 3: parse NvInferVersion.h (installed by libnvinfer-dev)."""
    for path in _TRT_HEADER_CANDIDATES:
        text = _read_file(path)
        if not text:
            continue
        parts = {}
        for key in ("MAJOR", "MINOR", "PATCH", "BUILD"):
            m = re.search(r"#define\s+NV_TENSORRT_%s\s+(\d+)" % key, text)
            if m:
                parts[key] = m.group(1)
        if "MAJOR" in parts and "MINOR" in parts:
            return ".".join(parts[k] for k in ("MAJOR", "MINOR", "PATCH", "BUILD") if k in parts)
    return None


def _trt_version_from_lib():
    """Layer 4: read the version off the libnvinfer.so.X.Y.Z file name."""
    best = None
    for pattern in _TRT_LIB_GLOBS:
        for path in glob.glob(pattern):
            m = re.search(r"libnvinfer\.so\.(\d+(?:\.\d+)*)$", path)
            if m and (best is None or len(m.group(1)) > len(best)):
                best = m.group(1)
    return best


def detect_tensorrt_info():
    """Find TensorRT WITHOUT requiring importable Python bindings.

    Returns a dict:
        version          -> str or None
        source           -> "python" | "dpkg" | "header" | "lib" | None
        python_bindings  -> True if `import tensorrt` works in THIS interpreter

    Why: on JetPack 4.x (Jetson Nano, L4T R32) TensorRT ships via apt and
    its Python bindings are compiled only for the system python3.6. In a
    Python 3.8 venv `import tensorrt` correctly fails, but the TensorRT
    C++ runtime (libnvinfer) is still installed and usable — e.g. by an
    onnxruntime-gpu build with the TensorRT execution provider.
    """
    version = _trt_version_from_python()
    if version:
        return {"version": version, "source": "python", "python_bindings": True}
    for source, fn in (("dpkg", _trt_version_from_dpkg),
                       ("header", _trt_version_from_header),
                       ("lib", _trt_version_from_lib)):
        version = fn()
        if version:
            return {"version": version, "source": source, "python_bindings": False}
    return {"version": None, "source": None, "python_bindings": False}


def detect_tensorrt():
    """Backward-compatible: just the version string (or None)."""
    return detect_tensorrt_info()["version"]


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
    trt = detect_tensorrt_info()
    software = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "jetpack_l4t": detect_jetpack(),
        "cuda": detect_cuda(),
        "tensorrt": trt["version"],
        "tensorrt_python_bindings": trt["python_bindings"],
        "tensorrt_source": trt["source"],
        "packages": detect_python_packages(),
    }
    return {"hardware": board, "software": software}

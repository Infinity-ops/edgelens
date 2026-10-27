"""Detection on a custom-carrier Jetson running a Yocto (OE4Tegra) rootfs.

Reproduces a real AGX Orin 32GB on a custom carrier ("Example Orin" here): model string has
no "jetson", no /etc/nv_tegra_release, CUDA only at /usr/local/cuda-12.6,
TensorRT libs in /usr/lib, no dpkg. v0.1.0 reported "Is Jetson: no" and
missed CUDA and TensorRT there.
"""
import builtins
import io
from unittest.mock import patch

from edgelens.hardware import detector

COMPATIBLE = b"nvidia,p3737-0000+p3701-0004\x00nvidia,p3701-0004\x00nvidia,tegra234\x00"
FILES = {
    "/proc/device-tree/model": "Example Orin\x00",
    "/sys/devices/soc0/family": "Tegra\n",
    "/usr/local/cuda-12.6/version.json": '{"cuda": {"name": "CUDA SDK", "version": "12.6.68"}}',
}
GLOBS = {
    "/usr/local/cuda-*/version.json": ["/usr/local/cuda-12.6/version.json"],
    "/usr/lib/libnvinfer.so.*": ["/usr/lib/libnvinfer.so.10", "/usr/lib/libnvinfer.so.10.3.0"],
}
_real_open = builtins.open


def _fake_open(path, mode="r", *a, **kw):
    if path == "/proc/device-tree/compatible":
        return io.BytesIO(COMPATIBLE)
    if path in FILES:
        return io.StringIO(FILES[path])
    if isinstance(path, str) and path.startswith(("/proc/", "/sys/", "/etc/", "/usr/")):
        raise FileNotFoundError(path)
    return _real_open(path, mode, *a, **kw)


def _yocto_orin():
    return [
        patch.object(builtins, "open", _fake_open),
        patch.object(detector.os.path, "exists", lambda p: False),
        patch.object(detector.glob, "glob", lambda pat: GLOBS.get(pat, [])),
        patch.object(detector.subprocess, "run", side_effect=FileNotFoundError),
        patch.object(detector, "_trt_version_from_python", lambda: None),
    ]


def _run(fn):
    ps = _yocto_orin()
    for p in ps:
        p.start()
    try:
        return fn()
    finally:
        for p in ps:
            p.stop()


def test_is_jetson_from_compatible():
    assert _run(detector.is_jetson) is True


def test_module_and_soc():
    board = _run(detector.detect_board)
    assert board["model"] == "Example Orin"
    assert board["module"] == "Jetson AGX Orin 32GB"
    assert board["soc"] == "tegra234"


def test_cuda_from_versioned_dir():
    assert _run(detector.detect_cuda) == "12.6.68"


def test_tensorrt_from_usr_lib():
    info = _run(detector.detect_tensorrt_info)
    assert info["version"] == "10.3.0"
    assert info["source"] == "lib"


def test_no_l4t_file_is_none_not_crash():
    assert _run(detector.detect_jetpack) is None


def test_soc0_family_alone_is_enough():
    def no_compat(path, mode="r", *a, **kw):
        if path == "/proc/device-tree/compatible":
            raise FileNotFoundError(path)
        return _fake_open(path, mode, *a, **kw)
    ps = _yocto_orin()
    ps[0] = patch.object(builtins, "open", no_compat)
    for p in ps:
        p.start()
    try:
        assert detector.is_jetson() is True
        assert detector.detect_soc() is None
    finally:
        for p in ps:
            p.stop()


def test_cuda_runtime_only_install_falls_back_to_cudart():
    files = {k: v for k, v in FILES.items() if "cuda" not in k}
    globs = dict(GLOBS)
    globs["/usr/local/cuda-*/version.json"] = []
    globs["/usr/local/cuda*/targets/*/lib/libcudart.so.*"] = [
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12",
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12.6.68",
    ]
    with patch.object(builtins, "open", lambda p, m="r", *a, **k: io.StringIO(files[p]) if p in files else (_ for _ in ()).throw(FileNotFoundError(p))), \
         patch.object(detector.glob, "glob", lambda pat: globs.get(pat, [])), \
         patch.object(detector.subprocess, "run", side_effect=FileNotFoundError):
        assert detector.detect_cuda() == "12.6.68"


def test_cuda_dir_name_is_the_last_resort():
    with patch.object(builtins, "open", side_effect=FileNotFoundError), \
         patch.object(detector.glob, "glob",
                      lambda pat: ["/usr/local/cuda-11.4", "/usr/local/cuda-12.6"] if pat == "/usr/local/cuda-*" else []), \
         patch.object(detector.os.path, "isdir", lambda p: True), \
         patch.object(detector.subprocess, "run", side_effect=FileNotFoundError):
        assert detector.detect_cuda() == "12.6"

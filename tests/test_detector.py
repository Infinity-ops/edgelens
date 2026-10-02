"""Tests for TensorRT detection on hosts where the Python bindings are
NOT importable (e.g. Jetson Nano / JetPack 4.x running a Python 3.8 venv,
where apt's TensorRT bindings exist only for the system python3.6)."""

import subprocess
from unittest.mock import patch

from edgelens.hardware import detector


def _fake_dpkg(version_by_pkg):
    def run(cmd, **kwargs):
        pkg = cmd[-1]
        if pkg in version_by_pkg:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="install ok installed|%s" % version_by_pkg[pkg], stderr="")
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="dpkg-query: no packages found matching %s" % pkg)
    return run


def test_python_bindings_win_when_importable():
    with patch.object(detector, "_trt_version_from_python", return_value="10.3.0"):
        info = detector.detect_tensorrt_info()
    assert info == {"version": "10.3.0", "source": "python", "python_bindings": True}


def test_jetson_nano_py38_falls_back_to_dpkg():
    with patch.object(detector, "_trt_version_from_python", return_value=None), \
         patch.object(detector.subprocess, "run",
                      side_effect=_fake_dpkg({"tensorrt": "8.2.1.8-1+cuda10.2"})):
        info = detector.detect_tensorrt_info()
    assert info["version"] == "8.2.1.8"
    assert info["source"] == "dpkg"
    assert info["python_bindings"] is False


def test_dpkg_error_text_is_never_parsed_as_a_version():
    with patch.object(detector, "_trt_version_from_python", return_value=None), \
         patch.object(detector.subprocess, "run", side_effect=_fake_dpkg({})), \
         patch.object(detector, "_trt_version_from_header", return_value=None), \
         patch.object(detector, "_trt_version_from_lib", return_value=None):
        info = detector.detect_tensorrt_info()
    assert info == {"version": None, "source": None, "python_bindings": False}


def test_header_fallback(tmp_path):
    header = tmp_path / "NvInferVersion.h"
    header.write_text(
        "#define NV_TENSORRT_MAJOR 8 //!< TensorRT major version.\n"
        "#define NV_TENSORRT_MINOR 2\n"
        "#define NV_TENSORRT_PATCH 1\n"
        "#define NV_TENSORRT_BUILD 8\n"
    )
    with patch.object(detector, "_TRT_HEADER_CANDIDATES", (str(header),)):
        assert detector._trt_version_from_header() == "8.2.1.8"


def test_lib_filename_fallback(tmp_path):
    (tmp_path / "libnvinfer.so.8").write_text("")
    (tmp_path / "libnvinfer.so.8.2.1").write_text("")
    with patch.object(detector, "_TRT_LIB_GLOBS", (str(tmp_path / "libnvinfer.so.*"),)):
        assert detector._trt_version_from_lib() == "8.2.1"


def test_no_dpkg_binary_does_not_crash():
    with patch.object(detector.subprocess, "run", side_effect=FileNotFoundError):
        assert detector._trt_version_from_dpkg() is None


def test_fingerprint_exposes_binding_status():
    with patch.object(detector, "detect_tensorrt_info",
                      return_value={"version": "8.2.1.8", "source": "dpkg",
                                    "python_bindings": False}):
        sw = detector.full_fingerprint()["software"]
    assert sw["tensorrt"] == "8.2.1.8"
    assert sw["tensorrt_python_bindings"] is False
    assert sw["tensorrt_source"] == "dpkg"

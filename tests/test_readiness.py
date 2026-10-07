"""Benchmark readiness (`edgelens doctor`). Inputs are the real states seen
on a Jetson Nano (JetPack 4.6) during v0.1.0 validation."""
from typer.testing import CliRunner

from edgelens.cli import app
from edgelens.hardware import readiness

NANO_CHMOD = ("sudo chmod o+r /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/rail_name_* "
              "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power*_input")


def _by_name(checks):
    return {c["name"]: c for c in checks}


def test_real_nano_before_fixes_has_three_warnings_with_copyable_fixes():
    # unpinned clocks, PyPI onnxruntime over the GPU build, root-only INA3221
    checks = readiness.assess(
        is_jetson=True, power_mode={"id": 0, "name": "MAXN"}, cpu_pinned=False, gpu_pinned=False,
        providers=["AzureExecutionProvider", "CPUExecutionProvider"],
        power_probe={"total_w": None, "permission_denied": True, "fix": NANO_CHMOD})
    c = _by_name(checks)
    assert c["Power mode"]["value"] == "MAXN"
    assert c["Clocks pinned"]["status"] == "warn"
    assert c["Clocks pinned"]["fix"] == "sudo jetson_clocks --store && sudo jetson_clocks"
    assert c["GPU inference"]["status"] == "warn"
    assert c["GPU inference"]["fix"] == "pip uninstall -y onnxruntime"
    assert c["Power sensor"]["fix"] == NANO_CHMOD
    status, text = readiness.summary(checks)
    assert status == "warn" and text.startswith("3 item(s)")


def test_real_nano_after_fixes_is_ready():
    checks = readiness.assess(
        is_jetson=True, power_mode={"id": 0, "name": "MAXN"}, cpu_pinned=True, gpu_pinned=True,
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        power_probe={"total_w": 3.58, "method": "input_rail"})
    c = _by_name(checks)
    assert c["GPU inference"]["value"] == "Tensorrt, CUDA"
    assert c["Power sensor"]["value"] == "3.58 W (input_rail)"
    assert readiness.summary(checks)[0] == "ok"


def test_partially_pinned_is_still_a_warning():
    # e.g. the Nano's 5W mode pins the CPU but not the GPU
    checks = readiness.assess(True, {"id": 1, "name": "5W"}, True, False,
                              ["CUDAExecutionProvider"], {"total_w": 2.1, "method": "input_rail"})
    assert _by_name(checks)["Clocks pinned"]["value"] == "no (GPU scaling)"


def test_non_jetson_makes_no_board_claims():
    checks = readiness.assess(False, None, None, None, ["CPUExecutionProvider"], {})
    names = [c["name"] for c in checks]
    assert names == ["GPU inference"]                    # no clocks/power-mode/sensor checks
    assert _by_name(checks)["GPU inference"]["status"] == "info"   # CPU-only is normal here
    status, text = readiness.summary(checks, is_jetson=False)
    assert status == "info" and "Not a Jetson" in text


def test_without_onnxruntime_is_informational():
    checks = readiness.assess(False, None, None, None, [], {})
    assert _by_name(checks)["ONNX Runtime"]["status"] == "info"


def test_doctor_prints_readiness():
    res = CliRunner().invoke(app, ["doctor"])
    assert res.exit_code == 0, res.output
    assert "Benchmark readiness" in res.output

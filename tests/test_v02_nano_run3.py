"""Third Jetson Nano run: a 30-minute DEM2 soak at 50 C reported thermal
throttling in 30 of 30 windows. The 'active' cooling devices were Tegra voltage
floors and voltage-temperature scaling, which never lower a clock."""

import json

from edgelens import soak as sk

NANO_COOLING = {"CORE-floor": 1, "DFLL-floor": 1, "vdd-gpu-vts": 2}


def _rows(n=30, cooling=None, cap_drop_at=None):
    rows = []
    for i in range(n):
        rows.append({"kind": "window", "window": i, "t_start_s": i * 60.0,
                     "t_end_s": (i + 1) * 60.0, "iterations": 60, "p50_ms": 10.0,
                     "p99_ms": 12.0, "max_temp_c": 50.0, "rss_mb": 300.0,
                     "cooling_active": dict(cooling or {}),
                     "cpu_cap_mhz_min": 1479.0 if cap_drop_at is None or i < cap_drop_at
                     else 921.0,
                     "gpu_cap_mhz_min": 921.6})
    return rows


def test_tegra_voltage_floors_and_vts_are_not_throttling():
    a = sk.analyze(_rows(cooling=NANO_COOLING))
    assert a["throttling"]["throttled_windows"] == 0
    for name in NANO_COOLING:
        assert not sk.throttling_cooling_device(name)
    for name in ("pwm-fan", "Fan"):
        assert not sk.throttling_cooling_device(name)


def test_real_throttling_devices_and_cap_drops_still_count():
    for name in ("gpu-balanced", "cpu-balanced", "emergency-balanced", "thermal-cpufreq-0",
                 "CORE-cap", "GPU-cap"):
        assert sk.throttling_cooling_device(name), name
    a = sk.analyze(_rows(cooling=dict(NANO_COOLING, **{"gpu-balanced": 1})))
    assert a["throttling"]["throttled_windows"] == 30
    assert "gpu-balanced" in a["throttling"]["windows"][0]["reasons"][0]
    assert "floor" not in a["throttling"]["windows"][0]["reasons"][0]
    # no device at all, but the CPU cap fell: still throttling
    a = sk.analyze(_rows(cooling=NANO_COOLING, cap_drop_at=20))
    assert a["throttling"]["throttled_windows"] == 10


def test_soak_report_reanalyses_an_old_jsonl_correctly(tmp_path):
    # the .jsonl written by the run that reported 30/30: `soak --report` fixes the verdict
    path = tmp_path / "soak_dem2.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "header", "deadline_ms": 3000, "limits": {}}) + "\n")
        for r in _rows(cooling=NANO_COOLING):
            f.write(json.dumps(r) + "\n")
    r = sk.report_from_jsonl(str(path))
    assert r["soak"]["throttling"]["throttled_windows"] == 0
    from edgelens.validate.engine import validate
    v = validate(r, {"max_throttled_windows": 0})
    assert v["checks"][0]["status"] == "PASS"


def test_progress_line_does_not_shout_for_voltage_floors():
    row = _rows(1, cooling=NANO_COOLING)[0]
    assert "ACTIVE" not in sk._progress_line(row, 1800)
    row["cooling_active"]["gpu-balanced"] = 1
    assert "THROTTLING DEVICE ACTIVE" in sk._progress_line(row, 1800)


def test_tests_do_not_read_the_real_kernel_log():
    from edgelens.hardware.storage import kernel_storage_errors
    assert kernel_storage_errors()["count"] == 0        # conftest: clean log

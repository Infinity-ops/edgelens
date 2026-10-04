"""Power (INA3221) and telemetry sampling tests, using fake sysfs trees
shaped like the real Jetson Nano (JetPack 4, iio) and Orin (JetPack 5/6,
hwmon) layouts."""
import time

import pytest

from edgelens.hardware import power, telemetry

NANO_LINE = ("RAM 2246/3964MB (lfb 4x2MB) SWAP 0/1982MB (cached 0MB) "
             "CPU [35%@1479,12%@1479,8%@1479,30%@1479] EMC_FREQ 14%@1600 "
             "GR3D_FREQ 48%@921 APE 25 PLL@41C CPU@43.5C PMIC@50C GPU@41.5C "
             "AO@50.5C thermal@42.25C POM_5V_IN 4321/4105 POM_5V_GPU 1102/980 "
             "POM_5V_CPU 1440/1300")
ORIN_LINE = ("RAM 5108/62841MB (lfb 9x4MB) CPU [1%@729,0%@729] EMC_FREQ 0%@2133 "
             "GR3D_FREQ 61%@[1300,1300] cpu@46.8C gpu@44.6C VDD_GPU_SOC 9387mW/9387mW "
             "VDD_CPU_CV 1597mW/1597mW VIN_SYS_5V0 4422mW/4422mW")


def test_parse_nano_tegrastats_line():
    p = telemetry.parse_tegrastats_line(NANO_LINE)
    assert p["gpu_percent"] == 48.0
    assert p["emc_percent"] == 14.0
    assert p["rails_mw"] == {"POM_5V_IN": 4321.0, "POM_5V_GPU": 1102.0, "POM_5V_CPU": 1440.0}


def test_parse_orin_tegrastats_line_and_ram_is_not_a_rail():
    p = telemetry.parse_tegrastats_line(ORIN_LINE)
    assert p["gpu_percent"] == 61.0
    assert set(p["rails_mw"]) == {"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"}
    assert "RAM" not in p["rails_mw"]


def _make_nano_iio(root):
    dev = root / "ina3221x" / "6-0040" / "iio:device0"
    dev.mkdir(parents=True)
    for i, (name, mw) in enumerate([("POM_5V_IN", 4000), ("POM_5V_GPU", 1000), ("POM_5V_CPU", 1500)]):
        (dev / f"rail_name_{i}").write_text(name)
        (dev / f"in_power{i}_input").write_text(str(mw))
    return str(root / "ina3221x" / "*" / "iio:device*")


def _make_orin_hwmon(root):
    dev = root / "ina3221" / "1-0040" / "hwmon" / "hwmon3"
    dev.mkdir(parents=True)
    for i, (name, mv, ma) in enumerate([("VDD_GPU_SOC", 5000, 2000), ("VDD_CPU_CV", 5000, 400),
                                        ("VIN_SYS_5V0", 5000, 800), ("NC", 0, 0)], start=1):
        (dev / f"in{i}_label").write_text(name)
        (dev / f"in{i}_input").write_text(str(mv))
        (dev / f"curr{i}_input").write_text(str(ma))
    return str(root / "ina3221" / "*" / "hwmon" / "hwmon*")


def test_nano_iio_power_uses_input_rail(tmp_path):
    reader = power.PowerReader(iio_glob=_make_nano_iio(tmp_path), hwmon_glob=str(tmp_path / "none*"))
    rails = reader.read()
    assert rails == {"POM_5V_IN": 4.0, "POM_5V_GPU": 1.0, "POM_5V_CPU": 1.5}
    assert power.total_power(rails) == (4.0, "input_rail")


def test_orin_hwmon_power_sums_rails_and_skips_nc(tmp_path):
    reader = power.PowerReader(iio_glob=str(tmp_path / "none*"), hwmon_glob=_make_orin_hwmon(tmp_path))
    rails = reader.read()
    assert "NC" not in rails
    assert rails["VDD_GPU_SOC"] == pytest.approx(10.0)
    total, method = power.total_power(rails)
    assert method == "sum_of_rails"
    assert total == pytest.approx(10.0 + 2.0 + 4.0)


def test_no_ina3221_means_unavailable(tmp_path):
    reader = power.PowerReader(iio_glob=str(tmp_path / "x*"), hwmon_glob=str(tmp_path / "y*"))
    assert not reader.available()
    assert reader.read() == {}


def test_energy_is_trapezoidal_integral():
    samples = [{"timestamp": t, "power_w": p, "rails_w": {"POM_5V_IN": p}}
               for t, p in [(0.0, 4.0), (1.0, 6.0), (2.0, 6.0)]]
    s = power.summarize_power(samples)
    assert s["energy_j"] == pytest.approx(5.0 + 6.0)
    assert s["power_w_mean"] == pytest.approx(16.0 / 3, abs=1e-3)
    assert s["method"] == "input_rail"


def test_snapshot_is_non_blocking():
    t0 = time.perf_counter()
    for _ in range(5):
        telemetry.snapshot(stream=None, power_reader=power.PowerReader())
    # the old implementation blocked 200 ms per sample in cpu_percent()
    assert (time.perf_counter() - t0) < 0.5


def test_recorder_keeps_timestamped_series():
    rec = telemetry.TelemetryRecorder(interval_s=0.02, use_tegrastats=False)
    rec.start()
    time.sleep(0.15)
    summary = rec.stop()
    series = rec.series()
    assert summary["sample_count"] >= 3
    assert len(series) == summary["sample_count"]
    assert series[0]["t_s"] >= 0
    assert all(b["t_s"] >= a["t_s"] for a, b in zip(series, series[1:]))
    assert summary["sampler_cpu_ms_mean"] is not None
    assert summary["sampler_wall_ms_mean"] >= 0
    assert "power" in summary


def test_listed_but_unreadable_power_falls_back(tmp_path):
    # rails exist in sysfs but every read fails -> recorder must not keep a
    # dead reader (it would silently disable power for the whole run)
    dev = tmp_path / "ina3221x" / "6-0040" / "iio:device0"
    dev.mkdir(parents=True)
    (dev / "rail_name_0").write_text("POM_5V_IN")
    (dev / "in_power0_input").write_text("garbage")
    reader = power.PowerReader(iio_glob=str(tmp_path / "ina3221x" / "*" / "iio:device*"),
                               hwmon_glob=str(tmp_path / "none*"))
    assert reader.available() and reader.read() == {}
    rec = telemetry.TelemetryRecorder(interval_s=0.05, use_tegrastats=False)
    rec._power = reader
    rec.start()
    rec.stop()
    assert rec._power is None


def test_probe_sources_shape():
    src = telemetry.probe_sources(tegrastats_timeout_s=0.1)
    assert set(src) == {"gpu_load_sysfs", "power_ina3221", "thermal_zones", "tegrastats"}
    assert "read_ms" in src["power_ina3221"]

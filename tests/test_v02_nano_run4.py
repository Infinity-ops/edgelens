"""Fourth Jetson Nano run (and the laptop run the same morning)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from edgelens import contract as ct
from edgelens import periodic as per
from edgelens import sweep as sw
from edgelens import timeline as tl
from edgelens.hardware import storage

ROOT = Path(__file__).resolve().parents[1]


# ---- laptop: a 1.0 s paced run measured as 0.9 s failed "min_duration: 1s"

def test_run_duration_is_the_run_window_not_the_sampler_span():
    r = {"iterations": 101, "fps": 100.0, "throughput_basis": "observed rate over the run",
         "telemetry": {"duration_s": 0.9}}
    assert ct.run_duration_s(r) == pytest.approx(1.01)
    # back-to-back runs (fps = 1 / mean latency) still use the telemetry span
    r2 = dict(r, throughput_basis="1 / mean latency (iterations back to back)")
    assert ct.run_duration_s(r2) == 0.9


# ---- Nano: a "dla" sweep row with GPU numbers on a board that has no DLA

def test_dla_backend_is_unavailable_without_a_dla(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(sw, "dla_engines", lambda: 0)
    summary, _ = sw.run_sweep(str(ROOT / "tests/fixtures/lstm_anomaly.onnx"), ["cpu", "dla"],
                              iterations=5, eval_samples=4)
    row = next(r for r in summary["backends"] if r["backend"] == "dla")
    assert row["status"] == "unavailable" and "no DLA" in row["reason"]


def test_dla_engines_is_zero_off_jetson():
    assert sw.dla_engines() == 0 or os.path.exists("/dev/nvhost-nvdla0") or \
        os.path.exists("/dev/nvhost-ctrl-nvdla0")


# ---- Nano: "extra energy per inference 0.0 mJ" at 1 Hz was below what the sensor resolves

def test_extra_energy_below_resolution_is_a_bound_not_zero():
    res = {"energy": {"available": True, "power_w_mean": 2.812}}
    row = per._energy_row(res, 1000.0, idle_w=2.810, n=30)
    assert row["dynamic_energy_per_inference_mj"] is None
    assert row["dynamic_energy_below_mj"] == pytest.approx(28.1, abs=0.1)   # 1% of 2.81 W x 1 s
    big = per._energy_row({"energy": {"available": True, "power_w_mean": 2.95}}, 1000.0,
                          idle_w=2.80, n=30)
    assert big["dynamic_energy_per_inference_mj"] == pytest.approx(150.0)


# ---- Nano: 20-27 % background on every check, and no way to see what it was

def test_background_load_names_the_busiest_process():
    from edgelens.hardware.telemetry import TelemetryRecorder
    import time
    hog = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        time.sleep(0.3)
        r = TelemetryRecorder(interval_s=0.2, use_tegrastats=False, use_power=False)
        r.start()
        time.sleep(1.2)
        bg = r.stop()["background"]
    finally:
        hog.kill()
        hog.wait()
    # The busy loop must be listed at ~1 core. Not necessarily FIRST: on a
    # dev machine a browser can genuinely use more during the test.
    row = [r for r in bg["top"] if r["pid"] == hog.pid]
    assert row and row[0]["cores"] > 0.6, bg["top"]
    row = ct.check_assumption({"telemetry": {"background": dict(bg, cpu_pct=99.0)}},
                              "max_background_cpu_pct", {"value": 10.0})
    assert row["status"] == ct.VIOLATED and f"pid {hog.pid}" in row["reason"]


# ---- Nano: 144 storage errors before a reboot, 0 after — same failing card

def test_storage_errors_are_remembered_per_card_across_a_reboot(monkeypatch):
    monkeypatch.setattr(storage, "_card_ids", lambda: {"mmcblk0": "cid:0353445344333247"})
    bad = ("[ 3205.9] blk_update_request: I/O error, dev mmcblk0, sector 52918591\n"
           "[ 3206.0] Buffer I/O error on dev mmcblk0p1, logical block 6614\n")
    monkeypatch.setattr(storage, "_kernel_log", lambda: (bad, "dmesg"))
    first = storage.kernel_storage_errors()
    assert first["count"] == 2
    # reboot: the kernel log is clean, the card is not
    monkeypatch.setattr(storage, "_kernel_log", lambda: ("[ 0.0] booted\n", "dmesg"))
    again = storage.kernel_storage_errors()
    assert again["count"] == 0 and again["remembered"][0]["count"] == 2
    from edgelens.hardware import readiness
    checks = readiness.assess(True, None, None, None, [], {}, storage=again)
    disk = next(c for c in checks if c["name"] == "Storage")
    assert disk["status"] == "warn" and "earlier boot" in disk["value"]
    # a NEW card has a new CID: clean
    monkeypatch.setattr(storage, "_card_ids", lambda: {"mmcblk0": "cid:99NEWCARD"})
    assert storage.kernel_storage_errors()["remembered"] == []
    # and the user can clear it after testing the card
    monkeypatch.setattr(storage, "_card_ids", lambda: {"mmcblk0": "cid:0353445344333247"})
    storage.forget()
    assert storage.kernel_storage_errors()["remembered"] == []


def test_doctor_forget_storage_errors(monkeypatch):
    from typer.testing import CliRunner
    from edgelens.cli import app
    monkeypatch.setattr(storage, "_card_ids", lambda: {"mmcblk0": "cid:1"})
    storage._save({"cid:1": {"device": "mmcblk0", "count": 3, "last_line": "x"}})
    res = CliRunner().invoke(app, ["doctor", "--forget-storage-errors"])
    assert res.exit_code == 0 and storage._load() == {}


# ---- Perfetto: two files of one boot, hours apart

def test_export_notes_files_recorded_far_apart():
    def res(origin):
        return {"clock": {"trace_origin_mono_ns": origin, "wall": {"wall_minus_mono_ns": 5}},
                "trace": {"events": [{"iter": 0, "stage": "s", "start_ns": 0, "dur_ns": 10**6,
                                      "thread": 1}]}}
    _, notes = tl.build([("a", res(10**9)), ("b", res(10**9 + 3600 * 10**9))])
    assert any("3600 s apart" in n and "--align start" in n for n in notes)
    _, notes = tl.build([("a", res(10**9)), ("b", res(2 * 10**9))])
    assert not notes


# ---- validation script: T4 is a tool check; the clock A/B is a measurement

def test_t4_gates_a_known_change_and_reports_clocks_as_info():
    text = (ROOT / "scripts" / "validate_on_jetson.sh").read_text()
    assert 'gate "T4 A/B known change"' in text and "EDGELENS_AB_EXTRA_MS" in text
    assert "INFO  T4 clocks" in text
    assert "expected IMPROVED" not in text          # a hardware effect is no longer a gate
    assert 'skip "E1 violated: $V" "the reference run did not run' in text

"""Fixes from the first Jetson Nano run of v0.2 (Layer 0/1)."""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from edgelens.benchmark import onnx_pipeline as op
from edgelens.markers import INCLUDE_DIR
from edgelens.markers.collector import MarkerCollector

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
CC = shutil.which("cc") or shutil.which("gcc")


class _FakeOrt:
    __file__ = "/nonexistent/onnxruntime/__init__.py"

    @staticmethod
    def get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def fake_gpu_ort(monkeypatch, tmp_path):
    monkeypatch.setattr(op, "ort", _FakeOrt)
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


def test_cuda_probe_crash_is_a_diagnosis_not_a_crash(fake_gpu_ort, monkeypatch):
    # what the Nano did: loading the CUDA provider killed Python with SIGBUS
    monkeypatch.setattr(op, "_PROBE_CODE",
                        "import os, signal\nprint('LISTED x', flush=True)\n"
                        "os.kill(os.getpid(), signal.SIGBUS)\n")
    usable, reason = op.cuda_probe()
    assert usable is False and "crashed Python" in reason and "SIGBUS" in reason
    assert op.cuda_probe_kind() == "crash"
    assert set(op.unloadable_providers()) == {"CUDAExecutionProvider"}
    assert op.usable_providers() == ["CPUExecutionProvider"]
    # a failure is NOT cached on disk (it must not outlive the problem): the
    # next process tests again — and sees the fix
    assert not (fake_gpu_ort / "edgelens" / "ort_cuda_probe.json").exists()
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    monkeypatch.setattr(op, "_PROBE_CODE", "print('LISTED CUDAExecutionProvider')\n"
                                           "print('CUDA_OK')\n")
    assert op.cuda_probe_kind() == "ok"


def test_cuda_probe_ok_and_error_paths(fake_gpu_ort, monkeypatch):
    monkeypatch.setattr(op, "_PROBE_CODE", "print('LISTED a'); print('CUDA_OK')")
    assert op.cuda_probe() == (True, None)
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_gpu_ort / "other"))
    monkeypatch.setattr(op, "_PROBE_CODE",
                        "import sys\nsys.stderr.write('libcublasLt.so.11: cannot open shared "
                        "object file\\n'); sys.exit(1)")
    usable, reason = op.cuda_probe()
    assert usable is False and "CUDA/cuDNN libraries" in reason


def test_two_onnxruntime_packages_are_reported_with_the_fix(monkeypatch):
    monkeypatch.setattr(op, "ort_distributions",
                        lambda: {"onnxruntime": "1.19.2", "onnxruntime-gpu": "1.16.0"})
    problems = op.ort_install_problems()
    assert len(problems) == 1
    assert "2 onnxruntime packages" in problems[0]["problem"]
    assert "pip uninstall -y onnxruntime onnxruntime-gpu" in problems[0]["fix"]
    assert "onnxruntime-gpu wheel" in problems[0]["fix"]


def test_doctor_survives_and_explains_a_broken_onnxruntime(fake_gpu_ort, monkeypatch):
    from typer.testing import CliRunner
    from edgelens.cli import app
    monkeypatch.setattr(op, "_PROBE_CODE", "import os, signal\nos.kill(os.getpid(), signal.SIGBUS)\n")
    monkeypatch.setattr(op, "ort_distributions",
                        lambda: {"onnxruntime": "1.19.2", "onnxruntime-gpu": "1.16.0"})
    res = CliRunner().invoke(app, ["doctor"])
    assert res.exit_code == 0, res.output
    assert "ONNX Runtime installation is broken" in res.output
    assert "pip uninstall" in res.output


# ---------------------------------------------------------------- markers under load

@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_span_stays_cheap_when_the_queue_is_full():
    """Nano: 234 us per span because every call re-sent a 32 KB batch.
    With chunked flushes and back-off, a full queue costs ~a memcpy."""
    d = tempfile.mkdtemp(prefix="eln")
    try:
        src = Path(d) / "b.c"
        exe = Path(d) / "b"
        src.write_text(r"""
#include "edgelens_mark.h"
#include <stdio.h>
int main(void) {
    uint64_t t0 = el_now_ns();
    for (int i = 0; i < 20000; i++) { el_span_t s = el_begin("x"); el_end(s); }
    uint64_t sent, dropped; el_stats(&sent, &dropped);
    printf("%.1f %llu %llu\n", (double)(el_now_ns() - t0) / 20000,
           (unsigned long long)sent, (unsigned long long)dropped);
    return 0;
}""")
        subprocess.run([CC, "-O2", "-I", INCLUDE_DIR, str(src), "-o", str(exe)], check=True)
        sock_path = os.path.join(d, "s.sock")
        blocked = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)   # never read: always full
        blocked.bind(sock_path)
        out = subprocess.run([str(exe)], env=dict(os.environ, EDGELENS_SOCKET=sock_path),
                             stdout=subprocess.PIPE, universal_newlines=True,
                             check=True).stdout.split()
        blocked.close()
        per_span_us = float(out[0]) / 1000.0
        assert per_span_us < 10.0, f"{per_span_us} us per span under backpressure"
        assert int(out[2]) > 0          # dropped (counted), since nobody ever reads
    finally:
        shutil.rmtree(d, ignore_errors=True)


@linux_only
def test_collector_listener_sees_each_new_pid_once_and_parsing_is_deferred(tmp_path):
    path = str(tmp_path / "s.sock")
    seen = []
    c = MarkerCollector(path)
    c.add_listener(lambda m: seen.append(m["pid"]))
    c.start()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    from edgelens.markers import protocol
    for i in range(50):
        s.sendto(protocol.format_span("a", i, 111, 1, i, i + 1, "-", "x"), path)
    s.sendto(protocol.format_span("b", 0, 222, 1, 5, 6, "-", "y") + b"\n"
             + protocol.format_span("b", 1, 222, 1, 6, 7, "-", "y"), path)
    time.sleep(0.3)
    first = c.drain()
    assert len(first) == 52
    assert c.drain() == []                       # nothing twice
    spans, st = c.stop()
    assert seen == [111, 222]
    assert st["received"] == 52 and st["lost"] == 0 and st["datagrams"] == 51
    assert any(x["backpressure"] for x in st["senders"])     # the batched sender


def test_sampler_measures_zone_cost_and_skips_slow_zones(monkeypatch):
    from edgelens.hardware import telemetry as tel
    reads = []
    monkeypatch.setattr(tel, "_thermal_zones", [("fast", "/f", 0.01), ("slow", "/s", 9.0)])
    monkeypatch.setattr(tel, "_read_zone", lambda p: reads.append(p) or 40.0)
    monkeypatch.setattr(tel, "_temp_calls", [0])
    for _ in range(40):
        temps = tel.read_temps()
    assert temps == {"fast": 40.0, "slow": 40.0}             # slow value reused
    assert reads.count("/f") == 40 and reads.count("/s") == 2
    assert tel.slow_zones() == [("slow", 9.0)]


def test_selftest_marks_large_jitter_with_the_desktop_hint(monkeypatch):
    from edgelens import selftest as st

    def fake_run(d, it, telemetry):
        return {"stage_stats_ms": {"reference": {"p50": d + 0.002, "p99": d + 2.0, "max": d + 3}},
                "latency": {"p50": d + 0.006}}
    monkeypatch.setattr(st, "_reference_run", fake_run)
    rows = st.check_harness([1.0], 100)
    jitter = [r for r in rows if r["check"] == "scheduling_jitter"][0]
    assert jitter["status"] == "WARN" and "multi-user.target" in jitter["fix"]
    assert [r for r in rows if r["check"] == "harness_accuracy"][0]["status"] == "PASS"


def test_validation_script_stops_on_a_crash_instead_of_continuing(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_on_jetson.sh"
    text = script.read_text()
    start = text.index("STEP_TIMEOUT=")
    end = text.index("gate() {")
    helpers = text[start:end]
    harness = tmp_path / "h.sh"
    harness.write_text("set -u\nOUT=%s\nmkdir -p $OUT\n: > $OUT/SUMMARY.txt\n%s\n"
                       "run python3 -c 'import os,signal; os.kill(os.getpid(), signal.SIGBUS)'\n"
                       "echo SHOULD-NOT-GET-HERE\n" % (tmp_path / "out", helpers))
    res = subprocess.run(["bash", str(harness)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True, cwd=str(tmp_path))
    assert res.returncode == 3
    assert "SHOULD-NOT-GET-HERE" not in res.stdout
    assert "STOP  crashed in" in (tmp_path / "out" / "SUMMARY.txt").read_text()

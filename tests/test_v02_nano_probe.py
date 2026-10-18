"""Second Jetson Nano run (onnxruntime-gpu 1.11, JetPack 4.6): the v0.2 CUDA
check allocated a bare CUDA tensor, which crashed on that build although real
sessions work (TensorRT ran there in v0.1). The check is now a real session."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edgelens.benchmark import onnx_pipeline as op
from edgelens.cli import app


class _FakeOrt:
    __file__ = "/nonexistent/onnxruntime/__init__.py"

    @staticmethod
    def get_available_providers():
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def gpu_ort(monkeypatch, tmp_path):
    monkeypatch.setattr(op, "ort", _FakeOrt)
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    monkeypatch.setattr(op, "ort_distributions", lambda: {"onnxruntime-gpu": "1.11.0"})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


def _crash_at(step):
    return ("import os, signal\nprint('STEP import', flush=True)\n"
            "print('LISTED CUDAExecutionProvider,CPUExecutionProvider', flush=True)\n"
            f"print('STEP {step}', flush=True)\nos.kill(os.getpid(), signal.SIGBUS)\n")


def test_real_probe_runs_a_session_on_the_embedded_model():
    pytest.importorskip("onnxruntime")
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", op._PROBE_CODE, op._PROBE_MODEL_B64],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("STEP import") and "LISTED " in out.stdout
    # the embedded model is a valid Relu model (checked on the CPU here)
    import base64
    import numpy as np
    import onnxruntime as ort
    s = ort.InferenceSession(base64.b64decode(op._PROBE_MODEL_B64),
                             providers=["CPUExecutionProvider"])
    y = s.run(None, {"x": np.array([[-1, 2, -3, 4]], np.float32)})[0]
    assert y.tolist() == [[0, 2, 0, 4]]


def test_crash_with_one_install_names_the_step_and_does_not_blame_two_packages(gpu_ort,
                                                                            monkeypatch):
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("session"))
    usable, reason = op.cuda_probe()
    assert usable is False and op.cuda_probe_kind() == "crash"
    assert "creating a CUDA session crashed Python (SIGBUS)" in reason
    assert "two onnxruntime packages" not in reason
    assert "--force-reinstall" in reason and "CPU provider" in reason


def test_crash_with_two_installs_says_so(gpu_ort, monkeypatch):
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("run"))
    monkeypatch.setattr(op, "ort_distributions",
                        lambda: {"onnxruntime": "1.19.2", "onnxruntime-gpu": "1.11.0"})
    reason = op.cuda_probe()[1]
    assert "running a CUDA session crashed" in reason and "two onnxruntime packages" in reason


def test_session_falling_back_to_cpu_means_missing_libraries(gpu_ort, monkeypatch):
    monkeypatch.setattr(op, "_PROBE_CODE",
                        "print('LISTED CUDAExecutionProvider,CPUExecutionProvider')\n"
                        "print('STEP session')\nprint('USED CPUExecutionProvider')\n")
    usable, reason = op.cuda_probe()
    assert usable is False and op.cuda_probe_kind() == "error"
    assert "fell back to the CPU" in reason


def test_a_verdict_cached_by_the_old_probe_is_ignored(gpu_ort, monkeypatch):  # noqa: E301
    # what the Nano has on disk: "crash", cached by probe v1 (no probe version in the key)
    key = op._install_key()
    key.pop("probe")
    path = gpu_ort / "edgelens" / "ort_cuda_probe.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": key, "result": [False, "old crash", "crash"]}))
    monkeypatch.setattr(op, "_PROBE_CODE",
                        "print('LISTED CUDAExecutionProvider,CPUExecutionProvider')\n"
                        "print('CUDA_OK')\n")
    assert op.cuda_probe() == (True, None)
    assert json.loads(path.read_text())["result"][2] == "ok"


def test_doctor_retests_a_failure_and_says_when_a_success_is_cached(gpu_ort, monkeypatch):
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("session"))
    res = CliRunner().invoke(app, ["doctor"])
    assert res.exit_code == 0, res.output
    out = " ".join(res.output.split())
    assert "GPU provider is unusable" in out and "installation is broken" not in out
    assert "edgelens doctor --no-cache" in out
    # fixed (e.g. wheel reinstalled, new card): a failure is never cached, so the
    # next doctor tests again by itself
    from edgelens.hardware import readiness

    def gpu_row():
        return next(c for c in readiness.gather(True, {}) if c["name"] == "GPU inference")
    monkeypatch.setattr(op, "_PROBE_CODE", _OK)
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    out = " ".join(CliRunner().invoke(app, ["doctor"]).output.split())
    assert "GPU provider is unusable" not in out, out
    assert "just now" in gpu_row()["advice"]
    # a success IS cached, and doctor says so
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    assert CliRunner().invoke(app, ["doctor"]).exit_code == 0
    assert "cached from" in gpu_row()["advice"] and "--no-cache" in gpu_row()["advice"]
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("run"))      # broke again
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    out = " ".join(CliRunner().invoke(app, ["doctor", "--no-cache"]).output.split())
    assert "running a CUDA session crashed" in out


_OK = "print('LISTED CUDAExecutionProvider,CPUExecutionProvider')\nprint('CUDA_OK')\n"


def test_only_success_is_cached(gpu_ort, monkeypatch):
    path = gpu_ort / "edgelens" / "ort_cuda_probe.json"
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("session"))
    op.cuda_probe()
    assert not path.exists()
    monkeypatch.setattr(op, "_PROBE_CODE", _OK)
    assert op.cuda_probe(force=True) == (True, None)
    saved = json.loads(path.read_text())
    assert saved["result"][2] == "ok" and saved["saved_at"]
    # a later failure removes the stale success
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("run"))
    assert op.cuda_probe(force=True)[0] is False and not path.exists()


def test_a_cached_failure_from_an_older_edgelens_is_never_trusted(gpu_ort, monkeypatch):
    path = gpu_ort / "edgelens" / "ort_cuda_probe.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": op._install_key(), "result": [False, "old", "crash"]}))
    monkeypatch.setattr(op, "_PROBE_CODE", _OK)
    assert op.cuda_probe() == (True, None)


@pytest.mark.parametrize("change", ["boot", "driver", "python"])
def test_reboot_driver_update_or_new_interpreter_tests_again(gpu_ort, monkeypatch, change):
    monkeypatch.setattr(op, "_PROBE_CODE", _OK)
    monkeypatch.setattr(op, "_boot_id", lambda: "boot-1")
    monkeypatch.setattr(op, "_cuda_driver_files", lambda: {"/usr/lib/libcuda.so.1.1": [1, 2]})
    assert op.cuda_probe() == (True, None)
    calls = []
    monkeypatch.setattr(op, "_run_isolated_probe", lambda: calls.append(1) or (True, None, "ok"))
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    op.cuda_probe()
    assert calls == [] and op.cuda_probe_cached_at()          # same everything: cache
    if change == "boot":
        monkeypatch.setattr(op, "_boot_id", lambda: "boot-2")
    elif change == "driver":
        monkeypatch.setattr(op, "_cuda_driver_files", lambda: {"/usr/lib/libcuda.so.1.1": [1, 3]})
    else:
        monkeypatch.setattr(op.sys if hasattr(op, "sys") else __import__("sys"),
                            "executable", "/other/venv/bin/python")
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    op.cuda_probe()
    assert calls == [1] and op.cuda_probe_cached_at() is None


def test_install_key_has_driver_and_boot(gpu_ort):
    key = op._install_key()
    for k in ("probe", "python", "ort_version", "cuda_driver", "boot_id"):
        assert k in key


# ---------------------------------------------------------------- the SD card

def test_sigbus_with_storage_errors_blames_the_card(gpu_ort, monkeypatch):
    from edgelens.hardware import storage
    log = ("[  12.1] mmc1: error -110 whilst initialising SD card\n"
           "[ 812.3] mmcblk0: error -84 transferring data, sector 2312, nr 8\n"
           "[ 812.3] blk_update_request: I/O error, dev mmcblk0, sector 2312\n"
           "[ 812.4] EXT4-fs error (device mmcblk0p1): ext4_find_entry:1455\n")
    monkeypatch.setattr(storage, "_kernel_log", lambda: (log, "dmesg"))
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("session"))
    reason = op.cuda_probe()[1]
    assert "3 storage error(s)" in reason and "SD card" in reason
    assert "--force-reinstall" not in reason


def test_storage_patterns_ignore_the_empty_slot_message():
    from edgelens.hardware import storage
    import re
    ok = "[   2.0] mmc1: error -110 whilst initialising SD card"
    assert not storage._PATTERNS.search(ok)
    for bad in ("Buffer I/O error on dev mmcblk0p1, logical block 3",
                "print_req_error: I/O error, dev mmcblk0, sector 7",
                "SQUASHFS error: zlib decompression failed"):
        assert storage._PATTERNS.search(bad), bad
    assert re  # noqa


def test_unreadable_kernel_log_still_mentions_the_card_for_sigbus(gpu_ort, monkeypatch):
    from edgelens.hardware import storage
    monkeypatch.setattr(storage, "_kernel_log", lambda: (None, "not readable"))
    monkeypatch.setattr(op, "_PROBE_CODE", _crash_at("session"))
    assert "sudo dmesg" in op.cuda_probe()[1]


def test_readiness_rows_use_the_real_findings():
    from edgelens.hardware import readiness
    checks = readiness.assess(True, {"name": "MAXN"}, True, True,
                              ["CUDAExecutionProvider", "CPUExecutionProvider"], {},
                              unloadable=["CUDAExecutionProvider"],
                              gpu_reason="creating a CUDA session crashed Python (SIGBUS) ...",
                              gpu_kind="crash",
                              storage={"readable": True, "count": 2, "lines": ["x I/O error"]})
    gpu = next(c for c in checks if c["name"] == "GPU inference")
    assert "crashed Python" in gpu["advice"] and "cuDNN" not in gpu["advice"]
    disk = next(c for c in checks if c["name"] == "Storage")
    assert disk["status"] == "warn" and "SD card" in disk["advice"]


def test_validation_preflight_prints_its_findings_once(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "scripts" / "validate_on_jetson.sh").read_text()
    block = text[text.index('log "preflight: is this'):]
    block = block[:block.index("# ----")]
    assert 'tail -6 "$OUT/steps.log"' not in block


@pytest.mark.parametrize("rc,stops,line", [(0, False, "PASS  preflight"),
                                           (3, False, "WARN  preflight"),
                                           (2, True, "Preflight failed")])
def test_validation_preflight_stops_only_for_a_broken_install(tmp_path, rc, stops, line):
    import os
    import subprocess
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "scripts" / "validate_on_jetson.sh").read_text()
    helpers = text[text.index("STEP_TIMEOUT="):text.index("gate() {")]
    block = text[text.index('log "preflight: is this'):text.index("# ---------------------"
                                                                    "--------------------------"
                                                                    "------------------- v0.1")]
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "python3").write_text(f"#!/bin/sh\ncat >/dev/null\necho 'onnxruntime CUDA probe: x'\n"
                                  f"exit {rc}\n")
    (shim / "python3").chmod(0o755)
    h = tmp_path / "h.sh"
    out = tmp_path / "o"
    h.write_text(f"set -u\nOUT={out}\nmkdir -p $OUT\n: > $OUT/SUMMARY.txt\n{helpers}\n{block}\n"
                 "echo CONTINUED\n")
    res = subprocess.run(["bash", str(h)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True,
                         env=dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}"))
    assert ("CONTINUED" not in res.stdout) == stops, res.stdout
    assert line in (out / "SUMMARY.txt").read_text() + res.stdout

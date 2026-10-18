"""Regression tests for the second review of v0.2 (Nano fixes + Layer 2)."""

import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from edgelens import periodic as per
from edgelens import sweep as sw
from edgelens.benchmark import onnx_pipeline as op
from edgelens.markers import collector as col
from edgelens.validate.engine import validate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_on_jetson.sh"
LSTM = Path(__file__).parent / "fixtures" / "lstm_anomaly.onnx"

try:
    import onnxruntime  # noqa: F401  (availability check)
    HAVE_ORT = True
except ImportError:            # pragma: no cover
    HAVE_ORT = False


# ---------------------------------------------------------------- sweep / equivalence

def test_integer_outputs_are_decisions_compared_exactly():
    ref = {"label": np.array([[1], [2], [3], [0]], np.int64)}
    same = sw.compare_outputs(ref, {"label": ref["label"].copy()})
    assert same["verdict"] == sw.EQUIVALENT and same["decision_agreement"] == 1.0
    flipped = sw.compare_outputs(ref, {"label": np.array([[1], [2], [0], [0]], np.int64)})
    assert flipped["verdict"] == sw.DECISIONS_DIFFER
    assert flipped["decision_agreement"] == pytest.approx(0.75)


def test_bool_outputs_are_decisions():
    ref = {"alarm": np.array([[True], [False]])}
    r = sw.compare_outputs(ref, {"alarm": np.array([[True], [True]])})
    assert r["verdict"] == sw.DECISIONS_DIFFER


def test_nan_in_a_backend_output_is_an_infinite_error_and_fails_validate():
    ref = {"score": np.array([[0.2], [0.7], [0.4]], np.float32)}
    bad = {"score": np.array([[0.2], [np.nan], [0.4]], np.float32)}
    r = sw.compare_outputs(ref, bad)
    assert r["max_abs_error"] == float("inf")
    assert r["non_finite_values"] == 1
    assert r["verdict"] != sw.EQUIVALENT
    res = {"equivalence": r, "schema_version": "0.2"}
    v = validate(res, {"max_output_error": 0.05})
    assert v["verdict"] == "FAIL", v["checks"]


def test_inf_where_reference_is_inf_is_not_an_error():
    ref = {"x": np.array([[np.inf, 1.0]], np.float32)}
    r = sw.compare_outputs(ref, {"x": np.array([[np.inf, 1.0]], np.float32)})
    assert r["max_abs_error"] == 0.0 and r["non_finite_values"] == 0


def test_explicit_threshold_wins_over_argmax_for_multi_value_outputs():
    ref = {"s": np.array([[0.1, 0.9], [0.6, 0.4]], np.float32)}
    cand = {"s": np.array([[0.1, 0.9], [0.6, 0.7]], np.float32)}      # 2nd value crosses 0.5
    auto = sw.compare_outputs(ref, cand)                                # argmax: 0 vs 1 flips
    thr = sw.compare_outputs(ref, cand, threshold=0.5)
    assert thr["decision_agreement"] == pytest.approx(0.5)
    assert auto["decision_agreement"] == pytest.approx(0.5)
    thr2 = sw.compare_outputs(ref, cand, threshold=0.8)                 # nothing crosses 0.8 differently
    assert thr2["decision_agreement"] == 1.0


def test_output_drift_finding_without_decisions_says_so():
    from edgelens.diagnose.insights import findings
    eq = {"verdict": sw.NUMERIC_DRIFT, "max_abs_error": 0.3, "decision_agreement": None,
          "reference_backend": "cpu"}
    res = {"equivalence": eq, "backend": {"name": "tensorrt-fp16"}}
    text = json.dumps(findings(res))
    assert "no decisions compared" in text and "None" not in text.replace("null", "")


@pytest.mark.skipif(not HAVE_ORT, reason="onnxruntime not installed")
def test_reference_backend_that_cannot_run_is_an_error(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad input")
    monkeypatch.setattr(sw, "run_outputs", boom)
    with pytest.raises(RuntimeError, match="reference backend 'cpu' could not run"):
        sw.run_sweep(str(LSTM), ["cpu"], iterations=5, eval_samples=4)


# ---------------------------------------------------------------- periodic

def test_energy_per_period_is_mean_power_times_period():
    res = {"energy": {"available": True, "power_w_mean": 2.0, "energy_per_iteration_j": 0.0019}}
    row = per._energy_row(res, 1000.0, idle_w=1.5, n=40)
    assert row["energy_per_period_mj"] == pytest.approx(2000.0)          # 2 W x 1 s
    assert row["dynamic_energy_per_inference_mj"] == pytest.approx(500.0)
    assert "energy_low_confidence" not in row


def test_primary_is_the_longest_period():
    from edgelens.core.pipeline import Pipeline
    p = Pipeline("quick")
    p.add_stage("m", lambda x=None: None)
    r = per.run_periodic(p, [5.0, 30.0, 15.0], iterations=4, warm_iterations=10,
                         idle_baseline_s=0)
    assert r["periodic"]["primary"]["period_ms"] == 30.0


# ---------------------------------------------------------------- collector / probe / marks

def test_collector_bounds_its_raw_backlog_without_drain(tmp_path, monkeypatch):
    monkeypatch.setattr(col, "RAW_PARSE_AT", 100)
    from edgelens.markers import protocol
    path = str(tmp_path / "s.sock")
    c = col.MarkerCollector(path)
    c.start()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    for i in range(1000):
        s.sendto(protocol.format_span("a", i, 1, 1, i, i + 1, "-", "x"), path)
        if i % 200 == 0:
            time.sleep(0.01)
    time.sleep(0.3)
    assert len(c._raw) <= 101
    spans, st = c.stop()
    assert st["received"] == 1000 and st["lost"] == 0


def test_probe_cache_keeps_only_definite_answers(monkeypatch, tmp_path):
    class FakeOrt:
        __file__ = "/nonexistent/onnxruntime/__init__.py"

        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(op, "ort", FakeOrt)
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(op, "_PROBE_CODE",
                        "import sys\nsys.stderr.write('libcudnn.so.8: cannot open shared "
                        "object file\\n'); sys.exit(1)")
    assert op.cuda_probe()[0] is False
    assert not (tmp_path / "edgelens" / "ort_cuda_probe.json").exists()   # an error: re-probe
    # fixing LD_LIBRARY_PATH must lead to a new probe, not a stale "ok"/"crash"
    monkeypatch.setattr(op, "_cuda_probe_result", None)
    monkeypatch.setattr(op, "_PROBE_CODE", "print('CUDA_OK')")
    assert op.cuda_probe() == (True, None)
    k1 = op._install_key()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    assert op._install_key() != k1


def test_python_span_with_empty_or_cr_name_still_parses(tmp_path, monkeypatch):
    import edgelens.mark as mark
    path = str(tmp_path / "m.sock")
    c = col.MarkerCollector(path)
    c.start()
    mark.configure(path)
    mark.record("", 1, 2)
    mark.record("a\rb", 3, 4)
    mark.flush()
    time.sleep(0.2)
    spans, st = c.stop()
    names = sorted(s["name"] for s in spans)
    assert names == ["?", "a b"] and st["lost"] == 0


# ---------------------------------------------------------------- validation script

def _helpers():
    text = SCRIPT.read_text()
    return text[text.index("STEP_TIMEOUT="):text.index("gate() {")]


def _run_harness(tmp_path, body):
    h = tmp_path / "h.sh"
    h.write_text("set -u\nOUT=%s\nmkdir -p $OUT\n: > $OUT/SUMMARY.txt\n%s\n%s\n"
                 % (tmp_path / "o", _helpers(), body))
    return subprocess.run(["bash", str(h)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)


def test_script_early_sigkill_is_a_crash_not_a_timeout(tmp_path):
    res = _run_harness(tmp_path, "run python3 -c 'import os,signal; os.kill(os.getpid(), "
                                 "signal.SIGKILL)'\necho SHOULD-NOT-GET-HERE")
    assert res.returncode == 3 and "SHOULD-NOT-GET-HERE" not in res.stdout
    assert "out of memory" in res.stdout


def test_script_real_timeout_continues(tmp_path):
    res = _run_harness(tmp_path, "STEP_TIMEOUT=1\nrun sleep 5\necho rc=$?")
    assert "TIMEOUT after 1s" in res.stdout and "rc=124" in res.stdout


def test_script_requirement_checks_are_gates():
    text = SCRIPT.read_text()
    assert 'gate "I3 FP16 requirements"' in text
    assert 'gate "I2 soak requirements"' in text

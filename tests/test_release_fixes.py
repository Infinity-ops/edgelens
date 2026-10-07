"""Regression tests for issues found in the pre-release (v0.1.0) review.

Each test names the real failure it guards against.
"""

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
from typer.testing import CliRunner

import edgelens
from edgelens.cli import app
from edgelens.diagnose.engine import diagnose
from edgelens.hardware import detector

runner = CliRunner()
REPO = pathlib.Path(__file__).resolve().parents[1]


# --- packaging ---------------------------------------------------------------

def test_version_is_identical_in_pyproject_and_package():
    # Two hand-edited copies of the version drift apart; the release
    # workflow checks this too, but catch it on every PR.
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "(.+?)"', text, re.M).group(1)
    assert edgelens.__version__ == declared


# --- detector ------------------------------------------------------------------

def test_device_tree_model_nul_byte_is_stripped(tmp_path):
    # Real Jetson Nano: /proc/device-tree/model ends in "\x00", which .strip()
    # keeps, so every result stored "...Developer Kit\u0000" as the board.
    f = tmp_path / "model"
    f.write_bytes(b"NVIDIA Jetson Nano Developer Kit\x00")
    assert detector._read_file(str(f)) == "NVIDIA Jetson Nano Developer Kit"


# --- diagnosis wording -------------------------------------------------------------

def _cpu_model_run(environment):
    return {
        "pipeline_source": "onnxruntime:CPUExecutionProvider (small_cnn.onnx)",
        "stage_avg_ms": {"capture": 0.01, "preprocess": 1.0, "h2d_copy": 0.01,
                         "inference": 20.0, "d2h_copy": 0.01, "postprocess": 0.05},
        "telemetry": {"cpu_percent_mean": 90, "gpu_percent_mean": None,
                      "sample_count": 30, "cpu_count": 8},
        "environment": environment,
    }


def test_inference_on_cpu_on_laptop_does_not_tell_user_to_install_jetson_wheel():
    # First PyPI users run --model on a laptop with no GPU provider at all;
    # "the GPU is not used" + "install the JetPack wheel" is wrong advice there.
    v = diagnose(_cpu_model_run({"is_jetson": False,
                                 "ort_providers": ["CPUExecutionProvider"]}))
    p = v["primary"]
    assert p["type"] == "INFERENCE_ON_CPU"
    assert "GPU is not used" not in p["detail"]
    assert "no GPU execution provider that can load" in p["detail"]
    assert p["evidence"]["gpu_execution_providers_available"] == []


def test_inference_on_cpu_on_jetson_points_at_the_gpu_wheel():
    v = diagnose(_cpu_model_run({"is_jetson": True,
                                 "ort_providers": ["CPUExecutionProvider"]}))
    p = v["primary"]
    assert "GPU is not used" in p["detail"]
    assert "onnxruntime-gpu" in p["recommendation"]


def test_inference_on_cpu_names_available_gpu_providers():
    v = diagnose(_cpu_model_run({"is_jetson": False, "ort_providers": [
        "CUDAExecutionProvider", "CPUExecutionProvider"]}))
    assert "CUDAExecutionProvider" in v["primary"]["recommendation"]


def test_single_core_preprocess_wording_reports_cores_busy_not_idle_claim():
    # Real Nano run: preprocess 33.5%, CPU 42.3% on 4 cores = ~1.7 cores busy.
    # The old text claimed "about one core's worth ... other cores sit idle".
    r = {"stage_avg_ms": {"capture": 0.01, "preprocess": 2.53, "h2d_copy": 0.65,
                          "inference": 4.10, "d2h_copy": 0.20, "postprocess": 0.07},
         "pipeline_source": "onnxruntime:CUDAExecutionProvider (small_cnn.onnx)",
         "telemetry": {"cpu_percent_mean": 42.26, "gpu_percent_mean": 49.09,
                       "sample_count": 22, "cpu_count": 4}}
    p = diagnose(r)["primary"]
    assert p["type"] == "CPU_BOUND_PREPROCESS"
    assert "~1.7 of 4 cores busy" in p["detail"]
    assert "idle" not in p["detail"]


# --- CLI errors ----------------------------------------------------------------------

def test_diagnose_malformed_json_is_one_line_error_not_traceback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    res = runner.invoke(app, ["diagnose", str(bad)])
    assert res.exit_code == 1
    assert "Cannot read" in res.output
    assert not isinstance(res.exception, json.JSONDecodeError)


def test_compare_rejects_json_that_is_not_a_result(tmp_path):
    other = tmp_path / "package.json"
    other.write_text(json.dumps({"name": "x"}))
    res = runner.invoke(app, ["compare", str(other), str(other)])
    assert res.exit_code == 1
    assert "not an EdgeLens" in res.output


def test_benchmark_rejects_zero_iterations(tmp_path):
    res = runner.invoke(app, ["benchmark", "--demo", "--iterations", "0",
                              "--save", str(tmp_path / "z.json")])
    assert res.exit_code != 0
    assert not (tmp_path / "z.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="locale behaviour checked on Linux")
def test_report_writes_utf8_under_non_utf8_locale(tmp_path):
    # LANG=C without UTF-8 mode: Path.write_text() defaulted to ASCII and the
    # report crashed with UnicodeEncodeError on the first em dash.
    res = runner.invoke(app, ["benchmark", "--demo", "--iterations", "20",
                              "--save", str(tmp_path / "d.json")])
    assert res.exit_code == 0, res.output
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    proc = subprocess.run(
        [sys.executable, "-m", "edgelens.cli", "report", str(tmp_path / "d.json"),
         "--output", str(tmp_path / "r.html")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, out
    assert "UnicodeEncodeError" not in out
    html = (tmp_path / "r.html").read_bytes().decode("utf-8")
    assert "—" in html


# --- harness -------------------------------------------------------------------------

def test_finite_source_running_out_gives_actionable_error():
    # A list or a camera generator that ends used to escape as a bare
    # StopIteration() with no message.
    p = edgelens.Pipeline("t")
    p.add_stage("a", lambda x: x)
    with pytest.raises(ValueError, match=r"ran out after 3 item\(s\).*needs 7"):
        p.run(iterations=5, warmup=2, source=[1, 2, 3], telemetry=False)


def test_finite_source_error_reaches_cli_users_as_runtime_error():
    from edgelens.benchmark.runner import run_benchmark
    p = edgelens.Pipeline("t")
    p.add_stage("a", lambda x: x)
    with pytest.raises(RuntimeError, match="ran out"):
        run_benchmark(pipeline=p, iterations=5, warmup=0, source=iter([1]))


# --- GPU providers that are listed but cannot load ------------------------------------
# Real laptop: onnxruntime-gpu listed Tensorrt+CUDA, libcublasLt was missing.
# get_available_providers() still listed them, sessions silently fell back to
# CPU, `benchmark --model` crashed in h2d_copy and doctor showed "GPU inference ok".

FIXTURE = str(REPO / "tests" / "fixtures" / "tiny_model.onnx")
LISTED = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def _needs_ort():
    return pytest.importorskip("onnxruntime")


@pytest.fixture
def cuda_listed_but_broken(monkeypatch):
    from edgelens.benchmark import onnx_pipeline as op
    monkeypatch.setattr(op, "available_providers", lambda: list(LISTED))
    monkeypatch.setattr(op, "cuda_probe", lambda: (False, "libcublasLt.so.12: not found"))
    return op


def test_unloadable_gpu_providers_are_not_picked(cuda_listed_but_broken):
    op = cuda_listed_but_broken
    assert op.unloadable_providers() == ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    assert op.usable_providers() == ["CPUExecutionProvider"]
    assert op.pick_provider() == "CPUExecutionProvider"


def test_explicit_unloadable_provider_fails_with_actionable_message(cuda_listed_but_broken):
    _needs_ort()
    with pytest.raises(RuntimeError, match=r"(?s)cannot load here.*--provider CPUExecutionProvider"):
        cuda_listed_but_broken.OnnxStagePipeline(FIXTURE, provider="CUDAExecutionProvider")


def test_auto_provider_falls_back_to_cpu_and_records_why(cuda_listed_but_broken):
    _needs_ort()
    from edgelens.benchmark.runner import run_benchmark
    r = run_benchmark(iterations=5, warmup=1, model_path=FIXTURE)
    assert "CPUExecutionProvider" in r["pipeline_source"]
    skipped = r["pipeline"]["config"]["provider_fallbacks"]
    assert [s["provider"] for s in skipped] == ["TensorrtExecutionProvider",
                                                "CUDAExecutionProvider"]
    assert "cannot load" in skipped[0]["reason"]


def _cpu_session_instead(monkeypatch, op):
    """Simulate onnxruntime's silent fallback: ask for X, get a CPU session."""
    ort = _needs_ort()
    monkeypatch.setattr(op, "available_providers", lambda: list(LISTED))
    monkeypatch.setattr(op, "cuda_probe", lambda: (True, None))
    monkeypatch.setattr(op, "_session_with", lambda path, prov: (
        ort.InferenceSession(path, providers=["CPUExecutionProvider"]),
        "*** EP Error\nEP Error libnvinfer.so.10: cannot open shared object file\n"))


def test_explicit_provider_that_silently_falls_back_is_an_error(monkeypatch):
    from edgelens.benchmark import onnx_pipeline as op
    _cpu_session_instead(monkeypatch, op)
    with pytest.raises(RuntimeError, match="could not enable 'TensorrtExecutionProvider'"):
        op.OnnxStagePipeline(FIXTURE, provider="TensorrtExecutionProvider")


def test_auto_provider_skips_silent_fallbacks_until_one_really_loads(monkeypatch):
    from edgelens.benchmark import onnx_pipeline as op
    _cpu_session_instead(monkeypatch, op)
    p = op.OnnxStagePipeline(FIXTURE)
    assert p.provider == "CPUExecutionProvider"
    assert [f["provider"] for f in p.provider_fallbacks] == ["TensorrtExecutionProvider",
                                                             "CUDAExecutionProvider"]
    assert "libnvinfer" in p.provider_fallbacks[0]["reason"]


def test_readiness_warns_about_unloadable_gpu_providers_on_any_host():
    from edgelens.hardware import readiness
    checks = readiness.assess(is_jetson=False, power_mode=None, cpu_pinned=None,
                              gpu_pinned=None, providers=LISTED, power_probe=None,
                              unloadable=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
    gpu = [c for c in checks if c["name"] == "GPU inference"][0]
    assert gpu["status"] == "warn" and "cannot load" in gpu["value"]
    assert readiness.summary(checks, is_jetson=False)[0] == "warn"


def test_inference_on_cpu_advice_uses_loadable_providers_only():
    env = {"is_jetson": False, "ort_providers": LISTED,
           "ort_providers_usable": ["CPUExecutionProvider"]}
    p = diagnose(_cpu_model_run(env))["primary"]
    assert "no GPU execution provider that can load" in p["detail"]
    assert "available here" not in p["recommendation"]

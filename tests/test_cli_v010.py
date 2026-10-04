"""End-to-end CLI tests for the v0.1.0 features."""
import json
import os

import pytest
from typer.testing import CliRunner

from edgelens.cli import app
from edgelens.core.schema import SCHEMA_VERSION

HERE = os.path.dirname(__file__)
SMALL_CNN = os.path.join(HERE, "fixtures", "small_cnn.onnx")
TS_EXAMPLE = os.path.join(HERE, "..", "examples", "timeseries_vibration.py")
runner = CliRunner()

try:
    import onnxruntime  # noqa: F401
    HAS_ORT = True
except Exception:
    HAS_ORT = False


def _bench(tmp_path, *args, name="run.json"):
    out = tmp_path / name
    res = runner.invoke(app, ["benchmark", *args, "--save", str(out)])
    assert res.exit_code == 0, res.output
    return json.loads(out.read_text()), out


@pytest.mark.skipif(not HAS_ORT, reason="needs onnxruntime")
def test_model_benchmark_writes_v1_document(tmp_path):
    r, _ = _bench(tmp_path, "--model", SMALL_CNN, "--provider", "CPUExecutionProvider",
                  "--iterations", "30", "--deadline-ms", "1000")
    assert r["schema_version"] == SCHEMA_VERSION
    assert r["deadline"]["met"] is True
    assert r["pipeline"]["pack"] == "vision"
    assert r["identity"]["model_sha256"]
    assert r["trace"]["event_count"] == 30 * 6
    assert {"min", "p50", "max", "jitter", "p99_9"} <= set(r["latency"])
    assert r["pipeline"]["config"]["model_inputs"][0]["dtype"] == "float32"


def test_timeseries_pipeline_script_from_cli(tmp_path):
    r, path = _bench(tmp_path, "--pipeline", TS_EXAMPLE, "--iterations", "50")
    assert r["pipeline"]["pack"] == "timeseries"
    assert r["requirements"]["deadline_ms"] == 51.2
    assert r["pack_metrics"]["real_time_factor"]["mean"] < 1
    res = runner.invoke(app, ["diagnose", str(path)])
    assert res.exit_code == 0, res.output
    assert "DEADLINE_MET" in res.output


def test_paced_cli_run_has_no_simulated_backlog(tmp_path):
    r, _ = _bench(tmp_path, "--pipeline", TS_EXAMPLE, "--iterations", "5",
                  "--period-ms", "2", "--pace")
    assert r["requirements"]["paced"] is True
    assert r["backlog"] is None


def test_compare_strict_env_exit_codes(tmp_path):
    a, pa = _bench(tmp_path, "--demo", "--iterations", "20", name="a.json")
    b = dict(a)
    b["environment"] = dict(a["environment"], power_mode={"id": 1, "name": "5W"})
    pb = tmp_path / "b.json"
    pb.write_text(json.dumps(b))
    assert runner.invoke(app, ["compare", str(pa), str(pb)]).exit_code == 0
    res = runner.invoke(app, ["compare", str(pa), str(pb), "--strict-env"])
    assert res.exit_code == 2
    assert "Environment mismatch" in res.output


def test_report_escapes_untrusted_names(tmp_path):
    from edgelens.benchmark.runner import run_benchmark
    evil = "<script>alert(1)</script>"
    result = run_benchmark(iterations=3, warmup=0, stage_fns={evil: lambda: None})
    path = tmp_path / "evil.json"
    path.write_text(json.dumps(result))
    out = tmp_path / "evil.html"
    res = runner.invoke(app, ["report", str(path), "--output", str(out)])
    assert res.exit_code == 0, res.output
    html = out.read_text()
    assert evil not in html
    assert "&lt;script&gt;" in html


def test_pre_v010_json_still_diagnoses_reports_and_compares(tmp_path):
    old = {"mode": "hardware", "pipeline_source": "old", "iterations": 10,
           "stage_avg_ms": {"capture": 1, "preprocess": 9, "h2d_copy": 1, "inference": 5,
                            "d2h_copy": 1, "postprocess": 1},
           "total_latency_ms": 18, "fps": 55.5, "latency_p50_ms": 18,
           "latency_p95_ms": 19, "latency_p99_ms": 20,
           "telemetry": {"cpu_percent_mean": 95, "gpu_percent_mean": 20,
                         "max_temp_c": 50, "sample_count": 10}}
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old))
    res = runner.invoke(app, ["diagnose", str(p)])
    assert res.exit_code == 0 and "CPU_BOUND_PREPROCESS" in res.output
    assert runner.invoke(app, ["report", str(p), "--output", str(tmp_path / "o.html")]).exit_code == 0
    assert runner.invoke(app, ["compare", str(p), str(p)]).exit_code == 0


def test_monitor_runs_briefly():
    res = runner.invoke(app, ["monitor", "--duration", "1", "--interval", "0.2"])
    assert res.exit_code == 0, res.output


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_save_path_gives_one_line_error_not_traceback(tmp_path):
    # Real Nano case: power.json created by an earlier `sudo edgelens` run.
    target = tmp_path / "power.json"
    target.write_text("{}")
    target.chmod(0o400)
    res = runner.invoke(app, ["benchmark", "--demo", "--iterations", "5", "--save", str(target)])
    assert res.exit_code == 1
    assert "Could not write" in res.output and "chown" in res.output
    assert "Traceback" not in res.output

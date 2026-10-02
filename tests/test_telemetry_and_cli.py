import json

from typer.testing import CliRunner

from edgelens.cli import app
from edgelens.hardware.telemetry import TelemetryRecorder


def _sample(ts):
    return {"cpu_percent": 10.0, "gpu_percent": 40.0, "mem_percent": 50.0,
            "temps_c": {"GPU-therm": 45.0}, "timestamp": ts}


def test_duration_uses_wall_clock_not_requested_interval():
    # Real Nano behaviour: 22 samples, ~1s apart, with interval_s=0.2.
    rec = TelemetryRecorder(interval_s=0.2)
    rec._samples = [_sample(1000.0 + i * 1.0) for i in range(22)]
    summary = rec.summarize()
    assert summary["sample_count"] == 22
    assert summary["duration_s"] == 21.0          # not 22 * 0.2 = 4.4
    assert summary["effective_interval_s"] == 1.0


def test_benchmark_creates_missing_output_folder(tmp_path):
    out = tmp_path / "runs" / "nested" / "demo.json"
    result = CliRunner().invoke(app, ["benchmark", "--demo", "--iterations", "5",
                                      "--save", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["mode"] == "demo"

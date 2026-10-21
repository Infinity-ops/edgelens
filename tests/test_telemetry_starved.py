"""Telemetry starved by a pure-Python pipeline, and energy from too few samples.

AGX Orin, a ~1 s pure-Python benchmark: 2 telemetry samples, the first
finishing at the very end, and the output said "0.00 mJ/iteration ·
1000000.0 iterations/J". Both are now reported as what they are.
"""
from typer.testing import CliRunner

from edgelens.cli import app
from edgelens.hardware.telemetry import TelemetryRecorder


def _rec(n_samples, recording_s, interval_s=0.1):
    r = TelemetryRecorder(interval_s=interval_s, use_tegrastats=False, use_power=False)
    r.start_mono_ns = 0
    r._stop_mono_ns = int(recording_s * 1e9)
    return r._starvation(n_samples, [35.0] * n_samples, [1.0] * n_samples)


def test_starved_recording_is_flagged():
    st = _rec(2, 1.0)
    assert st and st["samples"] == 2 and st["expected"] == 10
    assert "GIL" in st["likely_cause"]


def test_healthy_or_short_recording_is_not_flagged():
    assert _rec(52, 5.5) is None          # the Orin model run: fine
    assert _rec(2, 0.2) is None           # too short to expect more


def test_energy_from_two_samples_prints_no_number(monkeypatch):
    import edgelens.cli as cli
    fake = {"mode": "hardware", "pipeline": {"name": "x", "pack": "custom", "stages": []},
            "pipeline_source": "x", "stage_stats_ms": {}, "stage_avg_ms": {}, "latency": {},
            "fps": 1.0, "iterations": 10,
            "energy": {"available": True, "power_w_mean": 6.4, "energy_per_iteration_j": 1e-06,
                       "iterations_per_joule": 1000000.0, "method": "sum_of_rails",
                       "low_confidence": "fewer than 3 power samples in the timed window"},
            "telemetry": {"sample_count": 2, "duration_s": 0.01,
                          "starved": {"samples": 2, "expected": 9, "recording_s": 0.96,
                                      "sampler_wall_ms_mean": 480.0, "sampler_cpu_ms_mean": 9.4,
                                      "likely_cause": "the pipeline holds Python's GIL"}}}
    from io import StringIO
    from rich.console import Console
    buf = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=300))
    cli._print_benchmark(fake)
    out = buf.getvalue()
    assert "1000000" not in out and "Energy: not measured" in out
    assert "Telemetry starved: 2 sample(s)" in out

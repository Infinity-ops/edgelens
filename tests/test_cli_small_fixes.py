"""Display fixes from the AGX Orin run."""
from io import StringIO

from rich.console import Console

import edgelens.cli as cli


def _capture(monkeypatch):
    buf = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=300))
    return buf


def test_model_benchmark_title_says_template_not_vision(monkeypatch):
    buf = _capture(monkeypatch)
    cli._print_benchmark({"mode": "hardware", "pipeline_source": "onnxruntime:CPUExecutionProvider (m.onnx)",
                          "pipeline": {"pack": "vision", "stages": []}, "stage_stats_ms": {}, "stage_avg_ms": {},
                          "latency": {}, "fps": 1.0, "iterations": 1})
    out = buf.getvalue()
    assert "vision-template pack" in out and "vision pack" not in out

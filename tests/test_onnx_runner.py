import os
from unittest.mock import patch

import pytest

from edgelens.benchmark.runner import run_benchmark

FIXTURE_MODEL = os.path.join(os.path.dirname(__file__), "fixtures", "tiny_model.onnx")


def test_model_path_runs_real_inference_not_placeholder():
    result = run_benchmark(iterations=10, warmup=2, model_path=FIXTURE_MODEL)
    assert result["mode"] == "hardware"
    assert "onnxruntime" in result["pipeline_source"]
    assert FIXTURE_MODEL in result["pipeline_source"]
    # Real timings should be present for every stage, and total latency
    # should be a small positive number (not a suspicious round sleep value).
    assert result["total_latency_ms"] > 0
    assert set(result["stage_avg_ms"].keys()) == {
        "capture", "preprocess", "h2d_copy", "inference", "d2h_copy", "postprocess",
    }


def test_cpu_provider_reports_near_zero_device_copies():
    result = run_benchmark(iterations=10, warmup=2, model_path=FIXTURE_MODEL,
                            provider="CPUExecutionProvider")
    # On CPU there is no device to copy to/from — copies must be ~0, not
    # a fabricated non-zero number.
    assert result["stage_avg_ms"]["h2d_copy"] < 0.5
    assert result["stage_avg_ms"]["d2h_copy"] < 0.5


def test_demo_flag_forces_simulated_mode_even_with_model_path():
    result = run_benchmark(iterations=5, demo=True, model_path=FIXTURE_MODEL)
    assert result["mode"] == "demo"
    assert result["pipeline_source"] == "simulated"


def test_no_pipeline_and_no_demo_on_jetson_raises_loudly():
    with patch("edgelens.hardware.detector.is_jetson", return_value=True):
        with pytest.raises(RuntimeError, match="needs something real to measure"):
            run_benchmark(iterations=5)


def test_no_pipeline_off_jetson_auto_falls_back_to_demo():
    with patch("edgelens.hardware.detector.is_jetson", return_value=False):
        result = run_benchmark(iterations=5)
        assert result["mode"] == "demo"


def test_real_model_off_jetson_still_uses_hardware_mode():
    # The key fix: a real .onnx model should NOT be forced into demo mode
    # just because the host isn't a Jetson — CPUExecutionProvider is a
    # legitimate real measurement on a laptop.
    with patch("edgelens.hardware.detector.is_jetson", return_value=False):
        result = run_benchmark(iterations=5, model_path=FIXTURE_MODEL)
        assert result["mode"] == "hardware"


def test_capture_stage_is_cheap_not_dominated_by_rng():
    # Regression test for a real bug found via hardware testing: capture()
    # used to call np.random.rand() fresh every iteration, which costs
    # ~1ms for a realistic CNN input size — enough to look like a genuine
    # bottleneck when it was actually just measuring NumPy's RNG cost.
    # capture() must now be cheap relative to inference, not comparable
    # to or larger than it.
    result = run_benchmark(iterations=30, warmup=5, model_path=FIXTURE_MODEL)
    capture_ms = result["stage_avg_ms"]["capture"]
    inference_ms = result["stage_avg_ms"]["inference"]
    assert capture_ms < inference_ms, (
        f"capture ({capture_ms}ms) should be cheap relative to inference "
        f"({inference_ms}ms) — if this fails, the RNG-per-iteration bug "
        f"may have regressed"
    )

from edgelens.compare.engine import compare


def _result(fps, p95, stage_ms, cpu=50, temp=55):
    return {
        "fps": fps,
        "total_latency_ms": 1000.0 / fps if fps else 0,
        "latency_p50_ms": p95 * 0.7,
        "latency_p95_ms": p95,
        "latency_p99_ms": p95 * 1.1,
        "stage_avg_ms": stage_ms,
        "telemetry": {"cpu_percent_mean": cpu, "gpu_percent_mean": 50, "max_temp_c": temp},
    }


def test_pass_when_performance_unchanged():
    before = _result(60, 16.0, {"preprocess": 5, "inference": 10})
    after = _result(60, 16.0, {"preprocess": 5, "inference": 10})
    result = compare(before, after)
    assert result["verdict"] == "PASS"


def test_regression_detected_on_fps_drop():
    before = _result(60, 16.0, {"preprocess": 5, "inference": 10})
    after = _result(45, 22.0, {"preprocess": 12, "inference": 10})
    result = compare(before, after)
    assert result["verdict"] == "REGRESSION"
    assert any("FPS dropped" in r for r in result["reasons"])
    assert any("preprocess" in r for r in result["reasons"])


def test_improvement_is_not_flagged_as_regression():
    before = _result(40, 25.0, {"preprocess": 12, "inference": 10})
    after = _result(60, 16.0, {"preprocess": 5, "inference": 10})
    result = compare(before, after)
    assert result["verdict"] == "PASS"
    assert result["metrics"]["fps"]["pct_change"] > 0

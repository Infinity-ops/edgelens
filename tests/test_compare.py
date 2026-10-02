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


# --- v0.1.0: generic stages, deadlines, environment awareness ---

def _env(power="MAXN", locked=True):
    return {"board": "NVIDIA Jetson Nano Developer Kit", "l4t": "R32.7.6", "cuda": "10.2",
            "tensorrt": "8.2.1", "python": "3.8.10", "packages": {"onnxruntime": "1.11.0"},
            "power_mode": {"id": 0, "name": power}, "cpu_clocks_locked": locked,
            "gpu_clocks_locked": locked}


def test_any_stage_names_are_compared_including_added_and_removed():
    before = _result(60, 16.0, {"acquire": 1, "fft": 5, "inference": 10})
    after = _result(60, 16.0, {"acquire": 1, "stft": 4, "inference": 10})
    stages = compare(before, after)["stages"]
    assert stages["fft"]["status"] == "removed"
    assert stages["stft"]["status"] == "added"
    assert stages["inference"]["status"] == "both"


def test_power_mode_change_is_reported_as_environment_difference():
    before = _result(60, 16.0, {"inference": 10})
    after = _result(40, 25.0, {"inference": 15})
    before["environment"] = _env("MAXN")
    after["environment"] = _env("5W")
    result = compare(before, after)
    assert result["verdict"] == "REGRESSION"
    assert result["environment"]["comparable"] is False
    fields = [d["field"] for d in result["environment"]["differences"]]
    assert fields == ["power_mode"]


def test_same_environment_is_comparable_and_old_files_are_unknown():
    before = _result(60, 16.0, {"inference": 10})
    after = _result(60, 16.0, {"inference": 10})
    assert compare(before, after)["environment"]["comparable"] is None   # pre-v0.1 files
    before["environment"] = after["environment"] = _env()
    assert compare(before, after)["environment"]["comparable"] is True


def test_deadline_miss_ratio_rise_is_a_regression():
    before = _result(60, 16.0, {"inference": 10})
    after = _result(60, 16.0, {"inference": 10})
    before["deadline"] = {"miss_ratio": 0.0}
    after["deadline"] = {"miss_ratio": 0.02}
    result = compare(before, after)
    assert result["verdict"] == "REGRESSION"
    assert any("Deadline miss ratio" in r for r in result["reasons"])

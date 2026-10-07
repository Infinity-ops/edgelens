"""`edgelens validate`: requirement -> PASS / FAIL / INCONCLUSIVE with evidence.
Fixture numbers are from real Jetson Nano runs (TensorRT, small_cnn.onnx)."""
import json

import pytest
from typer.testing import CliRunner

from edgelens.cli import app
from edgelens.validate import exit_code, validate


def _nano_run(clocks_pinned=False, power=True, iterations=300, deadline_ms=None, misses=0):
    """Real numbers: unpinned run 2026-10-04 (p50 11.364, p99 82.901, max 247.4,
    4.2 W, 51.3 mJ/iteration); pinned run (p50 6.544, p99 9.404)."""
    lat = ({"n": iterations, "mean": 6.758, "p50": 6.544, "p95": 8.09, "p99": 9.404,
            "p99_9": None, "max": 18.196} if clocks_pinned else
           {"n": iterations, "mean": 15.028, "p50": 11.364, "p95": 27.868, "p99": 82.901,
            "p99_9": None, "max": 247.424})
    r = {
        "schema_version": 1, "mode": "hardware", "iterations": iterations,
        "pipeline_source": "onnxruntime:TensorrtExecutionProvider (small_cnn.onnx)",
        "latency": lat, "fps": round(1000 / lat["mean"], 2),
        "stage_avg_ms": {"preprocess": 2.547, "h2d_copy": 1.49, "inference": 8.097,
                         "d2h_copy": 2.072},
        "stage_stats_ms": {"preprocess": {"mean": 2.547, "p99": 5.052},
                           "h2d_copy": {"mean": 1.49, "p99": 18.13},
                           "inference": {"mean": 8.097, "p99": 16.457},
                           "d2h_copy": {"mean": 2.072, "p99": 8.995}},
        "telemetry": {"cpu_percent_mean": 43.2, "gpu_percent_mean": 52.2, "sample_count": 34,
                      "max_temp_c": 52.0},
        "environment": {"board": "NVIDIA Jetson Nano Developer Kit", "is_jetson": True,
                        "power_mode": {"id": 0, "name": "MAXN"},
                        "cpu_clocks_locked": clocks_pinned, "gpu_clocks_locked": clocks_pinned},
        "identity": {"environment_id": "08f555cbd3d8", "run_id": "20261004T200436-5abd3e"},
        "energy": ({"available": True, "power_w_mean": 4.2, "energy_per_iteration_j": 0.0513,
                    "method": "input_rail"} if power else {"available": False}),
        "deadline": None,
    }
    if deadline_ms is not None:
        r["deadline"] = {"deadline_ms": deadline_ms, "iterations": iterations, "misses": misses,
                         "miss_ratio": misses / iterations, "met": misses == 0,
                         "worst_ms": lat["max"], "worst_overrun_ms": max(0, lat["max"] - deadline_ms),
                         "first_miss_iteration": 0 if misses else None,
                         "max_consecutive_misses": 1 if misses else 0, "slack_p50_ms": 1.0}
    return r


def test_unpinned_nano_fails_p99_and_points_at_the_tail():
    v = validate(_nano_run(), {"p99_ms": 20, "max_power_w": 5, "max_energy_mj": 60})
    assert v["verdict"] == "FAIL" and exit_code(v["verdict"]) == 1
    by = {c["requirement"]: c for c in v["checks"]}
    assert by["p99_ms"]["status"] == "FAIL" and by["p99_ms"]["measured"] == 82.901
    assert by["max_power_w"]["status"] == "PASS"
    assert by["max_energy_mj"]["measured"] == 51.3
    assert v["limiting_stage"]["stage"] == "h2d_copy"          # largest stage p99 (18.13 ms)
    assert v["diagnosis"]["primary"]["type"] == "CLOCKS_NOT_PINNED"


def test_pinned_nano_passes_the_same_requirements():
    v = validate(_nano_run(clocks_pinned=True), {"p99_ms": 20, "max_power_w": 5})
    assert v["verdict"] == "PASS" and exit_code("PASS") == 0
    assert "diagnosis" not in v


def test_unmeasurable_requirement_is_never_a_pass():
    r = _nano_run(clocks_pinned=True, power=False, iterations=50)
    r["latency"]["p99"] = None                      # 50 samples cannot support a p99
    v = validate(r,
                 {"p99_ms": 20, "max_energy_mj": 60, "max_miss_ratio": 0.001})
    assert v["verdict"] == "INCONCLUSIVE" and exit_code("INCONCLUSIVE") == 2
    reasons = {c["requirement"]: c["reason"] for c in v["checks"]}
    assert "at least 100 iterations" in reasons["p99_ms"]
    assert "power" in reasons["max_energy_mj"]
    assert "--deadline-ms" in reasons["max_miss_ratio"]


def test_one_fail_beats_not_measured():
    v = validate(_nano_run(power=False), {"p99_ms": 20, "max_energy_mj": 60})
    assert v["verdict"] == "FAIL"


def test_root_only_power_sensor_reason_carries_the_fix():
    r = _nano_run(power=False)
    r["telemetry"]["power"] = {"available": False, "reason": "permission_denied",
                               "fix": "sudo chmod o+r /sys/bus/i2c/drivers/ina3221x/x"}
    row = validate(r, {"max_power_w": 5})["checks"][0]
    assert row["status"] == "NOT_MEASURED" and "sudo chmod o+r" in row["reason"]


def test_miss_ratio_and_min_fps():
    r = _nano_run(clocks_pinned=True, deadline_ms=10, misses=3)       # 1.0 % missed
    v = validate(r, {"max_miss_ratio": 0.001, "min_fps": 100})
    by = {c["requirement"]: c for c in v["checks"]}
    assert by["max_miss_ratio"]["status"] == "FAIL"
    assert by["min_fps"]["status"] == "PASS" and by["min_fps"]["direction"] == "min"


def test_demo_result_can_never_validate():
    r = _nano_run(clocks_pinned=True)
    r["mode"] = "demo"
    assert validate(r, {"p99_ms": 1000})["verdict"] == "INCONCLUSIVE"


def test_bad_input_is_rejected():
    with pytest.raises(ValueError, match="No requirements"):
        validate(_nano_run(), {})
    with pytest.raises(ValueError, match="Unknown requirement"):
        validate(_nano_run(), {"p42_ms": 1})


@pytest.mark.parametrize("args,code", [
    (["--p99-ms", "20"], 1), (["--p95-ms", "30"], 0), (["--p99.9-ms", "20"], 2), ([], 1),
])
def test_cli_exit_codes_and_saved_verdict(tmp_path, args, code):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_nano_run()))
    res = CliRunner().invoke(app, ["validate", str(path), *args])
    assert res.exit_code == code, res.output
    if args:
        saved = json.loads((tmp_path / "run.validation.json").read_text())
        assert saved["schema_version"] == 1
        assert saved["verdict"] == {0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE"}[code]

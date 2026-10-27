"""AGX Orin, 0.2.0rc1 gate run: `periodic` printed "extra energy per
inference" of 57, 114 and 376 mJ for the same 0.3 ms LSTM at 1 Hz on a quiet
board. 376 mJ in a ~4 ms inference is ~100 W on a board drawing 4-9 W: the
idle baseline had moved, not the model."""
import pytest

from edgelens import periodic as per


def _res(pw, low=False):
    en = {"available": True, "power_w_mean": pw}
    if low:
        en["low_confidence"] = "few samples"
    return {"energy": en}


def test_row_above_the_back_to_back_power_is_not_attributable():
    # the Orin 1 Hz row: 4.281 W mean, 3.905 W idle; back to back drew ~4.1 W
    row = per._energy_row(_res(4.281), 1000.0, idle_w=3.905, n=30, idle_drift_w=0.01,
                          ceiling_w=4.10)
    assert row["dynamic_energy_per_inference_mj"] is None
    assert "back to back" in row["dynamic_energy_not_attributable"]
    assert row["energy_per_period_mj"] == pytest.approx(4281.0)    # still measured


def test_excess_smaller_than_the_idle_drift_is_a_bound():
    row = per._energy_row(_res(3.95), 1000.0, idle_w=3.95, n=30, idle_drift_w=0.10,
                          ceiling_w=5.0)
    assert row["dynamic_energy_per_inference_mj"] is None
    assert row["dynamic_energy_below_mj"] == pytest.approx(100.0)
    assert "drift" in row["dynamic_energy_below_reason"]


def test_a_real_excess_is_still_reported():
    row = per._energy_row(_res(2.95), 1000.0, idle_w=2.80, n=30, idle_drift_w=0.005,
                          ceiling_w=3.5)
    assert row["dynamic_energy_per_inference_mj"] == pytest.approx(150.0)
    assert "dynamic_energy_not_attributable" not in row


def _fake_run(calls, b2b_power, row_power):
    def run_harness(pipe, iterations=100, warmup=10, period_ms=None, pace=False, **kw):
        calls.append({"iterations": iterations, "period_ms": period_ms,
                      "max_seconds": kw.get("max_seconds")})
        lat = [0.3] * iterations
        pw = row_power if pace else b2b_power
        return {"iterations": iterations,
                "latency": {"p50": 0.3, "p99": 0.4, "mean": 0.3},
                "energy": {"available": True, "power_w_mean": pw, "power_sample_count": 20,
                           "energy_per_iteration_j": 0.001},
                "latency_series_ms": lat, "service_series_ms": lat}
    return run_harness


def test_periodic_measures_idle_twice_and_a_long_enough_back_to_back(monkeypatch):
    from edgelens.core import harness
    from edgelens.hardware import power as power_mod

    class Reader:
        def available(self):
            return True

        def read(self):
            return {"x": 1}
    idles = iter([3.90, 3.92])
    calls = []
    monkeypatch.setattr(power_mod, "PowerReader", Reader)
    monkeypatch.setattr(harness, "measure_idle_power", lambda s, r=None: next(idles))
    monkeypatch.setattr(harness, "run_harness", _fake_run(calls, 4.10, 4.281))
    from edgelens.core import identity
    monkeypatch.setattr(identity, "attach_identity", lambda *a, **k: None)

    r = per.run_periodic(object(), [1000.0], iterations=30, warm_iterations=200,
                         idle_baseline_s=1)
    p = r["periodic"]
    # 200 x 0.3 ms = 60 ms is less than one power sample: re-run for >= 2 s
    assert calls[1]["iterations"] >= 2000 / 0.3 and calls[1]["max_seconds"] == 6.0
    assert p["idle_power_before_w"] == 3.90 and p["idle_power_after_w"] == 3.92
    assert p["idle_power_w"] == pytest.approx(3.91) and p["idle_drift_w"] == pytest.approx(0.02)
    row = p["primary"]
    assert row["dynamic_energy_per_inference_mj"] is None
    assert row["dynamic_energy_not_attributable"]
    assert any("not reported" in n for n in p["notes"])


def test_the_table_says_not_attributable_instead_of_a_number(capsys):
    from edgelens import cli_insights
    row = {"period_ms": 1000.0, "p50_ms": 3.8, "p99_ms": None, "penalty_ratio_p50": 14.0,
           "power_w_mean": 4.281, "energy_per_period_mj": 4281.0,
           "dynamic_energy_per_inference_mj": None,
           "dynamic_energy_not_attributable": "mean power above back to back"}
    res = {"periodic": {"back_to_back": {"p50_ms": 0.27, "p99_ms": 0.58, "power_w_mean": 4.1,
                                         "energy_per_inference_mj": 0.9},
                        "idle_power_w": 3.91, "idle_power_before_w": 3.90,
                        "idle_power_after_w": 3.92, "periods": [row], "primary": row,
                        "duty_cycle_penalty": False, "gpu_railgate": None, "notes": []}}
    cli_insights.print_periodic(res)
    out = " ".join(capsys.readouterr().out.split())
    assert "(see" in out and "note)" in out and "375" not in out
    assert "before 3.9 W, after 3.92 W" in out


def test_a_ratio_without_enough_milliseconds_says_why_it_is_no_penalty(capsys):
    from edgelens import cli_insights
    row = {"period_ms": 100.0, "p50_ms": 0.30, "p99_ms": None, "penalty_ratio_p50": 3.97,
           "penalty_ms_p50": 0.224, "power_w_mean": None, "energy_per_period_mj": None,
           "dynamic_energy_per_inference_mj": None}
    res = {"periodic": {"back_to_back": {"p50_ms": 0.076, "p99_ms": 0.1}, "idle_power_w": None,
                        "periods": [row], "primary": row, "duty_cycle_penalty": False,
                        "gpu_railgate": None, "notes": []}}
    cli_insights.print_periodic(res)
    out = " ".join(capsys.readouterr().out.split())
    assert "3.97x" in out and "+0.224 ms, under the 0.5 ms that counts" in out

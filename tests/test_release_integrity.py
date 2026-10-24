"""v0.2 release review: measurement-integrity defects, each with a hand-worked
expected value."""
import numpy as np

from edgelens.core.result import build_result
from edgelens.core.stats import deadline_stats, latency_stats
from edgelens.validate.engine import validate

DESC = {"name": "t", "stages": [{"name": "a", "role": "inference"}]}


def _result(e2e, deadline_ms=None, telemetry=None, paced=False, period_ms=None, idle=None):
    return build_result(pipeline_desc=DESC, stage_samples_ms={"a": e2e}, e2e_ms=e2e,
                        mode="hardware", pipeline_source="t", deadline_ms=deadline_ms,
                        telemetry_summary=telemetry, paced=paced, period_ms=period_ms,
                        idle_power_w=idle)


def test_one_miss_in_millions_is_not_rounded_away():
    """miss_ratio is stored rounded to 6 decimals: 1 / 3,000,000 reads 0.0.
    validate must compare the exact ratio, or `--max-miss-ratio 0` PASSes a
    run that missed."""
    n = 3_000_000
    dl = deadline_stats([1.0] * (n - 1) + [20.0], 10)
    assert dl["miss_ratio"] == 0.0
    v = validate({"deadline": dl, "latency": {}}, {"max_miss_ratio": 0})
    assert v["verdict"] == "FAIL" and v["checks"][0]["status"] == "FAIL"


def test_paced_dynamic_energy_uses_the_window_not_the_latency():
    """300 iterations, one per second, 10 ms of work each; the board draws
    5 W on average over the 300 s window, 4 W idle -> 1 W dynamic x 1 s =
    1 J per iteration (not 1 W x 10 ms = 10 mJ)."""
    tel = {"power": {"available": True, "method": "input_rail", "power_w_mean": 5.0,
                     "energy_j": 1500.0, "duration_s": 300.0, "sample_count": 3000}}
    en = _result([10.0] * 300, telemetry=tel, paced=True, period_ms=1000, idle=4.0)["energy"]
    assert en["energy_per_iteration_j"] == 5.0
    assert en["dynamic_power_w"] == 1.0
    assert en["dynamic_energy_per_iteration_j"] == 1.0


def test_back_to_back_dynamic_energy_is_unchanged():
    """Back to back the window is ~ iterations x latency, so both formulas
    agree: 200 x 5 ms = 1 s at 9 W, 6 W idle -> 3 W x 5 ms = 15 mJ."""
    tel = {"power": {"available": True, "method": "input_rail", "power_w_mean": 9.0,
                     "energy_j": 9.0, "duration_s": 1.0, "sample_count": 10}}
    en = _result([5.0] * 200, telemetry=tel, idle=6.0)["energy"]
    assert en["dynamic_energy_per_iteration_j"] == 0.015


def test_no_thermal_zone_is_null_not_zero_degrees():
    """A laptop or container has no readable thermal zone: the result must
    say "unknown", never 0 C (which validate / contracts would also have to
    special-case)."""
    from edgelens.hardware import telemetry as tm
    rec = tm.TelemetryRecorder(interval_s=0.01)
    rec._samples = [{"timestamp": 1.0, "mono_ns": 1_000_000_000, "cpu_percent": 10.0,
                     "temps_c": {}},
                    {"timestamp": 1.1, "mono_ns": 1_100_000_000, "cpu_percent": 20.0,
                     "temps_c": {}}]
    summary = rec.summarize()
    assert summary["max_temp_c"] is None
    with_zone = tm.TelemetryRecorder(interval_s=0.01)
    with_zone._samples = [dict(rec._samples[0], temps_c={"cpu": 41.5}),
                          dict(rec._samples[1], temps_c={"cpu": 43.0})]
    assert with_zone.summarize()["max_temp_c"] == 43.0
    v = validate({"telemetry": summary, "latency": {}}, {"max_temp_c": 85})
    assert v["checks"][0]["status"] == "NOT_MEASURED"


def test_percentiles_match_numpy_and_the_sample_count_rule():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 2.0, 1000)
    st = latency_stats(list(x))
    assert st["p50"] == round(float(np.percentile(x, 50)), 3)
    assert st["p99"] == round(float(np.percentile(x, 99)), 3)
    assert st["p99_9"] == round(float(np.percentile(x, 99.9)), 3)
    assert st["std"] == round(float(np.std(x, ddof=1)), 3)
    few = latency_stats(list(x[:99]))
    assert few["p99"] is None and few["p95"] is not None
    assert any("p99 (needs >= 100" in m for m in few["insufficient_samples"])


def test_deadline_boundary_and_bursts():
    d = deadline_stats([5, 10, 10.0001, 11, 3, 12], 10)
    assert d["misses"] == 3                    # equal to the deadline is met
    assert d["max_consecutive_misses"] == 2 and d["first_miss_iteration"] == 2
    assert d["worst_overrun_ms"] == 2.0


def test_record_of_a_command_that_exits_at_once_keeps_its_result(monkeypatch, tmp_path):
    """`edgelens record -- ./fast_app`: the command can finish (a zombie)
    before process sampling starts; psutil then raises ZombieProcess, which
    used to abort the run with "No process with pid N"."""
    import sys
    from edgelens.attach import runner
    from edgelens.attach.procsampler import TargetError

    def gone(pid, **kw):
        import time
        import psutil
        end = time.monotonic() + 10
        while time.monotonic() < end:          # the real race: the child is a zombie
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        raise TargetError(f"No process with pid {pid}.")
    monkeypatch.setattr(runner, "ProcessSampler", gone)
    run = runner.observe(cmd=[sys.executable, "-c", "pass"], markers=False,
                         socket_path=str(tmp_path / "s.sock"), telemetry=False)
    assert run.ended_by == "command exited" and run.process is None
    assert "exited before process sampling" in run.process_note
    # a --pid that does not exist is still an error
    import pytest
    with pytest.raises(TargetError):
        runner.observe(pid=2 ** 22 + 12345, markers=False, telemetry=False, duration_s=0.1)

"""Tests for the generic engine: Pipeline, harness, observer, stats, packs."""
import json
import time

import numpy as np
import pytest

import edgelens as el
from edgelens.core.pipeline import infer_role
from edgelens.core.stats import deadline_stats, latency_stats, simulate_backlog

FAST = dict(telemetry=False)   # keep unit tests independent of host telemetry


# ---------------- stats ----------------

def test_percentiles_are_withheld_without_enough_samples():
    s = latency_stats(list(range(1, 151)))
    assert s["p99"] is not None            # 150 >= 100
    assert s["p99_9"] is None              # needs 1000
    assert any("p99_9" in x for x in s["insufficient_samples"])
    assert s["max"] == 150 and s["min"] == 1


def test_jitter_is_std_and_tail_spread_uses_max_when_p99_missing():
    s = latency_stats([10.0] * 9 + [20.0])
    assert s["p99"] is None
    assert s["tail_spread"] == pytest.approx(20.0 - s["p50"])
    assert s["jitter"] == s["std"] > 0


def test_deadline_stats_counts_bursts():
    d = deadline_stats([1, 1, 5, 6, 7, 1, 5], deadline_ms=4)
    assert d["misses"] == 4
    assert d["max_consecutive_misses"] == 3
    assert d["first_miss_iteration"] == 2
    assert d["worst_overrun_ms"] == 3
    assert d["met"] is False
    assert deadline_stats([1, 2], None) is None


def test_backlog_simulation_detects_unstable_queue():
    assert simulate_backlog([5.0] * 20, period_ms=10)["stable"] is True
    unstable = simulate_backlog([12.0] * 20, period_ms=10)
    assert unstable["stable"] is False
    assert unstable["max_backlog_items"] >= 3
    assert unstable["method"].startswith("simulated")


# ---------------- pipeline model ----------------

@pytest.mark.parametrize("name,role", [
    ("capture", "input"), ("h2d_copy", "transfer"), ("feature_extraction", "preprocess"),
    ("denoise_model", "inference"), ("health_estimation", "decision"), ("nms", "postprocess"),
    ("whatever", "other"),
])
def test_role_inference(name, role):
    assert infer_role(name) == role


def test_pipeline_rejects_bad_stages():
    p = el.Pipeline("x")
    p.add_stage("a", lambda: None)
    with pytest.raises(ValueError, match="Duplicate"):
        p.add_stage("a", lambda: None)
    with pytest.raises(TypeError, match="callable"):
        p.add_stage("b", 42)
    with pytest.raises(TypeError, match="required arguments"):
        p.add_stage("c", lambda a, b: None)
    with pytest.raises(ValueError, match="unknown role"):
        p.add_stage("d", lambda: None, role="magic")
    with pytest.raises(ValueError, match="Unknown pack"):
        el.Pipeline("y", pack="nope")


def test_data_flows_between_one_arg_stages_and_source():
    seen = []
    p = el.Pipeline("flow")
    p.add_stage("double", lambda x: x * 2)
    p.add_stage("legacy_zero_arg", lambda: None)       # gets nothing, returns None
    p.add_stage("sink", lambda x: seen.append(x))
    p.run(iterations=3, warmup=0, source=iter([1, 2, 3]), **FAST)
    assert seen == [None, None, None]                  # zero-arg stage breaks the chain
    seen.clear()
    q = el.Pipeline("flow2")
    q.add_stage("double", lambda x: x * 2)
    q.add_stage("sink", lambda x: seen.append(x))
    q.run(iterations=3, warmup=0, source=iter([1, 2, 3]), **FAST)
    assert seen == [2, 4, 6]


def test_decorator_registration_and_explicit_role():
    p = el.Pipeline("deco")

    @p.stage
    def grab():
        return 1

    @p.stage(role="inference", device="dla0")
    def head(x):
        return x

    assert p.stage_names == ["grab", "head"]
    assert p.describe()["stages"][1] == {"name": "head", "role": "inference", "device": "dla0"}


def test_stage_error_names_the_stage():
    p = el.Pipeline("boom")
    p.add_stage("ok", lambda: None)
    p.add_stage("bad", lambda: 1 / 0)
    with pytest.raises(RuntimeError, match="Stage 'bad' raised ZeroDivisionError"):
        p.run(iterations=2, warmup=0, **FAST)


# ---------------- harness result ----------------

def test_harness_result_shape_and_trace():
    p = el.Pipeline("shape")
    p.add_stage("a", lambda: time.sleep(0.001))
    p.add_stage("b", lambda: None)
    r = p.run(iterations=20, warmup=1, deadline_ms=0.5, **FAST)
    assert r["schema_version"] == el.SCHEMA_VERSION == 1
    assert r["iterations"] == 20
    assert r["trace"]["event_count"] == 40
    ev = r["trace"]["events"][0]
    assert set(ev) == {"iter", "stage", "start_ns", "dur_ns", "thread"}
    # end-to-end is measured per iteration, never less than the stage sum
    assert r["latency"]["mean"] >= r["stage_avg_ms"]["a"]
    assert r["deadline"]["misses"] == 20            # 1 ms sleep vs 0.5 ms deadline
    for key in ("environment_id", "experiment_id", "run_id"):
        assert r["identity"][key]
    json.dumps(r)                                    # fully serialisable


def test_paced_mode_measures_real_queueing():
    # 8 ms of work released every 5 ms: response time must GROW, which
    # back-to-back timing would never show.
    p = el.Pipeline("paced")
    p.add_stage("work", lambda: time.sleep(0.008))
    r = p.run(iterations=8, warmup=0, period_ms=5, pace=True, **FAST)
    assert r["requirements"]["paced"] is True
    assert r["backlog"] is None                      # measured, not simulated
    events = [e for e in r["trace"]["events"]]
    assert len(events) == 8
    assert r["latency"]["max"] > r["latency"]["min"] + 10


def test_pace_without_period_is_an_error():
    p = el.Pipeline("x")
    p.add_stage("a", lambda: None)
    with pytest.raises(ValueError, match="needs a period"):
        p.run(iterations=2, pace=True, **FAST)


# ---------------- packs ----------------

def test_timeseries_pack_derives_period_deadline_and_rtf():
    sig = el.WindowSource.synthetic(1000, seconds=2, channels=2)
    src = el.WindowSource(sig, window=100, hop=50)
    p = el.Pipeline("ts", pack="timeseries", sample_rate_hz=1000, window=100, hop=50)
    p.add_stage("filter", lambda w: w - w.mean(axis=-1, keepdims=True))
    p.add_stage("fft", lambda w: np.abs(np.fft.rfft(w)))
    p.add_stage("inference", lambda f: float(f.sum()))
    r = p.run(iterations=50, warmup=2, source=src, **FAST)
    cfg = r["pipeline"]["config"]
    assert cfg["hop_period_ms"] == 50.0 and cfg["overlap_pct"] == 50.0
    assert r["requirements"]["deadline_ms"] == 50.0
    m = r["pack_metrics"]
    assert m["keeps_up_on_average"] is True
    assert 0 < m["real_time_factor"]["mean"] < 1
    assert r["backlog"]["stable"] is True


def test_timeseries_pack_flags_cannot_keep_up():
    from edgelens.diagnose.engine import diagnose
    p = el.Pipeline("slow", pack="timeseries", sample_rate_hz=10_000, window=20, hop=20)  # 2 ms hop
    p.add_stage("inference", lambda: time.sleep(0.004))
    r = p.run(iterations=10, warmup=0, **FAST)
    assert r["pack_metrics"]["real_time_factor"]["mean"] > 1
    types = [f["type"] for f in diagnose(r)["all_findings"]]
    assert "CANNOT_KEEP_UP" in types


def test_timeseries_config_validation():
    with pytest.raises(ValueError, match="hop"):
        el.Pipeline("bad", pack="timeseries", sample_rate_hz=1000, window=10, hop=20)


def test_window_source_wraps_and_uses_last_axis():
    src = el.WindowSource(np.arange(10), window=4, hop=3)
    assert src.n_windows == 3
    got = [next(src).tolist() for _ in range(4)]
    assert got == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [0, 1, 2, 3]]


def test_vision_pack_target_fps_sets_frame_budget():
    p = el.Pipeline("cam", pack="vision", target_fps=50)
    p.add_stage("capture", lambda: None)
    r = p.run(iterations=5, warmup=0, **FAST)
    assert r["requirements"]["deadline_ms"] == 20.0


# ---------------- observer mode ----------------

def test_observer_auto_iterations_and_save(tmp_path):
    out = tmp_path / "obs.json"
    with el.trace("obs", deadline_ms=100, telemetry=False, save=str(out)) as t:
        for _ in range(5):
            with t.stage("capture"):
                time.sleep(0.001)
            with t.stage("inference"):
                pass
    r = t.result
    assert r["iterations"] == 5
    assert [s["name"] for s in r["pipeline"]["stages"]] == ["capture", "inference"]
    assert r["pipeline_source"] == "observer:obs"
    assert r["deadline"]["met"] is True
    assert json.loads(out.read_text())["iterations"] == 5


def test_observer_explicit_iteration_includes_gaps():
    with el.trace("gaps", telemetry=False) as t:
        for _ in range(3):
            with t.iteration():
                with t.stage("a"):
                    pass
                time.sleep(0.003)                    # untraced gap between stages
                with t.stage("b"):
                    pass
    r = t.result
    assert r["iterations"] == 3
    assert r["latency"]["min"] >= 3.0                # gap counted in end-to-end


def test_observer_threads_are_recorded():
    import threading
    with el.trace("mt", telemetry=False) as t:
        def worker():
            with t.stage("post"):
                pass
        th = threading.Thread(target=worker)
        with t.stage("infer"):
            pass
        th.start(); th.join()
    threads = {e["thread"] for e in t.result["trace"]["events"]}
    assert len(threads) == 2


def test_paced_rtf_uses_service_time_not_response_time():
    # Reproduces the real Nano run: ~5 ms of work released every 2 ms. The
    # response time grows with the queue, but RTF must stay ~work/period
    # (it was reported as 10.4 against the wrong 51.2 ms period).
    p = el.Pipeline("ts", pack="timeseries", sample_rate_hz=10_000, window=1024, hop=512)
    p.add_stage("work", lambda: time.sleep(0.004))
    r = p.run(iterations=15, warmup=0, period_ms=2, pace=True, **FAST)
    assert r["requirements"]["deadline_ms"] == 2            # explicit period wins over hop
    assert r["latency"]["max"] > 20                         # queueing is visible in latency
    rtf = r["pack_metrics"]["real_time_factor"]["mean"]
    assert 1.8 < rtf < 4.0                                  # ~4 ms / 2 ms
    assert r["pack_metrics"]["period_ms"] == 2


def test_explicit_deadline_beats_period():
    p = el.Pipeline("x")
    p.add_stage("a", lambda: None)
    r = p.run(iterations=3, warmup=0, period_ms=10, deadline_ms=25, **FAST)
    assert r["requirements"]["deadline_ms"] == 25

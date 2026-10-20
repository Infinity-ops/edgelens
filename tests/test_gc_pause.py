"""Python GC during the timed loop — found on an AGX Orin 32GB.

One generation-2 collection (27.4 ms) caused the only deadline miss in
30 000 iterations of a 0.18 ms pipeline; with the collector off the worst
iteration was 1.3 ms. Part of the cause was EdgeLens itself: one dict per
stage per iteration in the trace. These tests pin down:
  * the timed loop adds no GC-tracked objects per iteration,
  * GC pauses are recorded with the iteration they hit,
  * --gc normal | freeze | off do what they say and restore the collector,
  * diagnose names the collector (GC_PAUSE) instead of blaming a stage,
  * compare flags runs made with different --gc modes.
"""
import gc
import json
from pathlib import Path

import pytest

from edgelens.compare.engine import requirements_diff
from edgelens.core.eventbuf import EventBuffer
from edgelens.core.gcwatch import GcWatch
from edgelens.core.harness import run_harness
from edgelens.core.pipeline import Pipeline
from edgelens.diagnose.engine import diagnose

FIXTURES = Path(__file__).parent / "fixtures"


def _pipe(fn=None):
    p = Pipeline("gc-test")
    p.add_stage("a", fn or (lambda: None))
    p.add_stage("b", lambda: None)
    return p


# ---- trace storage ----

def test_event_buffer_round_trip_and_limit():
    buf = EventBuffer(limit=3)
    buf.add(0, "pre", 10, 5, 7)
    buf.add(0, "infer", 15, 20, 7)
    buf.add(1, "pre", 40, 6, 7)
    buf.add(1, "infer", 46, 21, 7)          # over the limit
    assert len(buf) == 3 and buf.truncated
    assert buf.to_dicts() == [
        {"iter": 0, "stage": "pre", "start_ns": 10, "dur_ns": 5, "thread": 7},
        {"iter": 0, "stage": "infer", "start_ns": 15, "dur_ns": 20, "thread": 7},
        {"iter": 1, "stage": "pre", "start_ns": 40, "dur_ns": 6, "thread": 7},
    ]


def test_timed_loop_adds_no_tracked_objects_per_iteration():
    seen = {}

    def probe():
        n = probe.calls = getattr(probe, "calls", 0) + 1
        if n == 11:                           # first timed iteration (after 10 warmup)
            seen["first"] = len(gc.get_objects())
        if n == 10 + 3000:                    # last timed iteration
            seen["last"] = len(gc.get_objects())

    run_harness(_pipe(probe), iterations=3000, warmup=10, telemetry=False,
                attach_ids=False)
    # The old trace added 2 dicts per iteration here (6000); now ~nothing.
    assert seen["last"] - seen["first"] < 300, seen


# ---- recording pauses ----

def test_pause_is_recorded_with_its_iteration_and_deadline():
    w = GcWatch("normal", min_pause_ms=0.0)
    w.start(origin_ns=0)
    try:
        w.current_iter = 2
        gc.collect()                          # a full (gen-2) collection
    finally:
        w.stop()
    s = w.summary(e2e_ms=[0.1, 0.1, 12.0], deadline_ms=10.0)
    assert s["collections"]["gen2"] >= 1
    p = [p for p in s["pauses"] if p["generation"] == 2][0]
    assert p["iter"] == 2 and p["iter_latency_ms"] == 12.0 and p["deadline_missed"] is True
    assert s["deadline_misses"] == 1 and s["deadline_misses_during_gc"] == 1
    assert w._callback not in gc.callbacks


def test_harness_result_has_gc_block():
    r = run_harness(_pipe(), iterations=50, warmup=2, telemetry=False, attach_ids=False)
    g = r["gc"]
    assert g["mode"] == "normal" and g["collector_enabled_during_run"] is True
    assert set(g["collections"]) == {"gen0", "gen1", "gen2"}
    json.dumps(r)                             # stays serialisable


def test_gc_off_disables_only_during_the_loop():
    during = {}
    r = run_harness(_pipe(lambda: during.setdefault("on", gc.isenabled())),
                    iterations=5, warmup=0, telemetry=False, attach_ids=False, gc_mode="off")
    assert during["on"] is False
    assert gc.isenabled()
    assert r["gc"]["mode"] == "off" and r["gc"]["collector_enabled_during_run"] is False


@pytest.mark.skipif(not hasattr(gc, "get_freeze_count"), reason="gc.freeze needs 3.7+")
def test_gc_freeze_is_undone_after_the_loop():
    during = {}
    run_harness(_pipe(lambda: during.setdefault("frozen", gc.get_freeze_count())),
                iterations=5, warmup=0, telemetry=False, attach_ids=False, gc_mode="freeze")
    assert during["frozen"] > 0
    assert gc.get_freeze_count() == 0


def test_gc_off_reenables_after_a_failing_stage():
    def boom():
        raise RuntimeError("x")
    with pytest.raises(Exception):
        run_harness(_pipe(boom), iterations=3, warmup=0, telemetry=False,
                    attach_ids=False, gc_mode="off")
    assert gc.isenabled()


def test_unknown_gc_mode_is_rejected():
    with pytest.raises(ValueError):
        run_harness(_pipe(), iterations=1, warmup=0, telemetry=False, gc_mode="sometimes")


# ---- diagnose / compare ----

def _orin_like_result():
    """The shape of the real Orin run: one 27.4 ms gen-2 pause, one miss."""
    return {
        "schema_version": 1, "stage_avg_ms": {"preprocess": 0.037, "inference": 0.122},
        "latency": {"n": 30000, "p50": 0.177, "p99": 0.237, "max": 27.405},
        "deadline": {"deadline_ms": 10.0, "iterations": 30000, "misses": 1,
                     "miss_ratio": 1 / 30000, "worst_ms": 27.405, "worst_overrun_ms": 17.405,
                     "max_consecutive_misses": 1, "met": False},
        "telemetry": {"sample_count": 52, "cpu_percent_mean": 25.0, "gpu_percent_mean": 0.0},
        "gc": {"mode": "normal", "collections": {"gen0": 300, "gen1": 27, "gen2": 1},
               "pauses": [{"start_ns": 1, "dur_ms": 27.4, "generation": 2, "iter": 3634,
                           "iter_latency_ms": 27.405, "deadline_missed": True}],
               "deadline_misses": 1, "deadline_misses_during_gc": 1},
    }


def test_diagnose_names_the_collector():
    out = diagnose(_orin_like_result())
    f = [f for f in out["all_findings"] if f["type"] == "GC_PAUSE"]
    assert f, [x["type"] for x in out["all_findings"]]
    assert "1 of 1 deadline miss" in f[0]["detail"] and "3634" in f[0]["detail"]
    assert f[0]["evidence_strength"] == "strong"


def test_no_gc_finding_without_a_gc_block():
    r = _orin_like_result()
    del r["gc"]
    assert not [f for f in diagnose(r)["all_findings"] if f["type"] == "GC_PAUSE"]


def test_v0_1_0_result_still_diagnoses():
    doc = json.loads((FIXTURES / "result_v0_1_0.json").read_text())
    assert not [f for f in diagnose(doc)["all_findings"] if f["type"] == "GC_PAUSE"]


def test_compare_flags_different_gc_modes():
    a, b = {"gc": {"mode": "normal"}}, {"gc": {"mode": "freeze"}}
    assert [d["field"] for d in requirements_diff(a, b)] == ["gc_mode"]
    assert requirements_diff({}, {"gc": {"mode": "normal"}}) == []   # old file = normal

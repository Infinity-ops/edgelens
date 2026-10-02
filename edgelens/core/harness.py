"""
edgelens.core.harness
-----------------------
Harness mode: EdgeLens drives the loop around a Pipeline.

Unpaced (default): iterations run back to back — measures pure service
time. If a period is known (from requirements or the pack), the backlog
that WOULD form at that arrival rate is estimated and labelled as such.

Paced (pace=True with a period): iteration k is released at t0 + k*period.
Response time is measured from the scheduled release to the end of the last
stage, so an overrun that delays the next release shows up as real queueing
in the next iteration's latency — measured, not simulated.
"""

import threading
import time

from ..hardware import power as power_mod
from ..hardware import telemetry as telemetry_mod
from .result import MAX_TRACE_EVENTS, build_result


class StageError(RuntimeError):
    pass


def _call(stage, arg, iteration):
    try:
        return stage.fn(arg) if stage.arity == 1 else stage.fn()
    except Exception as e:
        raise StageError(
            f"Stage '{stage.name}' raised {type(e).__name__} at iteration {iteration}: {e}"
        ) from e


def measure_idle_power(seconds, reader=None):
    reader = reader or power_mod.PowerReader()
    if not seconds or not reader.available():
        return None
    vals, end = [], time.time() + seconds
    while time.time() < end:
        w = power_mod.total_power_w(reader.read())
        if w is not None:
            vals.append(w)
        time.sleep(0.05)
    return sum(vals) / len(vals) if vals else None


def run_harness(pipeline, iterations=100, warmup=10, source=None, deadline_ms=None,
                period_ms=None, pace=False, telemetry=True, telemetry_interval_s=0.1,
                idle_baseline_s=None, pipeline_source=None, attach_ids=True,
                model_path=None):
    if len(pipeline) == 0:
        raise ValueError("Pipeline has no stages.")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    pack, cfg = pipeline.pack, pipeline.config
    if deadline_ms is None:
        deadline_ms = pack.default_deadline_ms(cfg)
    if period_ms is None:
        period_ms = pack.default_period_ms(cfg)
    if pace and not period_ms:
        raise ValueError("pace=True needs a period (period_ms=..., or a pack config that "
                         "defines one, e.g. timeseries sample_rate_hz/window/hop).")

    stages = pipeline.stages
    src = iter(source) if source is not None else None

    def one_pass(it):
        value = next(src) if src is not None else None
        for st in stages:
            value = _call(st, value, it)

    for w in range(warmup):
        one_pass(-1 - w)

    idle_w = measure_idle_power(idle_baseline_s) if idle_baseline_s else None

    samples = {s.name: [] for s in stages}
    e2e, events = [], []
    truncated = False
    tid = threading.get_ident()
    recorder = None
    if telemetry:
        recorder = telemetry_mod.TelemetryRecorder(interval_s=telemetry_interval_s)
        recorder.start()
    t_origin = time.perf_counter_ns()
    try:
        period_ns = int(period_ms * 1e6) if (pace and period_ms) else None
        for it in range(iterations):
            value = next(src) if src is not None else None
            if period_ns is not None:
                release = t_origin + it * period_ns
                now = time.perf_counter_ns()
                if now < release:
                    time.sleep((release - now) / 1e9)
                it_start = release           # response time counts from release
            else:
                it_start = time.perf_counter_ns()
            for st in stages:
                t0 = time.perf_counter_ns()
                value = _call(st, value, it)
                t1 = time.perf_counter_ns()
                samples[st.name].append((t1 - t0) / 1e6)
                if len(events) < MAX_TRACE_EVENTS:
                    events.append({"iter": it, "stage": st.name, "start_ns": t0 - t_origin,
                                   "dur_ns": t1 - t0, "thread": tid})
                else:
                    truncated = True
            e2e.append((time.perf_counter_ns() - it_start) / 1e6)
    finally:
        tel_summary = recorder.stop() if recorder else {}
        series = recorder.series() if recorder else []

    result = build_result(
        pipeline_desc=pipeline.describe(), stage_samples_ms=samples, e2e_ms=e2e,
        mode="hardware", pipeline_source=pipeline_source or f"pipeline:{pipeline.name}",
        telemetry_summary=tel_summary, telemetry_series=series, events=events,
        events_truncated=truncated, deadline_ms=deadline_ms, period_ms=period_ms,
        paced=bool(pace and period_ms), idle_power_w=idle_w, pack=pack,
    )
    if attach_ids:
        from .identity import attach_identity
        attach_identity(result, model_path=model_path)
    return result

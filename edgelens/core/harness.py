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

Python's garbage collector is watched for the timed loop (core.gcwatch) and
can be set with gc_mode="normal" | "freeze" | "off". The trace is stored in
integer columns (core.eventbuf), so EdgeLens adds no objects for the
collector to scan while it measures.
"""

import threading
import time

from ..hardware import power as power_mod
from ..hardware import telemetry as telemetry_mod
from .clock import now_ns
from .eventbuf import EventBuffer
from .gcwatch import GcWatch
from .result import build_result


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
                model_path=None, max_seconds=None, gc_mode="normal"):
    if len(pipeline) == 0:
        raise ValueError("Pipeline has no stages.")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    pack, cfg = pipeline.pack, pipeline.config
    # Precedence: explicit deadline > explicit period (an input arriving
    # every P ms must finish within P ms) > the pack's default.
    if deadline_ms is None:
        deadline_ms = period_ms if period_ms is not None else pack.default_deadline_ms(cfg)
    if period_ms is None:
        period_ms = pack.default_period_ms(cfg)
    if pace and not period_ms:
        raise ValueError("pace=True needs a period (period_ms=..., or a pack config that "
                         "defines one, e.g. timeseries sample_rate_hz/window/hop).")

    gcw = GcWatch(gc_mode)               # validates the mode before any work
    stages = pipeline.stages
    src = iter(source) if source is not None else None
    consumed = [0]

    def next_input():
        """Next item from the source. A finite source (list, generator, a
        camera iterator that ends) used to escape as a bare StopIteration
        with no message; say what ran out and how to fix it."""
        if src is None:
            return None
        try:
            item = next(src)
        except StopIteration:
            raise ValueError(
                f"The input source ran out after {consumed[0]} item(s), but warmup + "
                f"iterations needs {warmup + iterations}. Pass a longer source, lower "
                f"iterations/warmup, or use a source that repeats (WindowSource wraps "
                f"around; itertools.cycle(...) for a list)."
            ) from None
        consumed[0] += 1
        return item

    def one_pass(it):
        value = next_input()
        for st in stages:
            value = _call(st, value, it)

    for w in range(warmup):
        one_pass(-1 - w)

    idle_w = measure_idle_power(idle_baseline_s) if idle_baseline_s else None

    samples = {s.name: [] for s in stages}
    e2e, service = [], []
    events = EventBuffer()
    tid = threading.get_ident()
    recorder = None
    if telemetry:
        recorder = telemetry_mod.TelemetryRecorder(interval_s=telemetry_interval_s)
        recorder.start()
    t_origin = now_ns()
    it_end = t_origin
    gcw.start(t_origin)
    try:
        period_ns = int(period_ms * 1e6) if (pace and period_ms) else None
        stop_at = t_origin + int(max_seconds * 1e9) if max_seconds else None
        for it in range(iterations):
            if stop_at is not None and now_ns() >= stop_at:
                break                      # time-boxed run (soak windows)
            gcw.current_iter = it
            value = next_input()
            if period_ns is not None:
                release = t_origin + it * period_ns
                now = now_ns()
                if now < release:
                    time.sleep((release - now) / 1e9)
                it_start = release           # response time counts from release
            else:
                it_start = now_ns()
            work_start = now_ns()  # service time excludes queueing
            for st in stages:
                t0 = now_ns()
                value = _call(st, value, it)
                t1 = now_ns()
                samples[st.name].append((t1 - t0) / 1e6)
                events.add(it, st.name, t0 - t_origin, t1 - t0, tid)
            it_end = now_ns()
            e2e.append((it_end - it_start) / 1e6)
            service.append((it_end - work_start) / 1e6)
    finally:
        gcw.stop()
        tel_summary = recorder.stop() if recorder else {}
        series = recorder.series() if recorder else []

    result = build_result(
        pipeline_desc=pipeline.describe(), stage_samples_ms=samples, e2e_ms=e2e,
        service_ms=service,
        mode="hardware", pipeline_source=pipeline_source or f"pipeline:{pipeline.name}",
        telemetry_summary=tel_summary, telemetry_series=series, events=events.to_dicts(),
        events_truncated=events.truncated, deadline_ms=deadline_ms, period_ms=period_ms,
        paced=bool(pace and period_ms), idle_power_w=idle_w, pack=pack,
        trace_origin_ns=t_origin,
        throughput_window_ms=((it_end - t_origin) / 1e6 if (pace and period_ms and e2e) else None),
    )
    result["gc"] = gcw.summary(e2e, deadline_ms)
    # How the numbers were taken, so a reader can judge and repeat them.
    result["protocol"] = {
        "warmup_iterations": warmup,
        "warmup_policy": "untimed passes through every stage before the timed loop",
        "iterations_requested": iterations,
        "iterations_timed": len(e2e),
        "time_box_s": max_seconds,
        "paced": bool(pace and period_ms),
        "period_ms": period_ms,
        "gc_mode": gc_mode,
        "telemetry_interval_s": telemetry_interval_s if telemetry else None,
        "repeats": 1,
    }
    if attach_ids:
        from .identity import attach_identity
        attach_identity(result, model_path=model_path)
    return result

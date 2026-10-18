"""
edgelens.core.result
----------------------
Builds the single result document every run produces — harness, observer,
--model, --pipeline and demo alike. Everything downstream (diagnose,
report, compare) reads this one shape.

Top-level fields:
    schema_version, edgelens_version, mode, pipeline_source
    pipeline        {name, pack, stages:[{name, role}], config}
    requirements    {deadline_ms, period_ms, paced}
    iterations
    latency         end-to-end stats (min/mean/std/p50..p99_9/max/jitter);
                    in paced mode this is response time incl. queueing
    service_latency the pipeline's own work per iteration (= latency unpaced)
    stage_stats_ms  the same stats per stage
    deadline        miss analysis (None without a deadline)
    backlog         queueing estimate when a period is known and not paced
    pack_metrics    scenario-specific metrics
    telemetry       mean/peak summary (+ power), telemetry_series: the samples
    energy          energy per iteration / iterations per joule
    trace           {events:[{iter, stage, start_ns, dur_ns, thread}], truncated}
    identity        environment / experiment / run ids (attached by the caller)
  Legacy flat fields kept for older readers: stage_avg_ms, total_latency_ms,
  fps, latency_p50_ms, latency_p95_ms, latency_p99_ms.
"""

from .. import __version__
from . import clock as clock_mod
from .schema import SCHEMA_VERSION
from .stats import deadline_stats, latency_stats, legacy_percentile, simulate_backlog

MAX_TRACE_EVENTS = 200_000


def energy_summary(power, iterations, mean_e2e_ms, idle_power_w=None):
    if not power or not power.get("available") or not iterations:
        return {"available": False}
    out = {
        "available": True,
        "method": power.get("method"),
        "power_w_mean": power.get("power_w_mean"),
        "energy_j": power.get("energy_j"),
        # Measured energy of the timed window divided by iterations: in paced
        # mode this includes idle time between periods, which is the true
        # per-sample cost at that rate.
        "energy_per_iteration_j": round(power["energy_j"] / iterations, 6),
        "power_sample_count": power.get("sample_count"),
    }
    out["iterations_per_joule"] = (round(1.0 / out["energy_per_iteration_j"], 3)
                                   if out["energy_per_iteration_j"] > 0 else None)
    if idle_power_w is not None and power.get("power_w_mean") is not None:
        dynamic_w = max(0.0, power["power_w_mean"] - idle_power_w)
        out["idle_power_w"] = round(idle_power_w, 3)
        out["dynamic_power_w"] = round(dynamic_w, 3)
        out["dynamic_energy_per_iteration_j"] = round(dynamic_w * mean_e2e_ms / 1000.0, 6)
    if (power.get("sample_count") or 0) < 3:
        out["low_confidence"] = ("fewer than 3 power samples in the timed window; run more "
                                 "iterations for a trustworthy energy figure")
    return out


def windowed_energy(series, start_ns, end_ns, iterations, idle_power_w=None):
    """Energy over [start_ns, end_ns] only (CLOCK_MONOTONIC), from the
    telemetry series — so start-up (model load, engine build) and the idle
    tail of an observed application don't land in "energy per iteration".
    Power is linearly interpolated at the window edges. Never falls back to
    the whole run: if the series doesn't cover the window, energy is
    reported as unavailable, with the reason."""
    pts = sorted((s["t_mono_ns"], s["power_w"]) for s in series
                 if s.get("power_w") is not None and s.get("t_mono_ns") is not None)
    if not iterations or end_ns <= start_ns:
        return {"available": False, "reason": "no active window"}
    if len(pts) < 2 or pts[0][0] > start_ns or pts[-1][0] < end_ns:
        return {"available": False,
                "reason": "the power series does not cover the active window (first to last "
                          "span); energy per iteration can't be separated from start-up"}

    def at(t):
        import bisect
        k = bisect.bisect_left([p[0] for p in pts], t)
        if k < len(pts) and pts[k][0] == t:
            return pts[k][1]
        (t0, p0), (t1, p1) = pts[k - 1], pts[k]
        return p0 + (p1 - p0) * (t - t0) / (t1 - t0)

    inner = [p for p in pts if start_ns < p[0] < end_ns]
    knots = [(start_ns, at(start_ns))] + inner + [(end_ns, at(end_ns))]
    energy = sum((t1 - t0) / 1e9 * (p0 + p1) / 2.0
                 for (t0, p0), (t1, p1) in zip(knots, knots[1:]))
    window_s = (end_ns - start_ns) / 1e9
    mean_w = energy / window_s
    out = {"available": True, "method": "telemetry series, active window only",
           "window_s": round(window_s, 3), "power_w_mean": round(mean_w, 3),
           "energy_j": round(energy, 4),
           "energy_per_iteration_j": round(energy / iterations, 6),
           "power_sample_count": len(inner)}
    out["iterations_per_joule"] = round(iterations / energy, 3) if energy > 0 else None
    if idle_power_w is not None:
        out["idle_power_w"] = round(idle_power_w, 3)
        out["dynamic_power_w"] = round(max(0.0, mean_w - idle_power_w), 3)
        out["dynamic_energy_per_iteration_j"] = round(
            max(0.0, mean_w - idle_power_w) * window_s / iterations, 6)
    if len(inner) < 3:
        out["low_confidence"] = (f"only {len(inner)} power sample(s) inside the active window "
                                 f"({window_s:.3f} s); run longer for a trustworthy energy "
                                 f"figure")
    return out


def build_result(*, pipeline_desc, stage_samples_ms, e2e_ms, mode, pipeline_source,
                 service_ms=None,
                 telemetry_summary=None, telemetry_series=None, events=None,
                 events_truncated=False, deadline_ms=None, period_ms=None, paced=False,
                 idle_power_w=None, pack=None, extra=None, trace_origin_ns=None,
                 keep_series=True, throughput_window_ms=None, energy_window_ns=None):
    names = [s["name"] for s in pipeline_desc["stages"]]
    stage_stats = {n: latency_stats(stage_samples_ms.get(n, [])) for n in names}
    lat = latency_stats(e2e_ms)
    mean_e2e = lat.get("mean") or 0.0
    # Service time = the pipeline's own work per iteration. Equal to latency
    # when unpaced; in paced mode latency also contains queueing behind
    # earlier overruns, and capacity metrics must use service time.
    service_ms = service_ms if service_ms is not None else e2e_ms

    result = {
        "schema_version": SCHEMA_VERSION,
        "edgelens_version": __version__,
        "mode": mode,
        "pipeline_source": pipeline_source,
        "pipeline": pipeline_desc,
        "requirements": {"deadline_ms": deadline_ms, "period_ms": period_ms, "paced": paced},
        "iterations": len(e2e_ms),
        "latency": lat,
        "service_latency": lat if service_ms is e2e_ms else latency_stats(service_ms),
        "stage_stats_ms": stage_stats,
        "deadline": deadline_stats(e2e_ms, deadline_ms),
        "backlog": (simulate_backlog(service_ms, period_ms) if (period_ms and not paced) else None),
        # ---- legacy flat fields ----
        "stage_avg_ms": {n: (stage_stats[n].get("mean") or 0.0) for n in names},
        "total_latency_ms": round(mean_e2e, 3),
        # Back-to-back runs: throughput = 1 / mean latency. When EdgeLens does
        # NOT drive the loop (paced, markers, attach, ROS 2) iterations are
        # spaced by the application, so throughput is the observed rate over
        # the run window instead (v0.2; see throughput_basis).
        "fps": (round(len(e2e_ms) * 1000.0 / throughput_window_ms, 3)
                if throughput_window_ms else
                (round(1000.0 / mean_e2e, 2) if mean_e2e > 0 else 0.0)),
        "throughput_basis": ("observed rate over the run" if throughput_window_ms
                             else "1 / mean latency (iterations back to back)"),
        "latency_p50_ms": legacy_percentile(e2e_ms, 0.50),
        "latency_p95_ms": legacy_percentile(e2e_ms, 0.95),
        "latency_p99_ms": legacy_percentile(e2e_ms, 0.99),
        # ----
        "telemetry": telemetry_summary or {},
        "telemetry_series": telemetry_series or [],
        "energy": energy_summary((telemetry_summary or {}).get("power"), len(e2e_ms),
                                 mean_e2e, idle_power_w),
        "trace": {"events": events or [], "truncated": bool(events_truncated),
                  "event_count": len(events or [])},
        # Time integrity (v0.2): every timestamp above is CLOCK_MONOTONIC.
        # trace.events[].start_ns is relative to trace_origin_mono_ns, and
        # telemetry_series[].t_s to telemetry_origin_mono_ns, so the two line
        # up exactly: offset = telemetry_origin - trace_origin.
        "clock": dict(clock_mod.describe(),
                      trace_origin_mono_ns=trace_origin_ns,
                      telemetry_origin_mono_ns=(telemetry_summary or {}).get("origin_mono_ns")),
    }
    # Per-iteration end-to-end latencies (v0.2): needed for confidence
    # intervals and drift; bounded like the trace.
    if energy_window_ns is not None and (telemetry_summary or {}).get("power", {}).get("available"):
        result["energy"] = windowed_energy(telemetry_series or [], energy_window_ns[0],
                                           energy_window_ns[1], len(e2e_ms), idle_power_w)
    if keep_series:
        result["latency_series_ms"] = [round(v, 4) for v in e2e_ms[:MAX_TRACE_EVENTS]]
        if service_ms is not e2e_ms:
            # paced: the pipeline's own work per iteration (latency minus the
            # wait for the release time / OS wake-up)
            result["service_series_ms"] = [round(v, 4) for v in service_ms[:MAX_TRACE_EVENTS]]
    if pack is not None:
        result["pack_metrics"] = pack.metrics(result)
    if extra:
        result.update(extra)
    return result

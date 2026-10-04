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


def build_result(*, pipeline_desc, stage_samples_ms, e2e_ms, mode, pipeline_source,
                 service_ms=None,
                 telemetry_summary=None, telemetry_series=None, events=None,
                 events_truncated=False, deadline_ms=None, period_ms=None, paced=False,
                 idle_power_w=None, pack=None, extra=None):
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
        "fps": round(1000.0 / mean_e2e, 2) if mean_e2e > 0 else 0.0,
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
    }
    if pack is not None:
        result["pack_metrics"] = pack.metrics(result)
    if extra:
        result.update(extra)
    return result

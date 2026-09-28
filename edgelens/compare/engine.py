"""
edgelens.compare.engine
--------------------------
Compares two benchmark result JSON files (before/after) and reports the
delta on FPS, latency percentiles, per-stage timing, and telemetry.

This is the feature that turns EdgeLens from "a tool I ran once" into
"a tool I run every time I touch the pipeline" — flagged as
disproportionately high-value for its build cost by both reviewers.
"""

STAGES = ["capture", "preprocess", "h2d_copy", "inference", "d2h_copy", "postprocess"]

# A regression is flagged when FPS drops or P95 latency rises by more
# than this fraction. Deliberately simple/transparent, matching the
# rest of EdgeLens' rule-based (not black-box) philosophy.
REGRESSION_FPS_DROP_PCT = 5.0
REGRESSION_LATENCY_RISE_PCT = 5.0


def _pct_change(before, after):
    if before in (None, 0):
        return None
    return round(((after - before) / before) * 100, 1)


def compare(before, after):
    """
    Args:
        before, after: benchmark result dicts (as produced by
            edgelens.benchmark.runner.run_benchmark / loaded from the
            JSON files `edgelens benchmark` saves).

    Returns a structured comparison dict with per-metric deltas and a
    simple pass/regression verdict.
    """
    b_tel = before.get("telemetry", {}) or {}
    a_tel = after.get("telemetry", {}) or {}

    metrics = {
        "fps": (before.get("fps"), after.get("fps")),
        "total_latency_ms": (before.get("total_latency_ms"), after.get("total_latency_ms")),
        "latency_p50_ms": (before.get("latency_p50_ms"), after.get("latency_p50_ms")),
        "latency_p95_ms": (before.get("latency_p95_ms"), after.get("latency_p95_ms")),
        "latency_p99_ms": (before.get("latency_p99_ms"), after.get("latency_p99_ms")),
        "cpu_percent_mean": (
            b_tel.get("cpu_percent_mean", b_tel.get("cpu_percent")),
            a_tel.get("cpu_percent_mean", a_tel.get("cpu_percent")),
        ),
        "gpu_percent_mean": (
            b_tel.get("gpu_percent_mean", b_tel.get("gpu_percent")),
            a_tel.get("gpu_percent_mean", a_tel.get("gpu_percent")),
        ),
        "max_temp_c": (b_tel.get("max_temp_c"), a_tel.get("max_temp_c")),
    }

    metric_deltas = {}
    for name, (b, a) in metrics.items():
        metric_deltas[name] = {
            "before": b,
            "after": a,
            "pct_change": _pct_change(b, a) if (b is not None and a is not None) else None,
        }

    b_stages = before.get("stage_avg_ms", {})
    a_stages = after.get("stage_avg_ms", {})
    stage_deltas = {}
    for s in STAGES:
        b_ms = b_stages.get(s)
        a_ms = a_stages.get(s)
        stage_deltas[s] = {
            "before_ms": b_ms,
            "after_ms": a_ms,
            "pct_change": _pct_change(b_ms, a_ms) if (b_ms is not None and a_ms is not None) else None,
        }

    verdict, reasons = _verdict(metric_deltas, stage_deltas)

    return {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": metric_deltas,
        "stages": stage_deltas,
        "before_mode": before.get("mode"),
        "after_mode": after.get("mode"),
    }


def _verdict(metric_deltas, stage_deltas):
    reasons = []

    fps_change = metric_deltas["fps"]["pct_change"]
    if fps_change is not None and fps_change <= -REGRESSION_FPS_DROP_PCT:
        reasons.append(f"FPS dropped {fps_change:.1f}%")

    p95_change = metric_deltas["latency_p95_ms"]["pct_change"]
    if p95_change is not None and p95_change >= REGRESSION_LATENCY_RISE_PCT:
        reasons.append(f"P95 latency rose {p95_change:.1f}%")

    if reasons:
        # Identify the stage with the largest latency increase, as a
        # "likely regression source" hint — same evidence-first spirit
        # as the diagnosis engine: point at the number, don't just assert.
        worst_stage, worst_pct = None, 0
        for stage, d in stage_deltas.items():
            pct = d["pct_change"]
            if pct is not None and pct > worst_pct:
                worst_stage, worst_pct = stage, pct
        if worst_stage:
            reasons.append(f"Largest stage regression: {worst_stage} (+{worst_pct:.1f}%)")
        return "REGRESSION", reasons

    return "PASS", reasons

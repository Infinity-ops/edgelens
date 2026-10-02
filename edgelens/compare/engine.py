"""
edgelens.compare.engine
--------------------------
Compares two result documents (before/after) and reports deltas on
throughput, latency percentiles and tail, deadline misses, energy, and
every stage — for any pipeline, not just the classic six stages.

Environment-aware since v0.1.0: if the two runs were taken under different
environments (power mode, clock locking, board, JetPack, runtime versions),
the comparison says so and lists exactly what differs. A 30% "regression"
that is really nvpmodel 10W vs MAXN is not a code regression.
"""

from ..core.schema import SCHEMA_VERSION

# A regression is flagged when FPS drops or P95 latency rises by more than
# this fraction. Deliberately simple/transparent, like the rest of EdgeLens.
REGRESSION_FPS_DROP_PCT = 5.0
REGRESSION_LATENCY_RISE_PCT = 5.0
# Absolute rise in deadline miss ratio (percentage points) that is a regression.
REGRESSION_MISS_RATIO_RISE_PP = 0.1

# Environment fields compared, with readable labels.
_ENV_FIELDS = (
    ("board", "board"), ("l4t", "L4T/JetPack"), ("cuda", "CUDA"), ("tensorrt", "TensorRT"),
    ("python", "Python"), ("power_mode", "power mode (nvpmodel)"),
    ("cpu_clocks_locked", "CPU clocks locked (jetson_clocks)"),
    ("gpu_clocks_locked", "GPU clocks locked (jetson_clocks)"), ("packages", "package versions"),
)


def _pct_change(before, after):
    if before in (None, 0) or after is None:
        return None
    return round(((after - before) / before) * 100, 1)


def environment_diff(before, after):
    """List of {field, before, after} where the two environments differ.
    Empty when they match OR when either file predates environment capture
    (then `comparable` is None: unknown)."""
    b_env, a_env = before.get("environment"), after.get("environment")
    if not b_env or not a_env:
        return None
    diffs = []
    for key, label in _ENV_FIELDS:
        if b_env.get(key) != a_env.get(key):
            diffs.append({"field": key, "label": label,
                          "before": b_env.get(key), "after": a_env.get(key)})
    return diffs


def _get(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def compare(before, after):
    b_tel = before.get("telemetry", {}) or {}
    a_tel = after.get("telemetry", {}) or {}

    metrics = {
        "fps": (before.get("fps"), after.get("fps")),
        "total_latency_ms": (before.get("total_latency_ms"), after.get("total_latency_ms")),
        "latency_p50_ms": (before.get("latency_p50_ms"), after.get("latency_p50_ms")),
        "latency_p95_ms": (before.get("latency_p95_ms"), after.get("latency_p95_ms")),
        "latency_p99_ms": (before.get("latency_p99_ms"), after.get("latency_p99_ms")),
        "latency_max_ms": (_get(before, "latency", "max"), _get(after, "latency", "max")),
        "jitter_ms": (_get(before, "latency", "jitter"), _get(after, "latency", "jitter")),
        "deadline_miss_ratio": (_get(before, "deadline", "miss_ratio"),
                                _get(after, "deadline", "miss_ratio")),
        "power_w_mean": (_get(before, "energy", "power_w_mean"),
                         _get(after, "energy", "power_w_mean")),
        "energy_per_iteration_j": (_get(before, "energy", "energy_per_iteration_j"),
                                   _get(after, "energy", "energy_per_iteration_j")),
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
    metric_deltas = {
        name: {"before": b, "after": a,
               "pct_change": _pct_change(b, a) if (b is not None and a is not None) else None}
        for name, (b, a) in metrics.items()
        if b is not None or a is not None
    }

    # Stages: union of both runs, in the before-run's order, then any new ones.
    b_stages = before.get("stage_avg_ms", {}) or {}
    a_stages = after.get("stage_avg_ms", {}) or {}
    names = list(b_stages) + [n for n in a_stages if n not in b_stages]
    stage_deltas = {}
    for s in names:
        b_ms, a_ms = b_stages.get(s), a_stages.get(s)
        stage_deltas[s] = {
            "before_ms": b_ms, "after_ms": a_ms,
            "pct_change": _pct_change(b_ms, a_ms) if (b_ms is not None and a_ms is not None) else None,
            "status": ("added" if b_ms is None else "removed" if a_ms is None else "both"),
        }

    verdict, reasons = _verdict(metric_deltas, stage_deltas)
    env_diff = environment_diff(before, after)
    b_id, a_id = before.get("identity") or {}, after.get("identity") or {}

    return {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "reasons": reasons,
        "metrics": metric_deltas,
        "stages": stage_deltas,
        "before_mode": before.get("mode"),
        "after_mode": after.get("mode"),
        "environment": {
            "comparable": None if env_diff is None else not env_diff,
            "differences": env_diff or [],
            "before_environment_id": b_id.get("environment_id"),
            "after_environment_id": a_id.get("environment_id"),
        },
        "same_experiment": (b_id.get("experiment_id") == a_id.get("experiment_id")
                            if b_id.get("experiment_id") and a_id.get("experiment_id") else None),
        "run_ids": [b_id.get("run_id"), a_id.get("run_id")],
    }


def _verdict(metric_deltas, stage_deltas):
    reasons = []

    fps_change = (metric_deltas.get("fps") or {}).get("pct_change")
    if fps_change is not None and fps_change <= -REGRESSION_FPS_DROP_PCT:
        reasons.append(f"FPS dropped {fps_change:.1f}%")

    p95_change = (metric_deltas.get("latency_p95_ms") or {}).get("pct_change")
    if p95_change is not None and p95_change >= REGRESSION_LATENCY_RISE_PCT:
        reasons.append(f"P95 latency rose {p95_change:.1f}%")

    miss = metric_deltas.get("deadline_miss_ratio") or {}
    if miss.get("before") is not None and miss.get("after") is not None:
        rise_pp = (miss["after"] - miss["before"]) * 100
        if rise_pp >= REGRESSION_MISS_RATIO_RISE_PP:
            reasons.append(f"Deadline miss ratio rose {rise_pp:.2f} percentage points "
                           f"({miss['before']*100:.2f}% -> {miss['after']*100:.2f}%)")

    if reasons:
        # Point at the number: the stage with the largest latency increase.
        worst_stage, worst_pct = None, 0
        for stage, d in stage_deltas.items():
            pct = d["pct_change"]
            if pct is not None and pct > worst_pct:
                worst_stage, worst_pct = stage, pct
        if worst_stage:
            reasons.append(f"Largest stage regression: {worst_stage} (+{worst_pct:.1f}%)")
        return "REGRESSION", reasons

    return "PASS", reasons

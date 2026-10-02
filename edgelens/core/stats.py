"""
edgelens.core.stats
---------------------
Latency, jitter and deadline statistics.

Percentile honesty rule: a percentile pXX is only reported when the run has
at least one sample beyond it, i.e. n * (1 - p) >= 1. So p99 needs >= 100
samples and p99.9 needs >= 1000. Below that the value is None and the
percentile is listed under "insufficient_samples" — a p99.9 computed from
150 samples is just the maximum with a misleading name.
"""

import math

import numpy as np

PERCENTILES = (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("p99_9", 0.999))


def min_samples_for(p):
    return int(math.ceil(1.0 / (1.0 - p) - 1e-9))


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def latency_stats(values_ms):
    """Summary statistics for a list of latencies in milliseconds."""
    n = len(values_ms)
    if n == 0:
        return {"n": 0}
    a = np.asarray(values_ms, dtype=np.float64)
    out = {
        "n": n,
        "min": _r(a.min()),
        "mean": _r(a.mean()),
        "std": _r(a.std(ddof=1) if n > 1 else 0.0),
        "max": _r(a.max()),
    }
    insufficient = []
    for key, p in PERCENTILES:
        if n >= min_samples_for(p):
            out[key] = _r(np.percentile(a, p * 100))
        else:
            out[key] = None
            insufficient.append(f"{key} (needs >= {min_samples_for(p)} samples)")
    # Jitter = standard deviation of latency. tail_spread = how far the tail
    # sits above the typical case (p99 - p50, or max - p50 when p99 is not
    # supported by the sample count).
    out["jitter"] = out["std"]
    tail = out["p99"] if out["p99"] is not None else out["max"]
    out["tail_spread"] = _r(tail - out["p50"]) if out["p50"] is not None else None
    out["insufficient_samples"] = insufficient
    return out


def legacy_percentile(values_ms, p):
    """Always-defined percentile for the flat legacy fields
    (latency_p50_ms / p95 / p99) so older readers keep working."""
    if not values_ms:
        return 0.0
    return _r(np.percentile(np.asarray(values_ms, dtype=np.float64), p * 100))


def deadline_stats(values_ms, deadline_ms):
    """Deadline-miss analysis against a per-iteration deadline."""
    if deadline_ms is None or not values_ms:
        return None
    misses = [i for i, v in enumerate(values_ms) if v > deadline_ms]
    longest = run = 0
    prev = None
    for i in misses:
        run = run + 1 if prev is not None and i == prev + 1 else 1
        longest = max(longest, run)
        prev = i
    worst = max(values_ms)
    n = len(values_ms)
    return {
        "deadline_ms": deadline_ms,
        "iterations": n,
        "misses": len(misses),
        "miss_ratio": round(len(misses) / n, 6),
        "met": len(misses) == 0,
        "worst_ms": _r(worst),
        "worst_overrun_ms": _r(max(0.0, worst - deadline_ms)),
        "first_miss_iteration": misses[0] if misses else None,
        # Bursts matter more than isolated misses for real-time consumers.
        "max_consecutive_misses": longest,
        "slack_p50_ms": _r(deadline_ms - float(np.percentile(values_ms, 50))),
    }


def simulate_backlog(service_ms, period_ms):
    """Queueing behaviour if inputs arrived strictly every `period_ms` and
    were served one at a time with the MEASURED service times (Lindley's
    recursion: W[n+1] = max(0, W[n] + S[n] - T)).

    This is a derived estimate, labelled as such in the output — use paced
    mode (period_ms with pace=True) for a measured response time instead.
    """
    if not service_ms or not period_ms:
        return None
    wait, waits = 0.0, []
    for s in service_ms:
        waits.append(wait)
        wait = max(0.0, wait + s - period_ms)
    mean_service = float(np.mean(service_ms))
    max_wait = max(waits)
    return {
        "method": "simulated: periodic arrivals + measured service times",
        "period_ms": period_ms,
        "utilization": _r(mean_service / period_ms, 4),
        "stable": mean_service < period_ms,
        "max_queue_wait_ms": _r(max_wait),
        "max_backlog_items": int(math.ceil(max_wait / period_ms)) if max_wait > 0 else 0,
        "final_queue_wait_ms": _r(waits[-1]),
    }

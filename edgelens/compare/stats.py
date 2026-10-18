"""
edgelens.compare.stats
------------------------
Statistical before/after comparison. A single before/after pair on a
Jetson is mostly noise (DVFS, thermals, background load), so a verdict is
based on REPEATED RUNS:

    edgelens benchmark ... --repeat 5 --save base.json     # base.r01.json ... r05
    (change something)
    edgelens benchmark ... --repeat 5 --save new.json
    edgelens compare 'base.r*.json' 'new.r*.json'

Method (deliberately standard and explainable):
  * one value per run (e.g. that run's p99);
  * Welch's t interval for the difference of the two groups' means — the
    run-to-run spread is the noise that matters, and runs are independent
    where iterations within a run are not (thermal / DVFS phases make
    neighbouring latencies correlated);
  * verdict IMPROVED / REGRESSED when the 95% interval excludes zero, else
    NO DETECTABLE CHANGE together with the smallest change this many runs
    could have detected (80% power) — an honest "no difference" says how
    big a difference it could have missed.

Within a single run, `within_run_ci` gives a moving-block bootstrap
interval (blocks keep neighbouring iterations together, which a plain
bootstrap would wrongly treat as independent). It describes that one run's
sampling noise only, so it is shown, never used for a verdict.
"""

import math

import numpy as np

MIN_REPEATS = 3
ALPHA = 0.05

IMPROVED, REGRESSED, NO_CHANGE, INSUFFICIENT = (
    "IMPROVED", "REGRESSED", "NO DETECTABLE CHANGE", "INSUFFICIENT REPEATS")

# Student t quantiles: two-sided 95% (0.975) and one-sided 0.80 (for power).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
         30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
_T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889,
        9: 0.883, 10: 0.879, 12: 0.873, 15: 0.866, 20: 0.860, 30: 0.854, 60: 0.848,
        120: 0.845}
_Z975, _Z80 = 1.960, 0.842

# metric -> (label, unit, "lower" or "higher" is better)
METRICS = {
    "p99": ("Latency p99", "ms", "lower"),
    "p50": ("Latency p50", "ms", "lower"),
    "mean": ("Latency mean", "ms", "lower"),
    "max": ("Latency max", "ms", "lower"),
    "miss_ratio": ("Deadline miss ratio", "", "lower"),
    "energy_mj": ("Energy per iteration", "mJ", "lower"),
    "power_w": ("Mean power", "W", "lower"),
    "fps": ("Throughput", "it/s", "higher"),
}


def _tq(table, z, df):
    """Quantile for (possibly fractional) df by linear interpolation."""
    if df is None or df <= 0 or math.isnan(df):
        return None
    keys = sorted(table)
    if df >= keys[-1]:
        return z + (table[keys[-1]] - z) * keys[-1] / df
    if df <= keys[0]:
        return table[keys[0]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= df <= hi:
            f = (df - lo) / (hi - lo)
            return table[lo] + f * (table[hi] - table[lo])
    return z


def t975(df):
    return _tq(_T975, _Z975, df)


def t80(df):
    return _tq(_T80, _Z80, df)


def run_value(run, metric):
    lat = run.get("latency") or {}
    if metric in ("p99", "p50", "mean", "max"):
        return lat.get(metric)
    if metric == "miss_ratio":
        return (run.get("deadline") or {}).get("miss_ratio")
    if metric == "energy_mj":
        en = run.get("energy") or {}
        if not en.get("available") or en.get("low_confidence"):
            return None
        return round(en["energy_per_iteration_j"] * 1000.0, 4)
    if metric == "power_w":
        en = run.get("energy") or {}
        return en.get("power_w_mean") if en.get("available") else None
    if metric == "fps":
        return run.get("fps") or None
    raise ValueError(f"Unknown metric {metric}")


def welch(a, b):
    """(diff = mean(b) - mean(a), se, df) for two samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    diff = float(b.mean() - a.mean())
    if se2 <= 0:
        return diff, 0.0, float(na + nb - 2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return diff, math.sqrt(se2), float(df)


def compare_metric(before_vals, after_vals, metric):
    label, unit, better = METRICS[metric]
    a = [v for v in before_vals if v is not None]
    b = [v for v in after_vals if v is not None]
    out = {"metric": metric, "label": label, "unit": unit, "better": better,
           "n_before": len(a), "n_after": len(b),
           "before_values": a, "after_values": b}
    if len(a) < MIN_REPEATS or len(b) < MIN_REPEATS:
        out.update(verdict=INSUFFICIENT,
                   reason=f"needs >= {MIN_REPEATS} runs with this metric on each side "
                          f"(have {len(a)} and {len(b)}); benchmark with --repeat 5")
        return out
    ma, mb = float(np.mean(a)), float(np.mean(b))
    diff, se, df = welch(a, b)
    tc = t975(df)
    lo, hi = diff - tc * se, diff + tc * se
    mde = (tc + t80(df)) * se
    rel = (lambda x: round(100.0 * x / ma, 2) if ma else None)
    out.update(before_mean=round(ma, 4), after_mean=round(mb, 4),
               before_cv_pct=round(100 * float(np.std(a, ddof=1)) / ma, 2) if ma else None,
               after_cv_pct=round(100 * float(np.std(b, ddof=1)) / mb, 2) if mb else None,
               change=round(diff, 4), change_pct=rel(diff),
               ci95=[round(lo, 4), round(hi, 4)], ci95_pct=[rel(lo), rel(hi)],
               welch_df=round(df, 1),
               min_detectable_change=round(mde, 4), min_detectable_change_pct=rel(mde))
    if se == 0:
        verdict = NO_CHANGE if diff == 0 else (IMPROVED if (diff < 0) == (better == "lower")
                                               else REGRESSED)
    elif lo > 0 or hi < 0:
        got_lower = hi < 0
        verdict = IMPROVED if got_lower == (better == "lower") else REGRESSED
    else:
        verdict = NO_CHANGE
    out["verdict"] = verdict
    return out


def compare_groups(before_runs, after_runs, metrics=None, primary=None, min_effect_pct=2.0):
    """Compare two groups of repeated runs. Returns a verdict document."""
    if metrics is None:
        metrics = list(METRICS)
    rows = {}
    for m in metrics:
        bv = [run_value(r, m) for r in before_runs]
        av = [run_value(r, m) for r in after_runs]
        if all(v is None for v in bv + av):
            continue
        rows[m] = compare_metric(bv, av, m)
    if primary is None:
        enough = all((r.get("latency") or {}).get("p99") is not None
                     for r in before_runs + after_runs)
        primary = "p99" if enough else "mean"
    prim = rows.get(primary)
    verdict = prim["verdict"] if prim else INSUFFICIENT
    notes = []
    regression_counts = False
    if prim and verdict == REGRESSED:
        if prim.get("change_pct") is not None and abs(prim["change_pct"]) < min_effect_pct:
            notes.append(f"{prim['label']} regressed by {prim['change_pct']:+.2f}%: "
                         f"statistically detectable but below --min-effect "
                         f"{min_effect_pct}%, so not treated as a regression.")
        else:
            regression_counts = True
    return {"verdict": verdict, "primary_metric": primary, "metrics": rows, "notes": notes,
            "regression": regression_counts, "min_effect_pct": min_effect_pct,
            "n_before": len(before_runs), "n_after": len(after_runs)}


def within_run_ci(series_ms, stat="p99", n_boot=400, seed=0, block=None):
    """95% moving-block bootstrap interval of one run's statistic."""
    x = np.asarray(series_ms or [], float)
    n = len(x)
    if n < 20 or (stat == "p99" and n < 100):
        return None
    block = block or max(1, int(round(n ** 0.5)))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts_max = n - block + 1
    fn = {"p99": lambda v: np.percentile(v, 99), "p50": lambda v: np.percentile(v, 50),
          "mean": np.mean}[stat]
    vals = np.empty(n_boot)
    offs = np.arange(block)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:n]
        vals[i] = fn(x[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def environments(runs):
    """Distinct environment ids in a group (should be exactly one)."""
    return sorted({(r.get("identity") or {}).get("environment_id") or "unknown" for r in runs})

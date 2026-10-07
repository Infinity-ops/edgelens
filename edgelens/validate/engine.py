"""
edgelens.validate.engine
--------------------------
"Does this run meet its stated requirements?" — PASS / FAIL / INCONCLUSIVE,
one line per requirement, with the measured value next to the limit.

v0.1 takes requirements as CLI flags (`edgelens validate run.json --p99-ms 10
...`). v0.2 adds contract files and more requirement kinds (quality,
thermal, per-rail power); the verdict model below stays the same.

Honesty rules, same as the rest of EdgeLens:

* A requirement that the run cannot support is NOT_MEASURED, never PASS:
  a p99 limit on a 50-iteration run, an energy limit without a readable
  power sensor, a miss-ratio limit on a run without a deadline.
* A run with any NOT_MEASURED requirement and no FAIL is INCONCLUSIVE,
  never PASS. One FAIL makes the whole verdict FAIL.
* Demo (simulated) results can never validate anything.
* On FAIL, the verdict carries the diagnosis of the same run, so the answer
  to "why?" comes with the answer to "does it work?".
"""

from ..core.schema import SCHEMA_VERSION
from ..core.stats import min_samples_for

PASS, FAIL, NOT_MEASURED, INCONCLUSIVE = "PASS", "FAIL", "NOT_MEASURED", "INCONCLUSIVE"

# requirement key -> (label, unit, direction, how to read the measured value)
# direction "max": measured must be <= limit; "min": measured must be >= limit.
_LATENCY = {"p50_ms": ("p50", 0.50), "p95_ms": ("p95", 0.95), "p99_ms": ("p99", 0.99),
            "p99_9_ms": ("p99_9", 0.999)}

REQUIREMENT_KEYS = ("p50_ms", "p95_ms", "p99_ms", "p99_9_ms", "max_latency_ms",
                    "max_miss_ratio", "min_fps", "max_power_w", "max_energy_mj")


def _row(key, label, limit, measured, unit, status, reason=None, direction="max"):
    return {"requirement": key, "label": label, "limit": limit, "measured": measured,
            "unit": unit, "direction": direction, "status": status, "reason": reason}


def _compare(measured, limit, direction):
    return PASS if (measured <= limit if direction == "max" else measured >= limit) else FAIL


def _check_latency_percentile(result, key, limit):
    field, p = _LATENCY[key]
    lat = result.get("latency") or {}
    measured = lat.get(field)
    label = f"Latency {field.replace('_', '.')}"
    if measured is None:
        need = min_samples_for(p)
        return _row(key, label, limit, None, "ms", NOT_MEASURED,
                    f"needs at least {need} iterations, run had {lat.get('n', result.get('iterations', 0))}; "
                    f"rerun with --iterations {need} or more")
    return _row(key, label, limit, measured, "ms", _compare(measured, limit, "max"))


def _check(result, key, limit):
    if key in _LATENCY:
        return _check_latency_percentile(result, key, limit)

    if key == "max_latency_ms":
        measured = (result.get("latency") or {}).get("max")
        if measured is None:
            return _row(key, "Latency max", limit, None, "ms", NOT_MEASURED, "no latency data")
        return _row(key, "Latency max", limit, measured, "ms", _compare(measured, limit, "max"))

    if key == "max_miss_ratio":
        dl = result.get("deadline")
        if not dl:
            return _row(key, "Deadline miss ratio", limit, None, "", NOT_MEASURED,
                        "the run has no deadline; benchmark with --deadline-ms (or a pack "
                        "that defines one) to measure misses")
        row = _row(key, f"Deadline miss ratio ({dl['deadline_ms']} ms)", limit,
                   dl["miss_ratio"], "", _compare(dl["miss_ratio"], limit, "max"))
        row["detail"] = f"{dl['misses']} of {dl['iterations']} iterations missed"
        return row

    if key == "min_fps":
        measured = result.get("fps")
        if not measured:
            return _row(key, "Throughput", limit, None, "it/s", NOT_MEASURED, "no throughput data",
                        direction="min")
        return _row(key, "Throughput", limit, measured, "it/s", _compare(measured, limit, "min"),
                    direction="min")

    if key in ("max_power_w", "max_energy_mj"):
        en = result.get("energy") or {}
        label, unit = (("Mean power", "W") if key == "max_power_w"
                       else ("Energy per iteration", "mJ"))
        if not en.get("available"):
            pw = (result.get("telemetry") or {}).get("power") or {}
            if pw.get("reason") == "permission_denied":
                reason = f"power sensor is root-only; fix and rerun: {pw.get('fix')}"
            else:
                reason = "no board power sensor readings in this run (Jetson INA3221 needed)"
            return _row(key, label, limit, None, unit, NOT_MEASURED, reason)
        if en.get("low_confidence"):
            return _row(key, label, limit, None, unit, NOT_MEASURED, en["low_confidence"])
        measured = (en["power_w_mean"] if key == "max_power_w"
                    else round(en["energy_per_iteration_j"] * 1000.0, 3))
        return _row(key, label, limit, measured, unit, _compare(measured, limit, "max"))

    raise ValueError(f"Unknown requirement '{key}'. Known: {', '.join(REQUIREMENT_KEYS)}")


def _limiting_stage(result):
    """The stage that contributes most to the tail: largest p99 (falls back to
    the largest mean when p99 isn't supported by the sample count)."""
    stats = result.get("stage_stats_ms") or {}
    best, best_val, basis = None, -1.0, "p99"
    for name, s in stats.items():
        v = s.get("p99")
        if v is not None and v > best_val:
            best, best_val = name, v
    if best is None:
        basis = "mean"
        for name, ms in (result.get("stage_avg_ms") or {}).items():
            if ms is not None and ms > best_val:
                best, best_val = name, ms
    if best is None:
        return None
    s = stats.get(best) or {}
    return {"stage": best, "basis": basis, "p99_ms": s.get("p99"),
            "mean_ms": s.get("mean", (result.get("stage_avg_ms") or {}).get(best))}


def validate(result, requirements):
    """Check a result document against {requirement_key: limit}.

    Returns a verdict document (schema_version'd) with one row per
    requirement, the overall verdict, and — on FAIL — the limiting stage
    and the run's diagnosis.
    """
    reqs = {k: v for k, v in (requirements or {}).items() if v is not None}
    if not reqs:
        raise ValueError("No requirements given. Pass at least one, e.g. --p99-ms 10.")
    for k in reqs:
        if k not in REQUIREMENT_KEYS:
            raise ValueError(f"Unknown requirement '{k}'. Known: {', '.join(REQUIREMENT_KEYS)}")

    rows = [_check(result, k, reqs[k]) for k in REQUIREMENT_KEYS if k in reqs]
    statuses = {r["status"] for r in rows}

    notes = []
    if result.get("mode") == "demo":
        verdict = INCONCLUSIVE
        notes.append("This is a --demo (simulated) result: it cannot validate anything.")
    elif FAIL in statuses:
        verdict = FAIL
    elif NOT_MEASURED in statuses:
        verdict = INCONCLUSIVE
        notes.append("Some requirements could not be measured by this run; see each reason.")
    else:
        verdict = PASS

    ident = result.get("identity") or {}
    out = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "requirements": reqs,
        "checks": rows,
        "notes": notes,
        "run": {
            "pipeline_source": result.get("pipeline_source"),
            "iterations": result.get("iterations"),
            "mode": result.get("mode"),
            "environment_id": ident.get("environment_id"),
            "experiment_id": ident.get("experiment_id"),
            "run_id": ident.get("run_id"),
            "board": (result.get("environment") or {}).get("board"),
            "power_mode": ((result.get("environment") or {}).get("power_mode") or {}).get("name"),
        },
    }
    if verdict == FAIL:
        from ..diagnose.engine import diagnose
        d = diagnose(result)
        out["limiting_stage"] = _limiting_stage(result)
        out["diagnosis"] = {
            "primary": {k: d["primary"].get(k) for k in
                        ("type", "evidence_strength", "detail", "recommendation", "evidence")},
            "also_considered": [{"type": f["type"], "evidence_strength": f["evidence_strength"]}
                                for f in d.get("secondary", [])],
        }
    return out


def exit_code(verdict):
    """0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE (same 0/1/2 shape as `compare`)."""
    return {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}[verdict]

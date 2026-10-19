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

from ..core import integrity
from ..core.schema import SCHEMA_VERSION
from ..core.stats import min_samples_for

PASS, FAIL, NOT_MEASURED, INCONCLUSIVE = "PASS", "FAIL", "NOT_MEASURED", "INCONCLUSIVE"

# requirement key -> (label, unit, direction, how to read the measured value)
# direction "max": measured must be <= limit; "min": measured must be >= limit.
_LATENCY = {"p50_ms": ("p50", 0.50), "p95_ms": ("p95", 0.95), "p99_ms": ("p99", 0.99),
            "p99_9_ms": ("p99_9", 0.999)}

REQUIREMENT_KEYS = ("p50_ms", "p95_ms", "p99_ms", "p99_9_ms", "max_latency_ms",
                    "max_miss_ratio", "min_fps", "max_power_w", "max_energy_mj",
                    # v0.2 Layer 2
                    "max_output_error", "min_decision_agreement", "max_temp_c",
                    "max_latency_drift_pct", "max_mem_growth_mb_per_h",
                    "max_throttled_windows", "max_energy_per_period_mj",
                    # v0.2 Layer 4: decisions in time (edgelens replay / detect)
                    "max_time_to_detect_s", "max_false_alarms_per_h", "max_missed_events",
                    "max_rmse")


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

    if key in ("max_output_error", "min_decision_agreement"):
        eq = result.get("equivalence")
        if not eq:
            return _row(key, "Output equivalence", limit, None,
                        "abs" if key == "max_output_error" else "", NOT_MEASURED,
                        "no output comparison in this run; use a per-backend file from "
                        "`edgelens sweep`", direction="max" if key.startswith("max") else "min")
        if key == "max_output_error":
            return _row(key, f"Max |output error| vs {eq.get('reference_backend')}", limit,
                        eq["max_abs_error"], "abs", _compare(eq["max_abs_error"], limit, "max"))
        agree = eq.get("decision_agreement")
        if agree is None:
            return _row(key, "Decision agreement", limit, None, "", NOT_MEASURED,
                        "outputs have no decision (use --decision / --threshold)",
                        direction="min")
        row = _row(key, f"Decision agreement vs {eq.get('reference_backend')}", limit, agree,
                   "", _compare(agree, limit, "min"), direction="min")
        row["detail"] = (f"{eq.get('decisions_flipped', 0)} of {eq.get('decisions_compared', 0)} "
                         f"decisions flipped ({eq.get('inputs')})")
        return row

    if key == "max_temp_c":
        tel = result.get("telemetry") or {}
        peak = (result.get("soak") or {}).get("max_temp_c") or tel.get("max_temp_c")
        if not peak:
            return _row(key, "Peak temperature", limit, None, "C", NOT_MEASURED,
                        "no temperature readings in this run")
        return _row(key, "Peak temperature", limit, peak, "C", _compare(peak, limit, "max"))

    if key in ("max_latency_drift_pct", "max_mem_growth_mb_per_h", "max_throttled_windows"):
        soak = result.get("soak")
        label = {"max_latency_drift_pct": "Latency drift (p50, first vs last)",
                 "max_mem_growth_mb_per_h": "Memory growth",
                 "max_throttled_windows": "Windows with thermal throttling"}[key]
        unit = {"max_latency_drift_pct": "%", "max_mem_growth_mb_per_h": "MB/h",
                "max_throttled_windows": "windows"}[key]
        if not soak:
            return _row(key, label, limit, None, unit, NOT_MEASURED,
                        "not a soak run; use `edgelens soak --duration 30m ...`")
        if key == "max_latency_drift_pct":
            d = (soak.get("drift") or {}).get("p50") or {}
            if d.get("change_pct") is None:
                return _row(key, label, limit, None, unit, NOT_MEASURED,
                            "needs at least 4 windows with iterations")
            measured = abs(d["change_pct"])
        elif key == "max_mem_growth_mb_per_h":
            m = (soak.get("memory") or {}).get("process_rss_mb_per_h")
            if m is None:
                return _row(key, label, limit, None, unit, NOT_MEASURED,
                            "needs at least 4 windows")
            measured = max(0.0, m)
        else:
            measured = (soak.get("throttling") or {}).get("throttled_windows", 0)
        return _row(key, label, limit, round(measured, 3), unit,
                    _compare(measured, limit, "max"))

    if key == "max_energy_per_period_mj":
        per = result.get("periodic") or {}
        primary = per.get("primary") or {}
        e = primary.get("energy_per_period_mj")
        if e is None:
            return _row(key, "Energy per period", limit, None, "mJ", NOT_MEASURED,
                        "not a periodic run with a power sensor (`edgelens periodic`)")
        return _row(key, f"Energy per period ({primary.get('period_ms')} ms)", limit, e, "mJ",
                    _compare(e, limit, "max"))

    if key in _DECISION:
        return _check_decision(result, key, limit)

    raise ValueError(f"Unknown requirement '{key}'. Known: {', '.join(REQUIREMENT_KEYS)}")


_DECISION = {"max_time_to_detect_s": ("Time to detect (event -> alarm)", "s"),
             "max_false_alarms_per_h": ("False alarms per healthy hour", "/h"),
             "max_missed_events": ("Events missed", "events"),
             "max_rmse": ("Output error vs ground truth (RMSE)", "")}


def _check_decision(result, key, limit):
    """Decision metrics of a replay (one recording) or of `edgelens detect`
    (aggregate over recordings: the worst time to detect)."""
    label, unit = _DECISION[key]
    unit = None if unit == "" else unit
    dec = result.get("decision") or {}
    agg = (dec.get("aggregate") or {}).get("output")
    one = dec.get("output")
    if not agg and not one:
        return _row(key, label, limit, None, unit, NOT_MEASURED,
                    "no decisions in this run: use `edgelens replay` with a spec that has a "
                    "decision rule (and ground truth in the data)")
    if key == "max_time_to_detect_s":
        if agg:
            ttd = (agg.get("time_to_detect_s") or {}).get("max")
            missed, events = agg.get("missed") or 0, agg.get("events") or 0
        else:
            ttd, events = one.get("time_to_detect_s"), int(one.get("event_s") is not None)
            missed = int(bool(one.get("missed")))
        if missed:
            return _row(key, label + " (worst)" if agg else label, limit, None, unit, FAIL,
                        f"{missed} of {events} event(s) never raised the alarm")
        if ttd is None:
            return _row(key, label, limit, None, unit, NOT_MEASURED,
                        "no event in the ground truth (or no ground truth in the data)")
        return _row(key, label + (f" (worst of {events})" if agg else ""), limit, ttd, unit,
                    _compare(ttd, limit, "max"))
    if key == "max_false_alarms_per_h":
        src = agg or one
        v = src.get("false_alarms_per_h")
        if v is None:
            return _row(key, label, limit, None, unit, NOT_MEASURED,
                        "needs ground truth with a healthy level (decision.healthy_level)")
        row = _row(key, label, limit, v, unit, _compare(v, limit, "max"))
        row["detail"] = (f"{src.get('false_alarms')} false alarm(s) in "
                         f"{src.get('healthy_hours')} healthy hours")
        return row
    if key == "max_missed_events":
        src = agg or one
        if agg:
            missed, events = agg.get("missed") or 0, agg.get("events") or 0
        else:
            missed, events = int(bool(one.get("missed"))), int(one.get("event_s") is not None)
        if not events:
            return _row(key, label, limit, None, unit, NOT_MEASURED, "no event in the data")
        row = _row(key, label, limit, missed, unit, _compare(missed, limit, "max"))
        row["detail"] = f"{missed} of {events} missed"
        return row
    v = (agg or {}).get("rmse_mean") if agg else one.get("rmse")
    if v is None:
        return _row(key, label, limit, None, unit, NOT_MEASURED, "no ground truth in the data")
    return _row(key, label + (" (mean over recordings)" if agg else ""), limit, v, unit,
                _compare(v, limit, "max"))


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
    # Measurement integrity (v0.2): an issue that blocks a PASS (e.g. lost
    # marker events) turns that PASS into NOT_MEASURED. A FAIL stays a FAIL:
    # what was observed really happened.
    issues = integrity.check(result)
    for row in rows:
        block = integrity.blocking(issues, row["requirement"])
        if block and row["status"] == PASS:
            row["status"] = NOT_MEASURED
            row["reason"] = f"{block['code']}: {block['detail']}"
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
        "integrity": issues,
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
        out["limiting_stage"], out["diagnosis"] = failure_diagnosis(result, rows)
    return out


# Findings that explain a failed DECISION requirement (time to detect, false
# alarms, misses, accuracy): a stage bottleneck can't explain a 200 s alarm.
DECISION_FINDINGS = ("DECISION_LATENCY", "OUT_OF_ENVELOPE", "CADENCE_MISMATCH", "DATA_GAPS",
                     "WARMUP_PADDING", "DEAD_INPUTS")


def failure_diagnosis(result, rows):
    """(limiting stage, diagnosis) for a FAIL: the stage view when a timing
    requirement failed, the decision view when only decision requirements did."""
    from ..diagnose.engine import diagnose
    d = diagnose(result)
    failed = {r["requirement"] for r in rows if r["status"] == FAIL}
    primary, others = d["primary"], d.get("secondary", [])
    limiting = _limiting_stage(result)
    if failed and failed <= set(_DECISION):
        limiting = None
        ranked = [f for t in DECISION_FINDINGS for f in d["all_findings"] if f["type"] == t]
        if ranked:
            primary, others = ranked[0], ranked[1:3]
    return limiting, {
        "primary": {k: primary.get(k) for k in
                    ("type", "evidence_strength", "detail", "recommendation", "evidence")},
        "also_considered": [{"type": f["type"], "evidence_strength": f["evidence_strength"]}
                            for f in others],
    }


def exit_code(verdict):
    """0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE (same 0/1/2 shape as `compare`)."""
    return {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}[verdict]

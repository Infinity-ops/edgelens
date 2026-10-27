"""
edgelens.stream.detect
------------------------
Decision quality in TIME: when does the system raise the alarm, compared
with when the event really happened, and how often does it cry wolf?

Definitions (all times on the recording's own clock):

    onset          first step where the truth leaves its healthy level
    event          first step where the truth crosses the decision threshold
    time_to_detect first alarm at/after onset minus the event time
                   (negative = the alarm came before the threshold was
                   crossed: an early warning on a degrading system)
    missed         an event happened and no alarm followed the onset
    false alarm    an alarm EPISODE (off -> on) that starts while the truth is
                   healthy; reported per hour of healthy time
    early alarm    alarm time while the system is degraded but not yet past
                   the threshold — NOT a false alarm, and reported apart

The same numbers are computed for the raw model output and for the output
after post-processing (slew limit + mean), so the price of smoothing shows.
"""

import math

import numpy as np


def _alarm(values, dec):
    v = np.asarray(values, dtype=np.float64)
    return v <= dec["threshold"] if dec["alarm_when"] == "below" else v >= dec["threshold"]


def _truth_event(truth, dec):
    return truth <= dec["truth_threshold"] if dec["alarm_when"] == "below" \
        else truth >= dec["truth_threshold"]


def _healthy(truth, dec):
    lvl = dec.get("healthy_level")
    if lvl is None:
        return None
    return truth >= lvl - 1e-9 if dec["alarm_when"] == "below" else truth <= lvl + 1e-9


def _first(mask, t, after=None):
    m = mask if after is None else (mask & (t >= after))
    return float(t[np.argmax(m)]) if m.any() else None


def _episodes(mask):
    """Indices where an alarm episode starts (off -> on)."""
    m = np.asarray(mask, dtype=bool)
    starts = np.nonzero(m & ~np.concatenate([[False], m[:-1]]))[0]
    return starts


def score(t, truth, values, dec, step_s):
    """Detection metrics for one output series."""
    t = np.asarray(t, dtype=np.float64)
    alarm = _alarm(values, dec)
    out = {"alarm_steps": int(alarm.sum()), "alarm_episodes": int(len(_episodes(alarm))),
           "first_alarm_s": _first(alarm, t)}
    if truth is None:
        return out
    truth = np.asarray(truth, dtype=np.float64)
    ev = _truth_event(truth, dec)
    healthy = _healthy(truth, dec)
    onset = _first(~healthy, t) if healthy is not None else None
    event = _first(ev, t)
    first = _first(alarm, t, after=onset) if onset is not None else _first(alarm, t)
    out.update(onset_s=onset, event_s=event, alarm_after_onset_s=first,
               time_to_detect_s=(round(first - event, 3) if (first is not None and
                                                             event is not None) else None),
               detect_from_onset_s=(round(first - onset, 3) if (first is not None and
                                                                onset is not None) else None),
               missed=bool(event is not None and first is None))
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    out["rmse"] = (round(float(np.sqrt(np.mean((v[ok] - truth[ok]) ** 2))), 5)
                   if ok.any() else None)
    if healthy is not None:
        starts = _episodes(alarm)
        fa = [int(s) for s in starts if healthy[s]]
        healthy_h = healthy.sum() * step_s / 3600.0
        out.update(false_alarms=len(fa),
                   false_alarm_at_s=[round(float(t[s]), 2) for s in fa[:10]],
                   false_alarm_time_s=round(float((alarm & healthy).sum() * step_s), 2),
                   healthy_hours=round(float(healthy_h), 4),
                   false_alarms_per_h=(round(len(fa) / healthy_h, 3) if healthy_h > 0 else None))
        degraded_pre = (~healthy) & (~ev)
        out["early_alarm_s"] = round(float((alarm & degraded_pre).sum() * step_s), 2)
    return out


def latency_budget(spec, start_level=None):
    """Where decision time goes, by construction of the pipeline (seconds)."""
    f, o, dec = spec["features"], spec["output"], spec.get("decision")
    step = f["step_s"]
    out = {"step_s": step, "history_s": round(f["sequence"] * step, 3),
           "warmup_s": round((f["sequence"] - 1) * step, 3) if f["padding"] == "zeros" else
           round(f["sequence"] * step, 3),
           "block_wait_max_s": step,
           "mean_delay_s": round((o["mean_steps"] - 1) / 2.0 * step, 3)}
    if o["slew_per_step"] and dec and start_level is not None:
        dist = abs(start_level - dec["threshold"])
        out["slew_floor_s"] = round(math.ceil(dist / o["slew_per_step"]) * step, 3)
        out["slew_from_level"] = round(float(start_level), 4)
    return out


def summarize(t, truth, raw, out_values, states, spec, compute_ms=None):
    """The `decision` block of a replay result."""
    dec = spec.get("decision")
    step = spec["features"]["step_s"]
    states = list(states)
    counts = {}
    for s in states:
        counts[s] = counts.get(s, 0) + 1
    judged = len(states) - counts.get("warming_up", 0)
    summary = {"predictions": len(states), "guard": counts,
               "in_envelope_pct": (round(100.0 * counts.get("valid", 0) / judged, 2)
                                   if judged and any(s in counts for s in
                                                     ("valid", "degraded", "invalid"))
                                   else None)}
    if dec:
        summary["raw"] = score(t, truth, raw, dec, step)
        summary["output"] = score(t, truth, out_values, dec, step)
        level = None
        if truth is not None and len(out_values):
            h = summary["output"].get("onset_s")
            k = int(np.searchsorted(np.asarray(t), h)) if h is not None else len(t)
            pre = np.asarray(out_values[:k])[np.array(states[:k]) != "warming_up"]
            level = float(np.median(pre)) if len(pre) else None
        summary["budget"] = latency_budget(spec, level)
        if compute_ms is not None:
            summary["budget"]["compute_p99_ms"] = compute_ms
        r, o = summary["raw"].get("time_to_detect_s"), summary["output"].get("time_to_detect_s")
        if r is not None and o is not None:
            summary["budget"]["post_processing_cost_s"] = round(o - r, 3)
    return summary


def aggregate(summaries):
    """Across recordings (one fault each): the contract-level numbers."""
    def col(part, key):
        return [s[part][key] for s in summaries
                if s.get(part) and s[part].get(key) is not None]
    out = {"recordings": len(summaries)}
    for part in ("raw", "output"):
        ttd = col(part, "time_to_detect_s")
        fa = col(part, "false_alarms")
        hh = col(part, "healthy_hours")
        missed = sum(1 for s in summaries if (s.get(part) or {}).get("missed"))
        events = sum(1 for s in summaries if (s.get(part) or {}).get("event_s") is not None)
        rm = col(part, "rmse")
        out[part] = {"events": events, "missed": missed,
                     "time_to_detect_s": ({"max": max(ttd), "median": float(np.median(ttd)),
                                           "min": min(ttd)} if ttd else None),
                     "false_alarms": sum(fa) if fa else None,
                     "healthy_hours": round(sum(hh), 4) if hh else None,
                     "false_alarms_per_h": (round(sum(fa) / sum(hh), 3)
                                            if hh and sum(hh) > 0 else None),
                     "early_alarm_s": sum(col(part, "early_alarm_s")) or 0.0,
                     "rmse_mean": round(float(np.mean(rm)), 5) if rm else None}
    env = [s["in_envelope_pct"] for s in summaries if s.get("in_envelope_pct") is not None]
    out["in_envelope_pct_min"] = min(env) if env else None
    return out

"""
edgelens.stream.perturb
-------------------------
Data-level fault injection: what a real sensor bus does to the model's input
(noise, offset, a stuck or dead channel, dropped samples, spikes, a stalled
recorder, a different sample rate) — applied to a recording BEFORE the
pipeline, swept over severity, scored with the same detection metrics.

    edgelens perturb spec.yaml test.csv --faults faults.yaml --save robustness.json

faults.yaml (plain data, shareable between teams — the same file can drive
another tool's fault campaign):

    seed: 1
    faults:
      - name: imu-noise
        kind: noise                   # see KINDS
        columns: ["angular_velocity.*", "linear_acceleration.*"]   # globs
        severity: [0, 0.5, 1, 2]      # unit depends on the kind (KINDS)
        start_s: 0                    # optional: when the fault begins
        duration_s: null              # optional: how long it lasts (null = to the end)

The unperturbed baseline (severity 0; 1 for `rate`) is always included, so
every curve starts from the model's real behaviour.
"""

import fnmatch
import json

import numpy as np

# kind -> what severity means
KINDS = {
    "noise": "Gaussian noise, sigma in units of the column's training scale (IQR / std)",
    "bias": "constant offset, in units of the column's training scale",
    "gain": "multiply by (1 + severity)",
    "stuck": "the channel freezes at its value at start_s (severity = 1: on, 0: off)",
    "dropout": "probability that a row is lost and the previous value is held (0..1)",
    "spike": "amplitude (training scales) of spikes on 1% of rows",
    "gap": "seconds of rows removed at start_s (a stalled recorder or bus)",
    "rate": "sample-rate factor: 0.5 = half as many rows per second, 2 = twice as many",
}


class FaultError(ValueError):
    pass


def load_faults(path):
    with open(path) as f:
        text = f.read()
    if str(path).endswith(".json"):
        doc = json.loads(text)
    else:
        import yaml
        doc = yaml.safe_load(text)
    return parse_faults(doc)


def parse_faults(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("faults"), list) or not doc["faults"]:
        raise FaultError("a fault file has 'faults:' with at least one fault")
    for k in doc:
        if k not in ("seed", "faults", "description"):
            raise FaultError(f"unknown key '{k}' (seed, faults, description)")
    out = []
    for i, f in enumerate(doc["faults"]):
        for k in f:
            if k not in ("name", "kind", "columns", "severity", "start_s", "duration_s"):
                raise FaultError(f"faults[{i}]: unknown key '{k}'")
        if f.get("kind") not in KINDS:
            raise FaultError(f"faults[{i}].kind: one of {', '.join(KINDS)}")
        sev = f.get("severity", [0, 1])
        sev = sev if isinstance(sev, list) else [sev]
        if not all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in sev):
            raise FaultError(f"faults[{i}].severity: numbers")
        if 0 not in sev and f["kind"] != "rate":
            sev = [0] + sev
        if f["kind"] == "rate" and 1 not in sev:
            sev = [1] + sev
        cols = f.get("columns", ["*"])
        out.append({"name": f.get("name") or f"{f['kind']}-{i}", "kind": f["kind"],
                    "columns": cols if isinstance(cols, list) else [cols],
                    "severity": sorted(float(s) for s in sev),
                    "start_s": float(f.get("start_s") or 0.0),
                    "duration_s": (float(f["duration_s"]) if f.get("duration_s") is not None
                                   else None)})
    return {"seed": int(doc.get("seed", 1)), "faults": out}


def _cols(rec, patterns):
    idx = [i for i, c in enumerate(rec.columns) if any(fnmatch.fnmatch(c, p) for p in patterns)]
    if not idx:
        raise FaultError(f"columns {patterns} match none of {rec.columns[:6]}...")
    return idx


def apply(rec, fault, severity, spec, rng):
    """A perturbed COPY of the recording."""
    r = rec.copy()
    kind = fault["kind"]
    if (kind != "rate" and severity == 0) or (kind == "rate" and severity == 1):
        return r
    t = r.time_s - r.time_s[0]
    end = fault["start_s"] + fault["duration_s"] if fault["duration_s"] is not None else np.inf
    rows = (t >= fault["start_s"]) & (t < end)
    if kind == "gap":
        cut = (t >= fault["start_s"]) & (t < fault["start_s"] + severity)
        keep = ~cut
        r.time_s, r.x = r.time_s[keep], r.x[keep]
        if r.truth is not None:
            r.truth = r.truth[keep]
        return r
    if kind == "rate":
        n = len(r)
        idx = np.arange(0, n, 1.0 / severity)
        idx = np.clip(np.round(idx).astype(int), 0, n - 1)
        # the recording keeps its duration; the pipeline still assumes rate_hz
        r.x = r.x[idx]
        r.truth = r.truth[idx] if r.truth is not None else None
        t0 = r.time_s[0]
        r.time_s = t0 + np.arange(len(idx)) * (1.0 / (spec["data"]["rate_hz"] * severity))
        return r
    ci = _cols(r, fault["columns"])
    sc = np.asarray(spec["scaler"]["scale"])[ci]
    sub = r.x[np.ix_(rows, ci)]
    if kind == "noise":
        sub = sub + rng.standard_normal(sub.shape) * severity * sc
    elif kind == "bias":
        sub = sub + severity * sc
    elif kind == "gain":
        sub = sub * (1.0 + severity)
    elif kind == "stuck":
        first = np.argmax(rows) if rows.any() else 0
        sub = np.repeat(r.x[first, ci][None], sub.shape[0], axis=0)
    elif kind == "dropout":
        lost = rng.random(sub.shape[0]) < min(1.0, severity)
        for k in range(1, sub.shape[0]):
            if lost[k]:
                sub[k] = sub[k - 1]
    elif kind == "spike":
        hit = rng.random(sub.shape) < 0.01
        sub = sub + hit * rng.choice([-1.0, 1.0], size=sub.shape) * severity * sc
    r.x[np.ix_(rows, ci)] = sub
    return r


def sweep(recordings, spec, model, faults, guard=None, progress=None, tolerance_s=None):
    """Every fault x severity x recording -> detection metrics. Returns the
    robustness document (curves per fault, severity on x)."""
    from . import detect as det
    from .runner import simulate
    say = progress or (lambda m: None)
    rng_seed = faults["seed"]
    step = spec["features"]["step_s"]
    if tolerance_s is None:
        tolerance_s = 10 * step          # ten model steps later counts as a change
    curves = []
    for f in faults["faults"]:
        points = []
        for sev in f["severity"]:
            per = []
            for k, rec in enumerate(recordings):
                rng = np.random.default_rng(rng_seed + k)
                say(f"{f['name']} severity {sev:g}: {rec.name}")
                pr = apply(rec, f, sev, spec, rng)
                t, truth, raw, out, states = simulate(pr, spec, model, guard)
                s = det.summarize(t, truth, raw, out, states, spec)
                s["recording"] = rec.name
                per.append(s)
            agg = det.aggregate(per)
            o = agg.get("output") or {}
            ttd = o.get("time_to_detect_s") or {}
            points.append({"severity": sev, "rmse": o.get("rmse_mean"),
                           "time_to_detect_max_s": ttd.get("max"),
                           "time_to_detect_median_s": ttd.get("median"),
                           "missed": o.get("missed"), "events": o.get("events"),
                           "false_alarms": o.get("false_alarms"),
                           "false_alarms_per_h": o.get("false_alarms_per_h"),
                           "in_envelope_pct_min": agg.get("in_envelope_pct_min"),
                           "per_recording": [{"recording": s["recording"],
                                              "rmse": (s.get("output") or {}).get("rmse"),
                                              "time_to_detect_s": (s.get("output") or {}).get(
                                                  "time_to_detect_s"),
                                              "false_alarms": (s.get("output") or {}).get(
                                                  "false_alarms"),
                                              "in_envelope_pct": s.get("in_envelope_pct")}
                                             for s in per]})
        curves.append({"fault": f, "meaning": KINDS[f["kind"]], "points": points,
                       "breaks_at": _breaking_point(points, tolerance_s,
                                                    1.0 if f["kind"] == "rate" else 0.0)})
    return {"kind": "edgelens-robustness", "version": 1, "spec": spec["name"],
            "step_s": step, "tolerance_s": tolerance_s, "seed": rng_seed,
            "recordings": [r.source for r in recordings],
            "curves": curves}


def _breaking_point(points, tolerance_s, neutral=0.0):
    """Mildest severity (closest to the unperturbed one) where the decision
    changes: a miss, a new false alarm, or a detection more than tolerance_s
    later than the baseline."""
    base = next((p for p in points if p["severity"] == neutral), None)
    if base is None:
        return None
    for p in sorted(points, key=lambda q: abs(q["severity"] - neutral)):
        if p is base:
            continue
        reasons = []
        if (p.get("missed") or 0) > (base.get("missed") or 0):
            reasons.append("fault missed")
        if (p.get("false_alarms") or 0) > (base.get("false_alarms") or 0):
            reasons.append("false alarm")
        a, b = base.get("time_to_detect_max_s"), p.get("time_to_detect_max_s")
        if a is not None and b is not None and b - a > tolerance_s:
            reasons.append(f"detection {b - a:+.0f} s later")
        if reasons:
            return {"severity": p["severity"], "why": reasons}
    return None


def markdown(doc):
    L = [f"# Robustness — {doc['spec']}", "",
         f"Recordings: {len(doc['recordings'])} · seed {doc['seed']} · severity 0 = unperturbed", ""]
    for c in doc["curves"]:
        f = c["fault"]
        L.append(f"## {f['name']} ({f['kind']}: {c['meaning']})")
        L.append("")
        L.append(f"Columns: `{', '.join(f['columns'])}`"
                 + (f" · from {f['start_s']:g} s" if f["start_s"] else "")
                 + (f" for {f['duration_s']:g} s" if f["duration_s"] is not None else ""))
        L.append("")
        L.append("| Severity | RMSE | Time to detect (max) | Missed | False alarms | In envelope (min) |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for p in c["points"]:
            ttd = p["time_to_detect_max_s"]
            env = p["in_envelope_pct_min"]
            L.append(f"| {p['severity']:g} | {_n(p['rmse'], 4)} | "
                     f"{'—' if ttd is None else format(ttd, '.1f') + ' s'} | "
                     f"{p['missed']}/{p['events']} | {_n(p['false_alarms'], 0)} | "
                     f"{'—' if env is None else format(env, 'g') + '%'} |")
        b = c["breaks_at"]
        L.append("")
        L.append(f"**Breaks at severity {b['severity']:g}** ({', '.join(b['why'])})." if b
                 else "**No decision change** over the tested severities.")
        L.append("")
    return "\n".join(L) + "\n"


def _n(v, d):
    return "—" if v is None else (f"{v:.{d}f}" if d else str(v))

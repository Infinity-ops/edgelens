"""
edgelens.stream.spec
----------------------
The stream spec: one file that describes a windowed time-series pipeline
exactly as it was trained, so it can be replayed, timed, checked and
attacked without the training code.

    name: motor-health
    data:
      time_column: timestamp          # seconds, or a date-time text
      rate_hz: 20                     # rows per second the model was trained on
      columns: [imu.ax, imu.ay, ...]  # model inputs, in training order
      target: {columns: [m1, m2, m3, m4], reduce: min, scale: 0.01}   # optional truth
    scaler: {file: scaler.json}       # {center, scale} per column; or kind: none
    features:
      block: 15                       # rows per model step (non-overlapping)
      stats: [min, std, max, mean, last]
      sequence: 100                   # steps per model input -> [1, 100, 5 x columns]
      padding: zeros                  # as trained (zeros) | none (wait for a full window)
    model: {onnx: model.onnx, provider: CPUExecutionProvider}
    output: {slew_per_step: 0.01, mean_steps: 50}
    decision: {threshold: 0.5, alarm_when: below, healthy_level: 1.0}
    deployment: {period_ms: 1000}     # how often the deployed system runs the model
    envelope: {file: envelope.json}   # optional: `edgelens envelope fit`

Unknown keys are errors: a typo must never silently change what is measured.
Relative paths are relative to the spec file.
"""

import difflib
import json
import os

STATS = ("min", "max", "mean", "std", "last", "first", "median", "range")
REDUCE = ("min", "max", "mean", "first")


class SpecError(ValueError):
    pass


TOP = ("name", "version", "description", "data", "scaler", "features", "model", "output",
       "decision", "deployment", "envelope")
KEYS = {
    "data": ("time_column", "rate_hz", "columns", "target", "max_gap_s"),
    "target": ("columns", "reduce", "scale"),
    "scaler": ("file", "center", "scale", "kind", "columns"),
    "features": ("block", "stats", "sequence", "padding", "layout"),
    "model": ("onnx", "python", "provider", "output_index", "threads"),
    "output": ("slew_per_step", "mean_steps"),
    "decision": ("threshold", "alarm_when", "healthy_level", "truth_threshold"),
    "deployment": ("period_ms",),
    "envelope": ("file", "invalid_pct"),
}


def _check_keys(where, obj, known):
    if not isinstance(obj, dict):
        raise SpecError(f"{where}: expected a mapping, got {type(obj).__name__}.")
    for k in obj:
        if k not in known:
            near = difflib.get_close_matches(str(k), known, n=1)
            raise SpecError(f"Unknown key '{k}' in {where}."
                            + (f" Did you mean '{near[0]}'?" if near else "")
                            + f" Known: {', '.join(known)}")


def _num(where, v, positive=True, integer=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SpecError(f"{where}: expected a number, got {v!r}.")
    if integer and int(v) != v:
        raise SpecError(f"{where}: expected a whole number, got {v!r}.")
    if positive and v <= 0:
        raise SpecError(f"{where}: must be > 0.")
    return int(v) if integer else float(v)


def _read_doc(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SpecError(f"Cannot read {path}: {e.strerror or e}") from None
    text = raw.decode("utf-8")
    if str(path).lower().endswith(".json"):
        try:
            return json.loads(text), raw
        except ValueError as e:
            raise SpecError(f"{path} is not valid JSON: {e}") from None
    import yaml
    try:
        return yaml.safe_load(text), raw
    except yaml.YAMLError as e:
        raise SpecError(f"{path} is not valid YAML: {e}") from None


def parse(doc, base_dir="."):
    """Validate a spec document and return the normalized spec (a dict)."""
    _check_keys("the stream spec", doc, TOP)
    for need in ("data", "features"):
        if need not in doc:
            raise SpecError(f"'{need}:' is required (see `edgelens stream init`).")
    d = doc["data"]
    _check_keys("data", d, KEYS["data"])
    cols = d.get("columns")
    if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
        raise SpecError("data.columns: a list of column names (the model inputs, in order).")
    if len(set(cols)) != len(cols):
        raise SpecError("data.columns: a column is listed twice.")
    rate = _num("data.rate_hz", d.get("rate_hz", 0))
    out = {"name": str(doc.get("name") or "stream"), "version": doc.get("version"),
           "description": doc.get("description"), "base_dir": os.path.abspath(base_dir)}
    out["data"] = {"time_column": str(d.get("time_column", "timestamp")), "rate_hz": rate,
                   "columns": list(cols),
                   "max_gap_s": _num("data.max_gap_s", d.get("max_gap_s", 20.0 / rate)),
                   "target": None}
    if d.get("target") is not None:
        t = d["target"]
        _check_keys("data.target", t, KEYS["target"])
        tc = t.get("columns")
        if isinstance(tc, str):
            tc = [tc]
        if not tc or not all(isinstance(c, str) for c in tc):
            raise SpecError("data.target.columns: one or more column names.")
        red = t.get("reduce", "min")
        if red not in REDUCE:
            raise SpecError(f"data.target.reduce: one of {', '.join(REDUCE)}.")
        out["data"]["target"] = {"columns": list(tc), "reduce": red,
                                 "scale": _num("data.target.scale", t.get("scale", 1.0))}

    f = doc["features"]
    _check_keys("features", f, KEYS["features"])
    stats = f.get("stats", ["mean"])
    if not isinstance(stats, list) or not stats or any(s not in STATS for s in stats):
        raise SpecError(f"features.stats: a list from {', '.join(STATS)}.")
    block = _num("features.block", f.get("block", 1), integer=True)
    if "std" in stats and block < 2:
        raise SpecError("features.stats: 'std' needs a block of at least 2 rows.")
    padding = f.get("padding", "zeros")
    if padding not in ("zeros", "none"):
        raise SpecError("features.padding: zeros (as most training code does) or none.")
    layout = f.get("layout", "stat-major")
    if layout not in ("stat-major", "column-major"):
        raise SpecError("features.layout: stat-major ([all mins, all stds, ...]) or "
                        "column-major ([col1 stats, col2 stats, ...]).")
    out["features"] = {"block": block, "stats": list(stats),
                       "sequence": _num("features.sequence", f.get("sequence", 1), integer=True),
                       "padding": padding, "layout": layout}
    out["features"]["width"] = len(stats) * len(cols)
    out["features"]["step_s"] = block / rate

    sc = doc.get("scaler") or {"kind": "none"}
    _check_keys("scaler", sc, KEYS["scaler"])
    if sc.get("file"):
        p = os.path.join(base_dir, sc["file"])
        try:
            with open(p) as fh:
                sd = json.load(fh)
        except (OSError, ValueError) as e:
            raise SpecError(f"scaler.file {p}: {e}") from None
        sc = dict(sd, file=sc["file"])
    if sc.get("kind", "affine" if "center" in sc else "none") == "none" and "center" not in sc:
        out["scaler"] = {"kind": "none", "center": [0.0] * len(cols), "scale": [1.0] * len(cols)}
    else:
        center, scale = sc.get("center"), sc.get("scale")
        if not isinstance(center, list) or not isinstance(scale, list) or \
                len(center) != len(cols) or len(scale) != len(cols):
            raise SpecError(f"scaler: 'center' and 'scale' need one number per column "
                            f"({len(cols)}).")
        if sc.get("columns") and list(sc["columns"]) != list(cols):
            raise SpecError("scaler: its columns are not data.columns in the same order — "
                            "a scaler applied to the wrong column is a silent accuracy bug.")
        if any(float(s) == 0 for s in scale):
            raise SpecError("scaler.scale: a 0 would divide by zero (sklearn stores 1 for "
                            "constant columns).")
        out["scaler"] = {"kind": sc.get("kind", "affine"), "center": [float(v) for v in center],
                         "scale": [float(v) for v in scale], "file": sc.get("file")}

    m = doc.get("model") or {}
    _check_keys("model", m, KEYS["model"])
    if m.get("onnx") and m.get("python"):
        raise SpecError("model: give onnx OR python, not both.")
    out["model"] = {"onnx": os.path.join(base_dir, m["onnx"]) if m.get("onnx") else None,
                    "python": m.get("python"), "provider": m.get("provider",
                                                                 "CPUExecutionProvider"),
                    "output_index": int(m.get("output_index", 0)),
                    "threads": m.get("threads")}
    o = doc.get("output") or {}
    _check_keys("output", o, KEYS["output"])
    out["output"] = {"slew_per_step": (_num("output.slew_per_step", o["slew_per_step"])
                                       if o.get("slew_per_step") is not None else None),
                     "mean_steps": _num("output.mean_steps", o.get("mean_steps", 1),
                                        integer=True)}
    dec = doc.get("decision") or {}
    _check_keys("decision", dec, KEYS["decision"])
    if dec:
        if "threshold" not in dec:
            raise SpecError("decision.threshold is required.")
        when = dec.get("alarm_when", "below")
        if when not in ("below", "above"):
            raise SpecError("decision.alarm_when: below or above.")
        thr = _num("decision.threshold", dec["threshold"], positive=False)
        out["decision"] = {"threshold": thr, "alarm_when": when,
                           "truth_threshold": _num("decision.truth_threshold",
                                                   dec.get("truth_threshold", thr), False),
                           "healthy_level": (_num("decision.healthy_level",
                                                  dec["healthy_level"], False)
                                             if dec.get("healthy_level") is not None else None)}
    else:
        out["decision"] = None
    dep = doc.get("deployment") or {}
    _check_keys("deployment", dep, KEYS["deployment"])
    out["deployment"] = {"period_ms": (_num("deployment.period_ms", dep["period_ms"])
                                       if dep.get("period_ms") is not None else None)}
    env = doc.get("envelope") or {}
    _check_keys("envelope", env, KEYS["envelope"])
    out["envelope"] = {"file": os.path.join(base_dir, env["file"]) if env.get("file") else None,
                       "invalid_pct": _num("envelope.invalid_pct", env.get("invalid_pct", 10.0))}
    return out


def load_spec(path, ignore_scaler=False):
    """Read and validate a .yaml/.json stream spec. Returns (spec, sha256).
    ignore_scaler: for fitting the scaler the spec names (it may not exist yet)."""
    import hashlib
    doc, raw = _read_doc(path)
    if not isinstance(doc, dict):
        raise SpecError(f"{path}: a stream spec is a mapping (key: value).")
    if ignore_scaler:
        doc = dict(doc, scaler={"kind": "none"})
    spec = parse(doc, base_dir=os.path.dirname(os.path.abspath(path)))
    spec["file"] = os.path.basename(str(path))
    spec["sha256"] = hashlib.sha256(raw).hexdigest()
    return spec, spec["sha256"]


def describe(spec):
    """Human summary: what the model sees and how fast decisions can be."""
    f, d = spec["features"], spec["data"]
    hist = f["sequence"] * f["step_s"]
    lines = [f"{len(d['columns'])} columns at {d['rate_hz']:g} Hz -> one model step every "
             f"{f['block']} rows ({f['step_s']:g} s)",
             f"model input [1, {f['sequence']}, {f['width']}] = {hist:g} s of history "
             f"({', '.join(f['stats'])} per column)"]
    if f["padding"] == "zeros" and f["sequence"] > 1:
        lines.append(f"warm-up: the first {(f['sequence'] - 1) * f['step_s']:g} s are padded "
                     f"with zeros (scaled space = the training centre)")
    o = spec["output"]
    if o["slew_per_step"]:
        lines.append(f"output slew limit {o['slew_per_step']:g} per step = "
                     f"{o['slew_per_step'] / f['step_s']:g} per second")
    if o["mean_steps"] > 1:
        lines.append(f"output mean over {o['mean_steps']} steps "
                     f"({o['mean_steps'] * f['step_s']:g} s; delay ~"
                     f"{(o['mean_steps'] - 1) / 2 * f['step_s']:g} s)")
    p = spec["deployment"]["period_ms"]
    if p:
        lines.append(f"deployed every {p:g} ms (trained step {f['step_s'] * 1000:g} ms)")
    return lines


TEMPLATE = """\
# EdgeLens stream spec: a windowed time-series pipeline, exactly as trained.
# Check it:   edgelens stream check THIS_FILE
# Replay:     edgelens replay THIS_FILE recording.csv --save run.json
name: my-sensor-model
version: "1"
data:
  time_column: timestamp          # seconds, or date-time text (2025-08-26 15:48:52.53)
  rate_hz: 20                     # rows per second in the training data
  columns: [ax, ay, az, current]  # model inputs, in training order
  # target: {columns: [health], reduce: min, scale: 0.01}   # ground truth, if recorded
scaler: {kind: none}              # or {file: scaler.json}: edgelens stream fit-scaler
features:
  block: 15                       # rows per model step (non-overlapping blocks)
  stats: [min, std, max, mean, last]
  sequence: 100                   # steps per model input
  padding: zeros                  # what training did before the window was full
model:
  onnx: model.onnx
  provider: CPUExecutionProvider
output:
  slew_per_step: 0.01             # max change per step (omit for none)
  mean_steps: 50                  # rolling mean (1 = none)
decision:
  threshold: 0.5
  alarm_when: below
  healthy_level: 1.0              # truth at/above this counts as healthy (false alarms)
# deployment: {period_ms: 1000}   # how often the real system calls the model
# envelope: {file: envelope.json} # edgelens envelope fit
"""

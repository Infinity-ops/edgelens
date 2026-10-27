"""
edgelens.stream.runner
------------------------
`edgelens replay`: a recording goes through the spec's pipeline one block at a
time, exactly as on the robot, inside the EdgeLens harness:

    ingest -> features -> inference -> postprocess -> decision

so every step is timed per stage (and on a Jetson the board's power,
clocks and temperature are recorded alongside), and the decisions are kept
with the recording's ground truth for `edgelens detect`. The result is an
ordinary EdgeLens result: validate / contract / compare / export --perfetto
work on it.

simulate() is the same arithmetic without the harness (batch features, batched
inference): for perturbation sweeps, where only the decisions matter.
"""

import os

import numpy as np

from . import detect as det
from . import envelope as envmod
from .data import audit
from .features import StreamFeaturizer, OutputFilter, batch_features, filter_series, windows

MAX_SERIES = 200_000


def _guard(spec):
    if spec["envelope"]["file"]:
        return envmod.Guard(envmod.load(spec["envelope"]["file"], spec), spec)
    return None


def _blocks(rec, spec):
    W = spec["features"]["block"]
    n = len(rec) // W
    gmax = spec["data"]["max_gap_s"]
    prev_end = None
    for k in range(n):
        sl = slice(k * W, (k + 1) * W)
        ts = rec.time_s[sl]
        gap = bool((np.diff(ts) > gmax).any()) or (prev_end is not None and ts[0] - prev_end > gmax)
        prev_end = ts[-1]
        yield {"k": k, "rows": rec.x[sl], "t": float(ts[-1] - rec.time_s[0]),
               "truth": float(rec.truth[(k + 1) * W - 1]) if rec.truth is not None else None,
               "gap": gap}


def _decide_states(spec, guard, window, warming, gap):
    if guard is None:
        if warming:
            return "warming_up"
        return "gap" if gap else "unchecked"
    if window is None:
        return "warming_up"
    return guard.state(window[0, -1], warming_up=warming, gap=gap)[0]


def build_pipeline(rec, spec, model, mode="incremental", guard=None):
    """(Pipeline, log) — the log fills while the pipeline runs."""
    from ..core.pipeline import Pipeline
    feat = StreamFeaturizer(spec, mode=mode)
    filt = OutputFilter(spec)
    dec = spec.get("decision")
    log = {"t": [], "truth": [], "raw": [], "out": [], "state": [], "alarm": []}
    pad_warm = spec["features"]["padding"] == "zeros"
    seq = spec["features"]["sequence"]
    pipe = Pipeline(spec["name"], pack="custom",
                    trained_step_ms=round(spec["features"]["step_s"] * 1000.0, 6),
                    deployment_period_ms=spec["deployment"]["period_ms"],
                    provider=model.info.get("provider"),
                    window=[seq, spec["features"]["width"]], features_mode=mode,
                    n_columns=len(spec["data"]["columns"]))

    def ingest(item):
        return item

    def features(item):
        item["window"] = feat.push(item["rows"])
        item["warming"] = (feat.steps < seq) if pad_warm else item["window"] is None
        return item

    def inference(item):
        item["raw"] = model(item["window"]) if item["window"] is not None else float("nan")
        return item

    def postprocess(item):
        item["out"] = filt(item["raw"]) if item["window"] is not None else float("nan")
        return item

    def decision(item):
        st = _decide_states(spec, guard, item["window"], item["warming"], item["gap"])
        alarm = False
        if dec and item["window"] is not None:
            v = item["out"]
            alarm = v <= dec["threshold"] if dec["alarm_when"] == "below" else v >= dec["threshold"]
        if len(log["t"]) < MAX_SERIES:
            log["t"].append(item["t"])
            log["truth"].append(item["truth"])
            log["raw"].append(item["raw"])
            log["out"].append(item["out"])
            log["state"].append(st)
            log["alarm"].append(bool(alarm))
        return alarm

    for name, fn, role in (("ingest", ingest, "input"), ("features", features, "preprocess"),
                           ("inference", inference, "inference"),
                           ("postprocess", postprocess, "postprocess"),
                           ("decision", decision, "decision")):
        pipe.add_stage(name, fn, role=role)
    return pipe, log


def replay(rec, spec, model, mode="incremental", speed=None, deadline_ms=None,
           telemetry=True, max_steps=None, train_constant=None):
    """Run one recording through the spec. speed=None: as fast as possible
    (compute cost); speed=1.0: real time (one step every step_s); 10: ten
    times faster than real time. Returns an EdgeLens result document."""
    from ..core.harness import run_harness
    from ..core.identity import attach_identity
    guard = _guard(spec)
    pipe, log = build_pipeline(rec, spec, model, mode=mode, guard=guard)
    n = len(rec) // spec["features"]["block"]
    if max_steps:
        n = min(n, int(max_steps))
    if n < 1:
        raise ValueError(f"{rec.source}: fewer rows ({len(rec)}) than one block "
                         f"({spec['features']['block']})")
    # warm the model up outside the timed window (first call allocates)
    shape = (1, spec["features"]["sequence"], spec["features"]["width"])
    for _ in range(3):
        model(np.zeros(shape, np.float32))
    period_ms = spec["features"]["step_s"] * 1000.0 / speed if speed else None
    if deadline_ms is None:
        # a step must be done before the next one is due: the (accelerated) period
        # when paced, otherwise the deployed period or the trained step
        deadline_ms = period_ms or spec["deployment"]["period_ms"] or \
            spec["features"]["step_s"] * 1000.0
    res = run_harness(pipe, iterations=n, warmup=0, source=_blocks(rec, spec),
                      deadline_ms=deadline_ms, period_ms=period_ms, pace=bool(speed),
                      telemetry=telemetry, attach_ids=False,
                      pipeline_source=(f"replay:{spec['name']}:"
                                       f"{model.info.get('file') or model.info.get('ref')}"
                                       f" ({model.info.get('provider')})"))
    attach_identity(res, model_path=spec["model"]["onnx"])
    compute = (res.get("service_latency") or {}).get("p99")
    t = np.asarray(log["t"])
    truth = np.asarray(log["truth"], dtype=np.float64) if rec.truth is not None else None
    summary = det.summarize(t, truth, log["raw"], log["out"], log["state"], spec,
                            compute_ms=compute)
    res["stream"] = _stream_block(rec, spec, model, mode, speed, train_constant)
    res["decision"] = dict(summary, series=_series(log, truth is not None))
    return res


def _stream_block(rec, spec, model, mode, speed, train_constant=None):
    from .spec import describe
    return {"spec": {"name": spec["name"], "version": spec.get("version"),
                     "file": spec.get("file"), "sha256": spec.get("sha256")},
            "recording": {"file": os.path.basename(rec.source or ""), "rows": len(rec)},
            "audit": audit(rec, spec, train_constant),
            "model": model.info, "features_mode": mode,
            "speed": speed, "trained_step_s": spec["features"]["step_s"],
            "deployment_period_ms": spec["deployment"]["period_ms"],
            "decision_rule": spec.get("decision"), "output_rule": spec["output"],
            "sequence": spec["features"]["sequence"], "padding": spec["features"]["padding"],
            "summary": describe(spec)}


def _series(log, with_truth):
    r = lambda v: None if v is None or v != v else round(float(v), 5)  # noqa: E731
    s = {"t_s": [round(v, 3) for v in log["t"]], "raw": [r(v) for v in log["raw"]],
         "out": [r(v) for v in log["out"]], "state": log["state"]}
    if with_truth:
        s["truth"] = [r(v) for v in log["truth"]]
    return s


def simulate(rec, spec, model, guard=None):
    """Decisions for a whole recording without timing (batched). Returns
    (t, truth, raw, out, states)."""
    F, info = batch_features(rec, spec)
    wins = windows(F, spec)
    raw = model.predict_many(wins)
    seq = spec["features"]["sequence"]
    pad = spec["features"]["padding"] == "zeros"
    if not pad:                       # no prediction until the window is full
        raw = np.concatenate([np.full(seq - 1, np.nan), raw])
    out = np.full(len(raw), np.nan)
    valid = ~np.isnan(raw)
    out[valid] = filter_series(raw[valid], spec)
    states = []
    for k in range(len(F)):
        warming = k < seq - 1
        if guard is None:
            states.append("warming_up" if warming else ("gap" if info["gap"][k] else "unchecked"))
        else:
            states.append(guard.state(F[k], warming_up=warming, gap=bool(info["gap"][k]))[0])
    t = info["t_end"] - rec.time_s[0]
    return t, info["truth"], raw, out, states

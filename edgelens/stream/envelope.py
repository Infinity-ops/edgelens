"""
edgelens.stream.envelope
--------------------------
The operating envelope: the range of every model input seen in training,
and a runtime guard that says, for each prediction, whether the model is
answering about data it has seen.

    edgelens envelope fit   spec.yaml train/*.csv --out envelope.json
    edgelens envelope check spec.yaml test/*.csv          (which inputs leave it)

Per prediction the guard reports one state:

    warming_up  the window still holds padding (the model sees zeros, not data)
    gap         a time gap inside the newest block (the recording/bus stalled)
    invalid     NaN, or more than invalid_pct % of the newest step's inputs
                outside the envelope
    degraded    some inputs outside, at most invalid_pct %
    valid       every input of the newest step inside the envelope

An input that was CONSTANT in training (a dead sensor channel, a z-position
of a ground robot) is "outside" as soon as it changes at all: the model never
learned what it means.
"""

import json
import os

import numpy as np

from .features import batch_features, feature_names

STATES = ("valid", "degraded", "invalid", "gap", "warming_up")


def fit(recordings, spec, quantile=0.001, margin=0.1):
    F = np.vstack([batch_features(r, spec)[0] for r in recordings])
    lo = np.nanquantile(F, quantile, axis=0)
    hi = np.nanquantile(F, 1.0 - quantile, axis=0)
    fmin, fmax = np.nanmin(F, axis=0), np.nanmax(F, axis=0)
    span = hi - lo
    lo_w, hi_w = lo - margin * span, hi + margin * span
    const = fmax == fmin
    tol = 1e-6 + 1e-6 * np.abs(fmin)
    lo_w = np.where(const, fmin - tol, lo_w)
    hi_w = np.where(const, fmax + tol, hi_w)
    names = feature_names(spec)
    return {"kind": "edgelens-envelope", "version": 1, "spec": spec.get("name"),
            "columns": spec["data"]["columns"], "stats": spec["features"]["stats"],
            "layout": spec["features"]["layout"], "features": names,
            "lo": lo_w.tolist(), "hi": hi_w.tolist(),
            "quantile": quantile, "margin": margin, "steps": int(F.shape[0]),
            "constant_features": [n for n, c in zip(names, const) if c],
            "fitted_on": [os.path.basename(r.source or "") for r in recordings]}


def load(path, spec=None):
    with open(path) as f:
        env = json.load(f)
    if env.get("kind") != "edgelens-envelope":
        raise ValueError(f"{path} is not an EdgeLens envelope (edgelens envelope fit)")
    if spec is not None and env["features"] != feature_names(spec):
        raise ValueError(f"{path} was fitted for other features than this spec "
                         f"(columns, stats or layout differ): refit it")
    return env


class Guard:
    def __init__(self, env, spec):
        self.lo = np.asarray(env["lo"])
        self.hi = np.asarray(env["hi"])
        self.names = env["features"]
        self.invalid_frac = spec["envelope"]["invalid_pct"] / 100.0
        self.width = len(self.lo)

    def out_mask(self, vec):
        v = np.asarray(vec, dtype=np.float64)
        return (v < self.lo) | (v > self.hi) | np.isnan(v)

    def state(self, vec, warming_up=False, gap=False):
        """(state, outside indices)."""
        if warming_up:
            return "warming_up", []
        if gap:
            return "gap", []
        m = self.out_mask(vec)
        n = int(m.sum())
        if n == 0:
            return "valid", []
        idx = np.nonzero(m)[0].tolist()
        if np.isnan(np.asarray(vec, dtype=np.float64)).any() or n > self.invalid_frac * self.width:
            return "invalid", idx
        return "degraded", idx


def check(recordings, spec, env):
    """Per recording: share of steps in each state, and the inputs that leave
    the envelope most often (warm-up excluded)."""
    g = Guard(env, spec)
    seq = spec["features"]["sequence"]
    rows, counts_all = [], np.zeros(g.width, dtype=np.int64)
    for r in recordings:
        F, info = batch_features(r, spec)
        states = {s: 0 for s in STATES}
        counts = np.zeros(g.width, dtype=np.int64)
        for k, vec in enumerate(F):
            st, idx = g.state(vec, warming_up=(spec["features"]["padding"] == "zeros"
                                               and k < seq - 1), gap=bool(info["gap"][k]))
            states[st] += 1
            if idx:
                counts[idx] += 1
        counts_all += counts
        judged = len(F) - states["warming_up"]
        rows.append({"recording": r.source, "steps": len(F), "states": states,
                     "in_envelope_pct": round(100.0 * states["valid"] / judged, 2)
                     if judged else None,
                     "top_outside": _top(counts, g.names, judged)})
    return {"recordings": rows, "top_outside": _top(counts_all, g.names,
                                                    sum(r["steps"] for r in rows))}


def _top(counts, names, n, k=8):
    order = np.argsort(-counts)[:k]
    return [{"feature": names[i], "steps": int(counts[i]),
             "pct": round(100.0 * counts[i] / n, 2) if n else None}
            for i in order if counts[i] > 0]


def by_column(top, spec):
    """Group "col_stat" names back to columns: which SENSOR leaves the envelope."""
    out = {}
    for t in top:
        col = t["feature"].rsplit("_", 1)[0]
        out[col] = max(out.get(col, 0), t["steps"])
    return out

"""
edgelens.stream.data
----------------------
Recordings (CSV, one row per merged sample) and what is wrong with them.

No pandas: a 40 000-row, 45-column recording loads in about a second with the
csv module and NumPy, on a Jetson Nano too.
"""

import csv
import datetime
import math
import os

import numpy as np

_EPOCH = datetime.datetime(1970, 1, 1)


class Recording:
    """time_s [N] (seconds, float64), x [N, C] raw inputs, truth [N] or None."""

    def __init__(self, time_s, x, columns, truth=None, source=None):
        self.time_s = np.asarray(time_s, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.columns = list(columns)
        self.truth = None if truth is None else np.asarray(truth, dtype=np.float64)
        self.source = source

    def __len__(self):
        return len(self.time_s)

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.source or "recording"))[0]

    def copy(self):
        return Recording(self.time_s.copy(), self.x.copy(), self.columns,
                         None if self.truth is None else self.truth.copy(), self.source)


def _time_parser():
    cache = {}

    def parse(text):
        try:
            return float(text)
        except ValueError:
            pass
        t = text.strip().replace("T", " ").rstrip("Z")
        main, _, frac = t.partition(".")
        base = cache.get(main)
        if base is None:
            try:
                dt = datetime.datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"cannot read time '{text}': use seconds or "
                                 f"'YYYY-MM-DD HH:MM:SS[.fraction]'") from None
            base = (dt - _EPOCH).total_seconds()     # timezone-free: only differences matter
            cache[main] = base
        return base + (float("0." + frac) if frac else 0.0)
    return parse


def load_csv(path, spec):
    """Read the spec's time, input and target columns from one CSV file."""
    d = spec["data"]
    tcol = d["time_column"]
    target = d.get("target")
    tgt_cols = target["columns"] if target else []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        index = {h.strip(): i for i, h in enumerate(header)}
        missing = [c for c in [tcol] + d["columns"] + tgt_cols if c not in index]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing[:5])}"
                             + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))
        ti = index[tcol]
        xi = [index[c] for c in d["columns"]]
        yi = [index[c] for c in tgt_cols]
        parse_t = _time_parser()
        times, xs, ys = [], [], []
        for row in reader:
            if not row:
                continue
            times.append(parse_t(row[ti]))
            xs.append([_f(row[i]) for i in xi])
            if yi:
                ys.append([_f(row[i]) for i in yi])
    if not times:
        raise ValueError(f"{path} has a header but no rows")
    truth = None
    if target:
        y = np.asarray(ys, dtype=np.float64)
        red = {"min": np.min, "max": np.max, "mean": np.mean,
               "first": lambda a, axis: a[:, 0]}[target["reduce"]]
        truth = red(y, axis=1) * target["scale"]
    return Recording(times, xs, d["columns"], truth, source=str(path))


def _f(s):
    try:
        return float(s)
    except ValueError:
        return math.nan


def audit(rec, spec, train_constant=None):
    """What a model cannot know about its input: rate, gaps, dead and broken
    columns. `train_constant` = columns constant in the training data."""
    d = spec["data"]
    dt = np.diff(rec.time_s)
    expected = 1.0 / d["rate_hz"]
    med = float(np.median(dt)) if len(dt) else None
    gaps = []
    if len(dt):
        for k in np.nonzero(dt > d["max_gap_s"])[0]:
            gaps.append({"at_s": round(float(rec.time_s[k] - rec.time_s[0]), 2),
                         "length_s": round(float(dt[k]), 3)})
    backwards = int((dt < 0).sum()) if len(dt) else 0
    nan_cols = {c: int(n) for c, n in zip(rec.columns, np.isnan(rec.x).sum(axis=0)) if n}
    const = [c for c, col in zip(rec.columns, rec.x.T) if np.nanmax(col) == np.nanmin(col)]
    out = {"rows": len(rec), "duration_s": round(float(rec.time_s[-1] - rec.time_s[0]), 3),
           "rate_hz_expected": d["rate_hz"],
           "rate_hz_measured": round(1.0 / med, 3) if med else None,
           "rate_mismatch": bool(med and abs(med - expected) > 0.1 * expected),
           "gaps": gaps, "gap_time_s": round(sum(g["length_s"] for g in gaps), 3),
           "time_backwards": backwards, "nan_values": nan_cols, "constant_columns": const}
    if train_constant is not None:
        out["dead_columns"] = list(train_constant)
    return out


def find_constant_columns(recordings):
    """Columns that never change across all the given recordings."""
    cols = recordings[0].columns
    lo = np.nanmin(np.vstack([r.x.min(axis=0) for r in recordings]), axis=0)
    hi = np.nanmax(np.vstack([r.x.max(axis=0) for r in recordings]), axis=0)
    return [c for c, a, b in zip(cols, lo, hi) if a == b]


def fit_scaler(recordings, kind="robust"):
    """Fit a per-column scaler on raw training rows (the same numbers as
    scikit-learn's RobustScaler / StandardScaler with default settings)."""
    x = np.vstack([r.x for r in recordings])
    if kind == "robust":
        center = np.nanmedian(x, axis=0)
        q1, q3 = np.nanpercentile(x, [25, 75], axis=0)
        scale = q3 - q1
    elif kind == "standard":
        center = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
    elif kind == "minmax":
        lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
        # x' = (x - lo) / (hi - lo): the same affine form
        center, scale = lo, hi - lo
    else:
        raise ValueError(f"unknown scaler kind '{kind}' (robust, standard, minmax)")
    scale = np.where(scale == 0, 1.0, scale)     # sklearn: constant column -> scale 1
    return {"kind": kind, "columns": recordings[0].columns, "center": center.tolist(),
            "scale": scale.tolist(), "rows": int(x.shape[0]),
            "fitted_on": [os.path.basename(r.source or "") for r in recordings]}

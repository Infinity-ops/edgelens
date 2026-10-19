"""
edgelens.stream.features
--------------------------
Block statistics -> model windows, two ways that must give the same numbers:

* batch_features(): the whole recording at once — the reference, written to
  match what training code does with pandas (`rolling(W).agg(...)` sampled
  every W rows, `std` with ddof=1, windows zero-padded at the start).
* StreamFeaturizer: what a deployed system does — rows arrive, one model
  step is formed per block. mode="incremental" computes only the NEW block
  and keeps the last `sequence` block vectors in a ring; mode="recompute"
  recomputes every block of the window from the raw history each step (what
  a straightforward port of the training code does). Same output, very
  different cost: `edgelens replay --features recompute` measures both.
"""

import numpy as np

_STAT = {
    "min": lambda b: b.min(axis=0),
    "max": lambda b: b.max(axis=0),
    "mean": lambda b: b.mean(axis=0),
    "std": lambda b: b.std(axis=0, ddof=1),
    "last": lambda b: b[-1],
    "first": lambda b: b[0],
    "median": lambda b: np.median(b, axis=0),
    "range": lambda b: b.max(axis=0) - b.min(axis=0),
}


def scale(x, spec):
    sc = spec["scaler"]
    return (x - np.asarray(sc["center"])) / np.asarray(sc["scale"])


def block_vector(block, spec):
    """One model step from one block of SCALED rows [W, C]."""
    parts = [_STAT[s](block) for s in spec["features"]["stats"]]
    if spec["features"]["layout"] == "column-major":
        return np.stack(parts, axis=1).reshape(-1)
    return np.concatenate(parts)


def _blocks_stat(b3, s):
    """Vectorized stat over axis 1 of [nblocks, W, C]."""
    if s == "std":
        return b3.std(axis=1, ddof=1)
    if s == "last":
        return b3[:, -1]
    if s == "first":
        return b3[:, 0]
    if s == "range":
        return b3.max(axis=1) - b3.min(axis=1)
    return getattr(np, s)(b3, axis=1)


def feature_names(spec):
    cols, stats = spec["data"]["columns"], spec["features"]["stats"]
    if spec["features"]["layout"] == "column-major":
        return [f"{c}_{s}" for c in cols for s in stats]
    return [f"{c}_{s}" for s in stats for c in cols]


def batch_features(rec, spec):
    """(F [nblocks, width] float64, info) for a whole recording. Block k covers
    rows k*W .. k*W+W-1; rows after the last full block are dropped (as
    pandas' rolling + iloc[W-1::W] does)."""
    W = spec["features"]["block"]
    n = (len(rec) // W) * W
    xs = scale(rec.x[:n], spec)
    b3 = xs.reshape(-1, W, xs.shape[1])
    parts = [_blocks_stat(b3, s) for s in spec["features"]["stats"]]
    if spec["features"]["layout"] == "column-major":
        F = np.stack(parts, axis=2).reshape(b3.shape[0], -1)
    else:
        F = np.concatenate(parts, axis=1)
    t = rec.time_s[:n].reshape(-1, W)
    gap_in_block = np.zeros(len(F), dtype=bool)
    if W > 1:
        gap_in_block = (np.diff(t, axis=1) > spec["data"]["max_gap_s"]).any(axis=1)
    # a gap between the previous block's end and this block's start counts too
    if len(F) > 1:
        gap_in_block[1:] |= (t[1:, 0] - t[:-1, -1]) > spec["data"]["max_gap_s"]
    info = {"t_end": t[:, -1], "gap": gap_in_block,
            "truth": (rec.truth[:n].reshape(-1, W)[:, -1] if rec.truth is not None else None)}
    return F, info


def windows(F, spec):
    """Model inputs [nwin, sequence, width] float32 (zero-padded like training
    when padding == "zeros"; otherwise only full windows)."""
    seq = spec["features"]["sequence"]
    F = F.astype(np.float32)
    if spec["features"]["padding"] == "zeros":
        F = np.vstack([np.zeros((seq - 1, F.shape[1]), np.float32), F])
    if len(F) < seq:
        return np.zeros((0, seq, F.shape[1]), np.float32)
    idx = np.arange(seq)[None, :] + np.arange(len(F) - seq + 1)[:, None]
    return F[idx]


class StreamFeaturizer:
    """Rows in, model windows out, one per block."""

    def __init__(self, spec, mode="incremental"):
        if mode not in ("incremental", "recompute"):
            raise ValueError("mode: incremental or recompute")
        self.spec = spec
        self.mode = mode
        f = spec["features"]
        self.W, self.seq, self.width = f["block"], f["sequence"], f["width"]
        self.center = np.asarray(spec["scaler"]["center"])
        self.scale_ = np.asarray(spec["scaler"]["scale"])
        self.steps = 0
        pad = self.seq - 1 if f["padding"] == "zeros" else 0
        self.pad = pad
        self._ring = np.zeros((self.seq + pad, self.width), np.float32)  # incremental
        self._n = pad              # rows of the ring in use (padding counts as rows)
        self._raw = []             # recompute: last seq*W raw rows

    @property
    def warming_up(self):
        """True while the window still contains padding."""
        return self.steps < self.seq

    def push(self, block_raw):
        """One block of RAW rows [W, C] -> window [1, seq, width] float32, or None
        when padding is "none" and the window is not full yet."""
        self.steps += 1
        if self.mode == "recompute":
            self._raw.append(np.asarray(block_raw, dtype=np.float64))
            if len(self._raw) > self.seq:
                self._raw.pop(0)
            xs = (np.vstack(self._raw) - self.center) / self.scale_
            b3 = xs.reshape(-1, self.W, xs.shape[1])
            vecs = np.stack([block_vector(b, self.spec) for b in b3]).astype(np.float32)
            if len(vecs) < self.seq:
                if not self.pad:
                    return None
                vecs = np.vstack([np.zeros((self.seq - len(vecs), self.width), np.float32),
                                  vecs])
            return vecs[None]
        v = block_vector((np.asarray(block_raw, dtype=np.float64) - self.center) / self.scale_,
                         self.spec)
        if self._n < len(self._ring):
            self._ring[self._n] = v
            self._n += 1
        else:
            self._ring[:-1] = self._ring[1:]
            self._ring[-1] = v
        if self._n < self.seq:
            return None
        # a copy: the ring moves on with the next block
        return self._ring[self._n - self.seq:self._n][None].copy()


class OutputFilter:
    """The post-processing a deployed model gets: slew limit per step, then a
    rolling mean (min_periods=1), applied causally one value at a time."""

    def __init__(self, spec):
        o = spec["output"]
        self.slew = o["slew_per_step"]
        self.n = o["mean_steps"]
        self.prev = None
        self.buf = []

    def __call__(self, v):
        v = float(v)
        if self.slew is not None and self.prev is not None:
            d = v - self.prev
            if abs(d) > self.slew:
                v = self.prev + (self.slew if d > 0 else -self.slew)
        self.prev = v
        self.buf.append(v)
        if len(self.buf) > self.n:
            self.buf.pop(0)
        return sum(self.buf) / len(self.buf)


def filter_series(values, spec):
    f = OutputFilter(spec)
    return np.array([f(v) for v in values])

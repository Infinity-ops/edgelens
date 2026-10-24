"""EdgeLens quick start: a sensor pipeline that needs only NumPy.

A vibration sensor at 4 kHz, a new 256-sample window every 32 ms, a small
linear "model". Deterministic (seeded), so every machine runs the same work.

    edgelens benchmark --pipeline quickstart.py --iterations 1000 --save run.json
"""
import numpy as np

import edgelens as el

RATE_HZ, WINDOW, HOP = 4000, 256, 128          # new window every 32 ms


def build_pipeline():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((2, RATE_HZ * 5)).astype(np.float32)
    weights = rng.standard_normal((2 * 32, 3)).astype(np.float32)
    hann = np.hanning(WINDOW).astype(np.float32)

    pipe = el.Pipeline("quickstart", pack="timeseries",
                       sample_rate_hz=RATE_HZ, window=WINDOW, hop=HOP)
    pipe.set_source(el.WindowSource(signal, window=WINDOW, hop=HOP))

    @pipe.stage
    def acquire(window):
        return np.ascontiguousarray(window, dtype=np.float32)

    @pipe.stage
    def fft(x):
        return np.abs(np.fft.rfft((x - x.mean(axis=-1, keepdims=True)) * hann, axis=-1))

    @pipe.stage
    def features(spec):
        return np.log1p(np.stack([b.mean(axis=-1) for b in np.array_split(spec, 32, -1)], -1))

    @pipe.stage(role="inference")
    def model(f):
        return int(np.argmax(f.reshape(-1) @ weights))

    @pipe.stage
    def decision(label):
        return label == 2

    return pipe

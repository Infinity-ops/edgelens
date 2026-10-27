"""
Predictive-maintenance pipeline: a window of 16 sensor channels x 50
samples -> features -> a small recurrent anomaly model -> alarm decision.

Numpy only, so this example repository runs on any Jetson (and any laptop).
Replace `infer` with your real model — onnxruntime / TensorRT / EDDL — and
`acquire` with your CAN / sensor input; nothing else in the CI changes.
"""
import numpy as np

import edgelens as el

CHANNELS, STEPS, HIDDEN = 16, 50, 64


def build_pipeline():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((4096, STEPS, CHANNELS)).astype(np.float32)
    w_in = rng.standard_normal((CHANNELS, HIDDEN)).astype(np.float32) * 0.2
    w_h = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32) * 0.1
    w_out = rng.standard_normal((HIDDEN, 1)).astype(np.float32) * 0.1
    mean, std = data.mean(axis=(0, 1)), data.std(axis=(0, 1)) + 1e-6

    pipe = el.Pipeline("anomaly-1hz", pack="timeseries")

    def windows():
        i = 0
        while True:
            yield data[i % len(data)]
            i += 1
    pipe.set_source(windows())

    @pipe.stage(role="input")
    def acquire(window):                      # CAN / sensor frames -> one window
        return np.ascontiguousarray(window)

    @pipe.stage(role="preprocess")
    def preprocess(x):                        # standardize per channel
        return (x - mean) / std

    @pipe.stage(role="inference")
    def infer(x):                             # stand-in for the LSTM: a tanh RNN
        h = np.zeros(HIDDEN, np.float32)
        for t in range(STEPS):
            h = np.tanh(x[t] @ w_in + h @ w_h)
        return float(1.0 / (1.0 + np.exp(-(h @ w_out)[0])))

    @pipe.stage(role="postprocess")
    def decide(score):                        # maintenance alarm
        return score > 0.9

    return pipe

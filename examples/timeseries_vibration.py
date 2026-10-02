"""
Predictive-maintenance style sensor pipeline, profiled with the timeseries pack.

    edgelens benchmark --pipeline examples/timeseries_vibration.py --iterations 2000
    edgelens diagnose

Replace the synthetic signal with your recording (np.load("vib.npy"), shape
[channels, T]) and `classify` with your real model (onnxruntime, TensorRT,
PyTorch...). Everything else stays the same.
"""
import numpy as np

import edgelens as el

SAMPLE_RATE_HZ = 10_000      # accelerometer rate
WINDOW = 1024                # samples per analysis window (102.4 ms)
HOP = 512                    # new window every 51.2 ms -> the real-time deadline


def build_pipeline():
    signal = el.WindowSource.synthetic(SAMPLE_RATE_HZ, seconds=10, channels=3)
    pipe = el.Pipeline("bearing-monitor", pack="timeseries",
                       sample_rate_hz=SAMPLE_RATE_HZ, window=WINDOW, hop=HOP)
    pipe.set_source(el.WindowSource(signal, window=WINDOW, hop=HOP))

    hann = np.hanning(WINDOW).astype(np.float32)
    weights = np.random.default_rng(0).standard_normal((3 * 64, 4)).astype(np.float32)

    @pipe.stage
    def acquire(window):                       # copy out of the driver buffer
        return np.ascontiguousarray(window, dtype=np.float32)

    @pipe.stage
    def filter(x):                             # remove DC offset per channel
        return x - x.mean(axis=-1, keepdims=True)

    @pipe.stage
    def fft(x):                                # magnitude spectrum per channel
        return np.abs(np.fft.rfft(x * hann, axis=-1))

    @pipe.stage
    def feature_extraction(spec):              # 64 log band energies per channel
        bands = np.array_split(spec, 64, axis=-1)
        return np.log1p(np.stack([b.mean(axis=-1) for b in bands], axis=-1)).reshape(-1)

    @pipe.stage(role="inference")
    def classify(features):                    # stand-in for your model
        logits = features @ weights
        return int(np.argmax(logits))

    @pipe.stage
    def decision(label):                       # raise a maintenance flag
        return label == 3

    return pipe

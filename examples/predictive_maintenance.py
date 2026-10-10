"""
Predictive-maintenance pipeline for EdgeLens (Demo 2 style).

    edgelens benchmark --pipeline predictive_maintenance.py --iterations 2000 --save pm.json
    edgelens diagnose pm.json
    edgelens validate pm.json --p99-ms 20 --max-miss-ratio 0.001

To use YOUR pipeline: replace the signal with your recording and the model
stage with your real model. Keep the stage names, or rename them freely.
"""
import numpy as np

import edgelens as el

# --- your timing parameters ---------------------------------------------
SAMPLE_RATE_HZ = 10_000    # accelerometer sampling rate
WINDOW = 1024              # samples per analysis window  (102.4 ms of signal)
HOP = 512                  # a new window every 512 samples (= every 51.2 ms)
CHANNELS = 3               # x, y, z axes


def build_pipeline():
    # 1. The data: a real recording (shape [channels, samples]) ...
    #      signal = np.load("vibration_recording.npy")
    # ... or a synthetic test signal until you have one:
    signal = el.WindowSource.synthetic(SAMPLE_RATE_HZ, seconds=60, channels=CHANNELS)

    # 2. The pipeline. The timeseries pack sets the deadline automatically:
    #    each window must be finished before the next one arrives (51.2 ms).
    pipe = el.Pipeline("bearing-monitor", pack="timeseries",
                       sample_rate_hz=SAMPLE_RATE_HZ, window=WINDOW, hop=HOP)
    pipe.set_source(el.WindowSource(signal, window=WINDOW, hop=HOP))

    hann = np.hanning(WINDOW).astype(np.float32)

    # Stand-in model: replace with your real one, e.g. onnxruntime:
    #   import onnxruntime as ort
    #   sess = ort.InferenceSession("bearing_model.onnx",
    #                               providers=["TensorrtExecutionProvider", "CPUExecutionProvider"])
    #   ...then in classify():  return sess.run(None, {"features": features[None, :]})[0]
    weights = np.random.default_rng(0).standard_normal((CHANNELS * 64, 3)).astype(np.float32)
    labels = ("healthy", "early wear", "damaged")

    # 3. The steps, in order. Each one receives the previous step's output.
    @pipe.stage
    def acquire(window):                          # copy the window out of the driver buffer
        return np.ascontiguousarray(window, dtype=np.float32)

    @pipe.stage
    def filter(x):                                # remove the DC offset per axis
        return x - x.mean(axis=-1, keepdims=True)

    @pipe.stage
    def fft(x):                                   # frequency spectrum per axis
        return np.abs(np.fft.rfft(x * hann, axis=-1))

    @pipe.stage
    def feature_extraction(spectrum):             # 64 band energies per axis
        bands = np.array_split(spectrum, 64, axis=-1)
        return np.log1p(np.stack([b.mean(axis=-1) for b in bands], axis=-1)).reshape(-1)

    @pipe.stage(role="inference")
    def classify(features):                       # the AI model
        return labels[int(np.argmax(features @ weights))]

    @pipe.stage
    def decision(state):                          # act on the result
        return state != "healthy"                 # True = schedule maintenance

    return pipe

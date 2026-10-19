"""
DEM2-style predictive maintenance, replayed from a ROS bag.

    python examples/rosbag_to_windows.py dem2.bag --out dem2_merged.npy
    edgelens benchmark --pipeline examples/dem2_bag_replay.py --iterations 300 \\
        --deadline-ms 3000 --save dem2.json
    edgelens validate dem2.json --p99-ms 3000 --max-miss-ratio 0

Real-time check at the true 1 Hz rate (slow on purpose: 1 iteration/s):
    edgelens benchmark --pipeline examples/dem2_bag_replay.py --iterations 120 \\
        --pace --period-ms 1000 --deadline-ms 3000 --save dem2_1hz.json

Shape of the problem (SML's data description): the model receives the last
X merged readings of /imu, /odom and /cmd_vel and predicts the robot's
state. Here X = 50 rows of the 100 Hz IMU timeline (0.5 s of history) and a
new prediction every 100 rows (1 Hz).

The model is a STAND-IN LSTM in NumPy (a bag-replay timing demo). For a real
trained windowed model, describe its preprocessing in a stream spec and use
`edgelens replay` / `edgelens detect` instead (docs/stream.md): that replays
the exact training features and measures decisions against ground truth.

Environment variables: DEM2_MERGED (default dem2_merged.npy), DEM2_WINDOW
(rows, default 50), DEM2_HOP (rows, default 100), DEM2_RATE_HZ (base topic
rate, default 100), DEM2_MAX_AGE_MS (freshness limit, default 200).
"""

import json
import os
from pathlib import Path

import numpy as np

import edgelens as el

MERGED = os.environ.get("DEM2_MERGED", "dem2_merged.npy")
WINDOW = int(os.environ.get("DEM2_WINDOW", 50))
HOP = int(os.environ.get("DEM2_HOP", 100))
RATE_HZ = float(os.environ.get("DEM2_RATE_HZ", 100))
MAX_AGE_MS = float(os.environ.get("DEM2_MAX_AGE_MS", 200))
HIDDEN = 32


def _columns(npy_path):
    meta = Path(npy_path).with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text())["columns"]
    return None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_pipeline():
    data = np.load(MERGED)                                  # [channels, rows]
    cols = _columns(MERGED) or [f"c{i}" for i in range(data.shape[0])]
    feat_idx = [i for i, c in enumerate(cols) if c != "time_s" and not c.startswith("age_ms:")]
    age_idx = [i for i, c in enumerate(cols) if c.startswith("age_ms:")]
    n_feat = len(feat_idx)

    rng = np.random.default_rng(0)
    w = rng.standard_normal((4 * HIDDEN, n_feat + HIDDEN)).astype(np.float32) * 0.1
    b = np.zeros(4 * HIDDEN, np.float32)
    w_out = rng.standard_normal(HIDDEN).astype(np.float32)
    mean = data[feat_idx].mean(axis=1, keepdims=True)
    std = data[feat_idx].std(axis=1, keepdims=True) + 1e-6

    pipe = el.Pipeline("dem2-bag-replay", pack="timeseries", sample_rate_hz=RATE_HZ,
                       window=WINDOW, hop=HOP)
    pipe.set_source(el.WindowSource(data, window=WINDOW, hop=HOP))

    @pipe.stage(role="input")
    def build_window(window):                               # [channels, X] -> features
        return {"x": np.ascontiguousarray(window[feat_idx], dtype=np.float32),
                "age_ms": window[age_idx].max(axis=1) if age_idx else np.zeros(0)}

    @pipe.stage(role="preprocess")
    def normalize(item):
        item["x"] = ((item["x"] - mean) / std).astype(np.float32).T      # [X, features]
        return item

    @pipe.stage(role="inference")
    def lstm(item):                                         # stand-in model
        h = np.zeros(HIDDEN, np.float32)
        c = np.zeros(HIDDEN, np.float32)
        for x_t in item["x"]:
            z = w @ np.concatenate([x_t, h]) + b
            i, f, g, o = np.split(z, 4)
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
        item["score"] = float(_sigmoid(w_out @ h))
        return item

    @pipe.stage(role="decision")
    def decide(item):
        # A prediction from stale inputs is not a prediction about "now".
        stale = bool(item["age_ms"].size and item["age_ms"].max() > MAX_AGE_MS)
        return {"anomaly": item["score"] > 0.5 and not stale, "stale_input": stale}

    return pipe

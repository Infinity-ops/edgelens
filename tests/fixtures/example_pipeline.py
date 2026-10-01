"""
Minimal working example for `edgelens benchmark --pipeline example_pipeline.py`.

Run it with:
    edgelens benchmark --pipeline tests/fixtures/example_pipeline.py

This one uses plain NumPy so it has no extra dependencies — a real
pipeline would replace capture() with an actual camera/video read,
h2d_copy()/inference()/d2h_copy() with your actual runtime (TensorRT,
PyTorch, etc.), matching whatever your deployment actually does.
"""

import numpy as np


def build_stage_fns():
    # One-time setup — called once before timing starts. Load your real
    # model, open your real camera, allocate your real buffers here.
    frame = np.random.rand(1, 3, 64, 64).astype(np.float32)
    state = {"frame": frame, "prepped": None, "output": None}

    def capture():
        # Replace with a real camera/video frame grab.
        state["frame"] = state["frame"]  # no-op placeholder

    def preprocess():
        f = state["frame"]
        state["prepped"] = (f - f.min()) / (np.ptp(f) + 1e-8)  # np.ptp(), not
        # f.ptp() — removed from ndarray in NumPy 2.x, function form still works

    def h2d_copy():
        # No real device in this example — replace with your actual
        # host-to-device transfer (e.g. torch .to('cuda'), a CUDA memcpy).
        pass

    def inference():
        # Replace with your real model call. This example just does a
        # trivial reduction so there's something to time.
        state["output"] = state["prepped"].sum()

    def d2h_copy():
        # No real device in this example — replace with your actual
        # device-to-host transfer.
        pass

    def postprocess():
        _ = float(state["output"])

    return {
        "capture": capture,
        "preprocess": preprocess,
        "h2d_copy": h2d_copy,
        "inference": inference,
        "d2h_copy": d2h_copy,
        "postprocess": postprocess,
    }
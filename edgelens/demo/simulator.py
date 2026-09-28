"""
edgelens.demo.simulator
-------------------------
Generates realistic synthetic pipeline-stage timings and telemetry so
the full CLI (benchmark -> diagnose -> report) can be exercised and
demoed on any machine, not just a Jetson. Used automatically when
edgelens detects it is NOT running on Jetson hardware, or explicitly
via `--demo`.

This is clearly labeled "demo" mode everywhere in the CLI/report output
— it must never be mistaken for a real hardware measurement.
"""

import random

SCENARIOS = ["balanced", "preprocess", "memory", "gpu", "thermal"]


def simulate_stage_timings(bottleneck="balanced"):
    base = {
        "capture": random.uniform(2.0, 4.0),
        "preprocess": random.uniform(4.0, 6.0),
        "h2d_copy": random.uniform(2.0, 4.0),
        "inference": random.uniform(10.0, 14.0),
        "d2h_copy": random.uniform(1.0, 2.0),
        "postprocess": random.uniform(2.0, 4.0),
    }
    if bottleneck == "preprocess":
        base["preprocess"] *= 2.8
    elif bottleneck == "memory":
        base["h2d_copy"] *= 3.0
        base["d2h_copy"] *= 2.6
    elif bottleneck == "gpu":
        base["inference"] *= 2.3
    elif bottleneck == "thermal":
        base["inference"] *= 1.8
        base["preprocess"] *= 1.3
    # jitter
    return {k: max(0.1, v * random.uniform(0.92, 1.08)) for k, v in base.items()}


def simulate_telemetry(bottleneck="balanced"):
    temp, gpu, cpu, mem = 52.0, 58.0, 40.0, 42.0
    if bottleneck == "thermal":
        temp, gpu, cpu = 84.0, 52.0, 55.0
    elif bottleneck == "gpu":
        gpu, cpu = 97.0, 28.0
    elif bottleneck == "preprocess":
        cpu, gpu = 93.0, 38.0
    elif bottleneck == "memory":
        cpu, gpu, mem = 58.0, 50.0, 88.0
    return {
        "cpu_percent": cpu,
        "gpu_percent": gpu,
        "mem_percent": mem,
        "mem_used_gb": round(mem / 100 * 8, 2),
        "mem_total_gb": 8.0,
        "swap_used_gb": 0.0,
        "temps_c": {"CPU-therm": temp, "GPU-therm": temp - 3},
    }

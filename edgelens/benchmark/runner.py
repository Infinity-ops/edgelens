"""
edgelens.benchmark.runner
---------------------------
Stage-level latency benchmark harness. This is the core of EdgeLens'
differentiation: measuring WHERE time goes in an inference pipeline,
not just an aggregate FPS number.

v0.1 uses wall-clock timing (time.perf_counter) around each stage.
CUDA-event-level timing (cudaEvent, for accurate GPU-side measurement
independent of CPU/host scheduling noise) is the documented v0.2
upgrade — see ROADMAP.md — and the stage interface below is designed
so that swapping in cudaEvent-based timers later doesn't change the
CLI or output schema.

Wiring in your real model:
  Replace `_default_pipeline_stage_fn` below with your own per-stage
  callables (capture/preprocess/h2d_copy/inference/d2h_copy/postprocess)
  when integrating your model. See README "Wiring your pipeline".
"""

import statistics
import time

from ..demo import simulator
from ..hardware import detector, telemetry

STAGES = ["capture", "preprocess", "h2d_copy", "inference", "d2h_copy", "postprocess"]


def run_benchmark(iterations=50, warmup=10, demo=False, demo_scenario="balanced",
                   stage_fns=None):
    """
    Run a benchmark and return a structured result dict.

    Args:
        iterations: number of timed iterations.
        warmup: untimed warmup iterations (real hardware mode only).
        demo: force synthetic/demo data even on Jetson.
        demo_scenario: one of edgelens.demo.simulator.SCENARIOS.
        stage_fns: optional dict[str, callable] of user-supplied stage
            functions for real pipelines. Each callable takes no args
            and does the work for that stage (timing is handled here).
            If omitted on real hardware, a placeholder pipeline is
            timed instead (clearly marked in the result).
    """
    is_jetson = detector.is_jetson()
    use_demo = demo or not is_jetson

    stage_samples = {s: [] for s in STAGES}

    if use_demo:
        for _ in range(iterations):
            timings = simulator.simulate_stage_timings(bottleneck=demo_scenario)
            for s in STAGES:
                stage_samples[s].append(timings[s])
        tel = simulator.simulate_telemetry(bottleneck=demo_scenario)
        mode = "demo"
        pipeline_source = "simulated"
    else:
        fns = stage_fns or _placeholder_stage_fns()
        pipeline_source = "user-supplied" if stage_fns else "placeholder (no model wired in)"
        for _ in range(warmup):
            for s in STAGES:
                fns[s]()
        for _ in range(iterations):
            for s in STAGES:
                t0 = time.perf_counter()
                fns[s]()
                stage_samples[s].append((time.perf_counter() - t0) * 1000.0)
        tel = telemetry.snapshot()
        mode = "hardware"

    result = _summarize(stage_samples, tel, mode, pipeline_source)
    return result


def _placeholder_stage_fns():
    """No-op stage functions used only when no real model is wired in yet,
    so the CLI still produces valid (if meaningless) output on real
    hardware rather than crashing. Real usage should pass stage_fns."""
    def make(delay):
        def _fn():
            time.sleep(delay)
        return _fn
    return {
        "capture": make(0.001),
        "preprocess": make(0.002),
        "h2d_copy": make(0.001),
        "inference": make(0.004),
        "d2h_copy": make(0.0005),
        "postprocess": make(0.001),
    }


def _summarize(stage_samples, telemetry_snapshot, mode, pipeline_source):
    stage_avg = {s: round(statistics.mean(v), 3) for s, v in stage_samples.items()}
    per_iter_totals = [sum(vals) for vals in zip(*stage_samples.values())]
    per_iter_totals.sort()
    n = len(per_iter_totals)

    def pct(p):
        if n == 0:
            return 0.0
        idx = min(n - 1, int(p * n))
        return round(per_iter_totals[idx], 3)

    total_avg = round(statistics.mean(per_iter_totals), 3) if per_iter_totals else 0.0
    fps = round(1000.0 / total_avg, 2) if total_avg > 0 else 0.0

    return {
        "mode": mode,
        "pipeline_source": pipeline_source,
        "iterations": n,
        "stage_avg_ms": stage_avg,
        "total_latency_ms": total_avg,
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "latency_p99_ms": pct(0.99),
        "fps": fps,
        "telemetry": telemetry_snapshot,
    }

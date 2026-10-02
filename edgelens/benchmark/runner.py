"""
edgelens.benchmark.runner
---------------------------
Stage-level latency benchmark harness. This is the core of EdgeLens'
differentiation: measuring WHERE time goes in an inference pipeline,
not just an aggregate FPS number.

v0.1.0 CHANGE FROM THE ALPHA SCAFFOLD:
    Real hardware mode now REQUIRES either `model_path` (a real .onnx
    file, executed via edgelens.benchmark.onnx_pipeline), `pipeline_path`
    (a user script, via edgelens.benchmark.pipeline_loader), or an
    explicit `stage_fns` dict. There is no more silent placeholder
    pipeline — a benchmark you didn't wire a real model or real stage
    functions into now raises RuntimeError instead of quietly reporting
    timings for time.sleep() calls. A tool that measures nothing must
    say so loudly, not produce a number that looks like a measurement.

CUDA-event-level timing (cudaEvent, for accurate GPU-side measurement
independent of CPU/host scheduling noise) is the documented v0.2/v0.3
upgrade — see ROADMAP.md.
"""

import statistics
import time

from ..demo import simulator
from ..hardware import detector, telemetry

STAGES = ["capture", "preprocess", "h2d_copy", "inference", "d2h_copy", "postprocess"]


def run_benchmark(iterations=50, warmup=10, demo=False, demo_scenario="balanced",
                   model_path=None, provider=None, stage_fns=None,
                   pipeline_path=None, telemetry_interval_s=0.2, input_shapes=None):
    """
    Run a benchmark and return a structured result dict.

    Args:
        iterations: number of timed iterations.
        warmup: untimed warmup iterations (real hardware mode only).
        demo: force synthetic/demo data even on Jetson.
        demo_scenario: one of edgelens.demo.simulator.SCENARIOS.
        model_path: path to a real .onnx file. When given, a real
            ONNX Runtime pipeline (edgelens.benchmark.onnx_pipeline)
            is built and benchmarked — this is the normal way to use
            EdgeLens on real hardware with a plain ONNX forward pass.
        provider: optional ONNX Runtime execution provider override
            (e.g. "CUDAExecutionProvider"). Auto-selected if omitted.
        stage_fns: optional dict[str, callable] of user-supplied stage
            functions, for wiring in your own camera/preprocessing/model
            from Python directly. Highest priority if multiple pipeline
            sources are given.
        pipeline_path: path to a Python script defining a top-level
            `build_stage_fns()` function (see
            edgelens.benchmark.pipeline_loader for the contract). The
            CLI equivalent of passing stage_fns from Python — use this
            when you need a real camera, a non-ONNX runtime, or anything
            --model's plain ONNX forward pass can't express.
        telemetry_interval_s: sampling interval for the background
            telemetry recorder during real hardware runs.

    Raises:
        RuntimeError: if running in real-hardware mode (not demo) and
            none of `stage_fns`, `pipeline_path`, or `model_path` was
            supplied, or if more than one is given at once (ambiguous).
            EdgeLens v0.1.0 does not fabricate results from a
            placeholder pipeline.
    """
    is_jetson = detector.is_jetson()
    has_real_pipeline = (
        model_path is not None or stage_fns is not None or pipeline_path is not None
    )

    if demo:
        use_demo = True
    elif has_real_pipeline:
        # A real .onnx model, a --pipeline script, or user-supplied
        # stage_fns all work on ANY host (CPUExecutionProvider on a
        # laptop, CUDA/TensorRT on Jetson) — being off-Jetson is not
        # itself a reason to fall back to demo.
        use_demo = False
    else:
        # No real pipeline given and not explicitly asked to demo: keep
        # the previous UX of auto-demo on a non-Jetson host (a bare
        # `edgelens benchmark` still "just works" for a first look), but
        # on a real Jetson with nothing wired in, fall through to
        # _resolve_stage_fns()'s loud RuntimeError instead of guessing.
        use_demo = not is_jetson

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
        fns, pipeline_source = _resolve_stage_fns(model_path, provider, stage_fns, pipeline_path,
                                                  input_shapes)

        for _ in range(warmup):
            for s in STAGES:
                fns[s]()

        recorder = telemetry.TelemetryRecorder(interval_s=telemetry_interval_s)
        recorder.start()
        try:
            for _ in range(iterations):
                for s in STAGES:
                    t0 = time.perf_counter()
                    fns[s]()
                    stage_samples[s].append((time.perf_counter() - t0) * 1000.0)
        finally:
            tel = recorder.stop()
        mode = "hardware"

    result = _summarize(stage_samples, tel, mode, pipeline_source)
    return result


def _resolve_stage_fns(model_path, provider, stage_fns, pipeline_path, input_shapes=None):
    given = [name for name, val in
             (("stage_fns", stage_fns), ("pipeline_path", pipeline_path), ("model_path", model_path))
             if val is not None]
    if len(given) > 1:
        raise RuntimeError(
            f"Ambiguous pipeline source: got {', '.join(given)} together. "
            f"Pass exactly one of --model, --pipeline, or stage_fns=... "
            f"from Python."
        )

    if stage_fns is not None:
        return stage_fns, "user-supplied"

    if pipeline_path is not None:
        from .pipeline_loader import load_stage_fns_from_script
        fns = load_stage_fns_from_script(pipeline_path)
        return fns, f"custom pipeline ({pipeline_path})"

    if model_path is not None:
        from .onnx_pipeline import OnnxStagePipeline
        pipeline = OnnxStagePipeline(model_path, provider=provider, input_shapes=input_shapes)
        source = f"onnxruntime:{pipeline.provider} ({model_path})"
        return pipeline.stage_fns(), source

    raise RuntimeError(
        "edgelens benchmark needs something real to measure.\n"
        "Pass one of:\n"
        "  --model path/to/model.onnx       (runs a real ONNX Runtime session)\n"
        "  --pipeline path/to/script.py      (your own capture/model/pipeline)\n"
        "  --demo                            (synthetic data for previewing output)\n"
        "  stage_fns=... from Python         (wire in your own pipeline directly)\n"
        "EdgeLens does not fabricate hardware-mode results from a placeholder "
        "pipeline — see ROADMAP.md."
    )


def _summarize(stage_samples, telemetry_summary, mode, pipeline_source):
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
        "telemetry": telemetry_summary,
    }
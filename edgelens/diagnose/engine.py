"""
edgelens.diagnose.engine
---------------------------
Rule-based bottleneck diagnosis. Deliberately simple (thresholds +
correlation between stage-time share and system telemetry) rather than
a black-box model — a correct simple rule beats a fragile clever one,
and the reasoning is fully inspectable/explainable to the developer.

v0.1.0 CHANGE FROM THE ALPHA SCAFFOLD:
    The old output called its heuristic score "confidence" (e.g.
    "confidence 78%"), which reads as a statistically calibrated
    probability. It wasn't one — it was `0.4 + stage_pct / 100`, an
    ad-hoc scoring formula. Every finding now reports:
      - evidence_strength: the same 0-1 heuristic score, honestly named
      - evidence: the actual measured numbers that produced the verdict

v0.1.0 FIX (found via real-hardware testing, not simulated):
    psutil.cpu_percent(interval=X) BLOCKS for X seconds per call. The
    default telemetry sampling interval is 200ms. A fast benchmark (a
    tiny model, or few iterations) can finish its entire timed loop in
    single-digit milliseconds — faster than one sampling interval — so
    the background TelemetryRecorder gets exactly ONE sample. Reporting
    that one instantaneous reading as if it reflects sustained load
    during the benchmark is misleading, especially for THERMAL (a
    laptop/board's ambient temperature at that instant may have nothing
    to do with a benchmark that ran for 5ms). All telemetry-dependent
    findings now check sample_count and, below MIN_RELIABLE_SAMPLES,
    are demoted: evidence_strength is halved and a caveat is appended
    to `detail` explaining exactly why. Purely stage-timing-based
    findings (MEMORY_TRANSFER_BOUND, BALANCED's stage read) are
    unaffected — they don't depend on telemetry sample count.

Verdict types: CPU_BOUND_PREPROCESS, INFERENCE_ON_CPU, MEMORY_TRANSFER_BOUND,
GPU_BOUND, THERMAL, MEMORY_BOUND, BALANCED.
"""

# Tunable thresholds — deliberately named constants so they're easy to
# adjust as real-world Jetson data comes in.
THERMAL_LIMIT_C = 80.0
MEMORY_LIMIT_PCT = 85.0
PREPROCESS_STAGE_PCT_THRESHOLD = 30.0
PREPROCESS_CPU_THRESHOLD = 70.0
TRANSFER_STAGE_PCT_THRESHOLD = 20.0
INFERENCE_STAGE_PCT_THRESHOLD = 50.0
GPU_BUSY_THRESHOLD = 85.0
# A serial (single-threaded) stage that saturates ONE core shows up as only
# 100/N % system-wide CPU on an N-core board (25% on a 4-core Nano). Accept
# preprocessing as CPU-bound when the mean is at least this fraction of one
# core's share. Learned from real Jetson Nano data: 33% mean CPU while a
# NumPy preprocess stage took 33.6% of every frame.
SINGLE_CORE_FRACTION = 0.8

# Below this many background telemetry samples, CPU/GPU/temperature
# readings are one-off snapshots, not a load average over the benchmark —
# see the module docstring for exactly why this happens.
MIN_RELIABLE_SAMPLES = 3
LOW_SAMPLE_STRENGTH_PENALTY = 0.5  # multiplier applied when below the floor

TELEMETRY_DEPENDENT_TYPES = {"THERMAL", "MEMORY_BOUND", "CPU_BOUND_PREPROCESS", "GPU_BOUND"}
# INFERENCE_ON_CPU is deliberately NOT telemetry-dependent: it's decided
# from the provider recorded in pipeline_source, not from sampled load.


def diagnose(benchmark_result):
    stages = benchmark_result.get("stage_avg_ms", {})
    total = sum(stages.values()) or 1e-9
    stage_pct = {k: (v / total) * 100 for k, v in stages.items()}

    tel = benchmark_result.get("telemetry", {}) or {}
    sample_count = tel.get("sample_count", 0) or 0
    telemetry_reliable = sample_count >= MIN_RELIABLE_SAMPLES

    # Prefer mean-over-window fields (from TelemetryRecorder); fall back to
    # legacy flat fields for older/simulated data that only has one sample.
    cpu = tel.get("cpu_percent_mean", tel.get("cpu_percent")) or 0
    gpu = tel.get("gpu_percent_mean", tel.get("gpu_percent")) or 0
    mem_pct = tel.get("mem_percent_mean", tel.get("mem_percent")) or 0

    # Thermal uses PEAK, not mean — throttling is a transient event that a
    # mean across the whole run would dilute away.
    max_temp = tel.get("max_temp_c")
    if max_temp is None:
        temps = tel.get("temps_c") or {}
        max_temp = max(temps.values()) if temps else 0

    pre_pct = stage_pct.get("preprocess", 0)
    transfer_pct = stage_pct.get("h2d_copy", 0) + stage_pct.get("d2h_copy", 0)
    infer_pct = stage_pct.get("inference", 0)

    findings = []

    if max_temp >= THERMAL_LIMIT_C:
        findings.append({
            "type": "THERMAL",
            "evidence_strength": round(min(0.95, 0.5 + (max_temp - THERMAL_LIMIT_C) / 40), 2),
            "detail": f"Peak temperature {max_temp:.1f}C exceeds {THERMAL_LIMIT_C:.0f}C — "
                      f"thermal throttling likely reducing clocks.",
            "recommendation": "Improve cooling (heatsink/fan/airflow), drop to a lower "
                              "power mode, or reduce sustained workload duration.",
            "evidence": {
                "peak_temperature_c": round(max_temp, 1),
                "threshold_c": THERMAL_LIMIT_C,
            },
        })

    if mem_pct >= MEMORY_LIMIT_PCT:
        findings.append({
            "type": "MEMORY_BOUND",
            "evidence_strength": round(min(0.9, 0.5 + (mem_pct - MEMORY_LIMIT_PCT) / 30), 2),
            "detail": f"System memory usage at {mem_pct:.1f}% (mean over the run) — risk "
                      f"of swapping, which causes severe, hard-to-diagnose latency spikes.",
            "recommendation": "Reduce batch size, free unused buffers/caches, or move to "
                              "a smaller model / lower input resolution.",
            "evidence": {
                "mem_percent_mean": round(mem_pct, 1),
                "threshold_pct": MEMORY_LIMIT_PCT,
            },
        })

    cpu_count = tel.get("cpu_count")
    one_core_pct = (100.0 / cpu_count) if cpu_count else None
    all_cores_busy = cpu >= PREPROCESS_CPU_THRESHOLD
    single_core_busy = (one_core_pct is not None
                        and cpu >= SINGLE_CORE_FRACTION * one_core_pct)

    if pre_pct >= PREPROCESS_STAGE_PCT_THRESHOLD and (all_cores_busy or single_core_busy):
        if all_cores_busy:
            detail = (f"Preprocessing consumes {pre_pct:.1f}% of total pipeline latency "
                      f"while mean CPU utilization is {cpu:.1f}%.")
        else:
            detail = (f"Preprocessing consumes {pre_pct:.1f}% of total pipeline latency. "
                      f"Mean CPU is {cpu:.1f}% across {cpu_count} cores — about one "
                      f"core's worth of work ({one_core_pct:.0f}% each), consistent with "
                      f"single-threaded preprocessing while the other cores sit mostly idle.")
        findings.append({
            "type": "CPU_BOUND_PREPROCESS",
            "evidence_strength": round(min(0.95, 0.4 + pre_pct / 100), 2),
            "detail": detail,
            "recommendation": "Move resize/color-conversion to the GPU (CUDA/VPI) or use "
                              "hardware-accelerated decode instead of CPU-side OpenCV; "
                              "or overlap preprocessing of frame N+1 with inference of "
                              "frame N on another core.",
            "evidence": {
                "preprocess_stage_pct": round(pre_pct, 1),
                "cpu_percent_mean": round(cpu, 1),
                "cpu_count": cpu_count,
                "thresholds": {
                    "stage_pct": PREPROCESS_STAGE_PCT_THRESHOLD,
                    "cpu_pct": PREPROCESS_CPU_THRESHOLD,
                    "single_core_pct": (round(SINGLE_CORE_FRACTION * one_core_pct, 1)
                                        if one_core_pct else None),
                },
            },
        })

    source = benchmark_result.get("pipeline_source") or ""
    if infer_pct >= INFERENCE_STAGE_PCT_THRESHOLD and "CPUExecutionProvider" in source:
        findings.append({
            "type": "INFERENCE_ON_CPU",
            "evidence_strength": round(min(0.9, 0.4 + infer_pct / 200), 2),
            "detail": f"Inference dominates latency ({infer_pct:.1f}% of pipeline) and runs "
                      f"on onnxruntime's CPUExecutionProvider — the GPU is not used for "
                      f"the model at all.",
            "recommendation": "Run on an accelerator provider: on Jetson, install the "
                              "JetPack-matched onnxruntime-gpu wheel and rerun with "
                              "--provider CUDAExecutionProvider or TensorrtExecutionProvider "
                              "(`edgelens doctor` shows which providers are available).",
            "evidence": {
                "inference_stage_pct": round(infer_pct, 1),
                "pipeline_source": source,
                "threshold_pct": INFERENCE_STAGE_PCT_THRESHOLD,
            },
        })

    if transfer_pct >= TRANSFER_STAGE_PCT_THRESHOLD:
        findings.append({
            "type": "MEMORY_TRANSFER_BOUND",
            "evidence_strength": round(min(0.85, 0.35 + transfer_pct / 100), 2),
            "detail": f"Host<->device memory transfers consume {transfer_pct:.1f}% of "
                      f"total latency.",
            "recommendation": "Use pinned/zero-copy memory, or keep pre/postprocessing "
                              "on-GPU to avoid repeated CPU<->GPU copies.",
            "evidence": {
                "transfer_stage_pct": round(transfer_pct, 1),
                "h2d_pct": round(stage_pct.get("h2d_copy", 0), 1),
                "d2h_pct": round(stage_pct.get("d2h_copy", 0), 1),
                "threshold_pct": TRANSFER_STAGE_PCT_THRESHOLD,
            },
        })

    if (infer_pct >= INFERENCE_STAGE_PCT_THRESHOLD and gpu >= GPU_BUSY_THRESHOLD
            and not findings):
        findings.append({
            "type": "GPU_BOUND",
            "evidence_strength": round(min(0.9, 0.35 + gpu / 100), 2),
            "detail": f"Inference dominates latency ({infer_pct:.1f}% of pipeline) with "
                      f"mean GPU utilization at {gpu:.1f}%. This is the healthy/expected "
                      f"case for a well-optimized pipeline — further gains come from the "
                      f"model itself, not the surrounding pipeline.",
            "recommendation": "Try a lower precision (FP16/INT8 via TensorRT), a lighter "
                              "model architecture, or a smaller input resolution.",
            "evidence": {
                "inference_stage_pct": round(infer_pct, 1),
                "gpu_percent_mean": round(gpu, 1),
                "thresholds": {
                    "stage_pct": INFERENCE_STAGE_PCT_THRESHOLD,
                    "gpu_pct": GPU_BUSY_THRESHOLD,
                },
            },
        })

    if not findings:
        findings.append({
            "type": "BALANCED",
            "evidence_strength": 0.5,
            "detail": "No single stage or resource dominates the latency budget — the "
                      "pipeline appears reasonably balanced.",
            "recommendation": "Consider a precision/power-mode sweep (edgelens roadmap "
                              "v0.4) to find further headroom.",
            "evidence": {
                "stage_pct": {k: round(v, 1) for k, v in stage_pct.items()},
                "cpu_percent_mean": round(cpu, 1),
                "gpu_percent_mean": round(gpu, 1),
            },
        })

    # Demote (not delete) telemetry-dependent findings when the sample
    # count is too low to trust — see module docstring for the mechanism.
    if not telemetry_reliable:
        for f in findings:
            if f["type"] in TELEMETRY_DEPENDENT_TYPES:
                f["evidence_strength"] = round(f["evidence_strength"] * LOW_SAMPLE_STRENGTH_PENALTY, 2)
                f["detail"] += (
                    f" [LOW CONFIDENCE: only {sample_count} telemetry sample(s) were "
                    f"collected — the benchmark likely completed faster than the "
                    f"telemetry sampling interval, so this reflects a single "
                    f"instantaneous reading, not sustained load during the benchmark. "
                    f"Increase --iterations or use a heavier model for a reliable "
                    f"telemetry-based verdict.]"
                )
                f["evidence"]["telemetry_sample_count"] = sample_count
                f["evidence"]["telemetry_reliable"] = False

    findings.sort(key=lambda f: f["evidence_strength"], reverse=True)

    return {
        "stage_pct": {k: round(v, 1) for k, v in stage_pct.items()},
        "resource_snapshot": {
            "cpu_percent_mean": round(cpu, 1),
            "gpu_percent_mean": round(gpu, 1),
            "mem_percent_mean": round(mem_pct, 1),
            "max_temp_c": round(max_temp, 1),
        },
        "telemetry_reliable": telemetry_reliable,
        "telemetry_sample_count": sample_count,
        "primary": findings[0],
        "secondary": findings[1:3],
        "all_findings": findings,
    }

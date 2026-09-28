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
    ad-hoc scoring formula. That's a fine heuristic, but the WORD
    "confidence" oversold it. Every finding now reports:
      - evidence_strength: the same 0-1 heuristic score, honestly named
      - evidence: the actual measured numbers that produced the verdict,
        so the developer can check the reasoning themselves rather than
        trust a label
    Nothing about the underlying logic changed — only what it's called
    and how transparently it shows its work.

Verdict types: CPU_BOUND_PREPROCESS, MEMORY_TRANSFER_BOUND, GPU_BOUND,
THERMAL, MEMORY_BOUND, BALANCED.
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


def diagnose(benchmark_result):
    stages = benchmark_result.get("stage_avg_ms", {})
    total = sum(stages.values()) or 1e-9
    stage_pct = {k: (v / total) * 100 for k, v in stages.items()}

    tel = benchmark_result.get("telemetry", {}) or {}
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

    if pre_pct >= PREPROCESS_STAGE_PCT_THRESHOLD and cpu >= PREPROCESS_CPU_THRESHOLD:
        findings.append({
            "type": "CPU_BOUND_PREPROCESS",
            "evidence_strength": round(min(0.95, 0.4 + pre_pct / 100), 2),
            "detail": f"Preprocessing consumes {pre_pct:.1f}% of total pipeline latency "
                      f"while mean CPU utilization is {cpu:.1f}%.",
            "recommendation": "Move resize/color-conversion to the GPU (CUDA/VPI) or use "
                              "hardware-accelerated decode instead of CPU-side OpenCV.",
            "evidence": {
                "preprocess_stage_pct": round(pre_pct, 1),
                "cpu_percent_mean": round(cpu, 1),
                "thresholds": {
                    "stage_pct": PREPROCESS_STAGE_PCT_THRESHOLD,
                    "cpu_pct": PREPROCESS_CPU_THRESHOLD,
                },
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

    findings.sort(key=lambda f: f["evidence_strength"], reverse=True)

    return {
        "stage_pct": {k: round(v, 1) for k, v in stage_pct.items()},
        "resource_snapshot": {
            "cpu_percent_mean": round(cpu, 1),
            "gpu_percent_mean": round(gpu, 1),
            "mem_percent_mean": round(mem_pct, 1),
            "max_temp_c": round(max_temp, 1),
        },
        "primary": findings[0],
        "secondary": findings[1:3],
        "all_findings": findings,
    }

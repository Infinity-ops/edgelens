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

v0.1.0: findings report evidence_strength as a CATEGORY (weak / moderate /
strong) plus the raw evidence numbers. The 0-1 number is kept only as
`rank_score` for ordering, because a value like 0.73 reads as a calibrated
probability and it is not one. Rules use stage ROLES, so they apply to any
pipeline (vision, timeseries, custom), and scenario packs add their own
findings. DEADLINE_MISSED fires for any run with a deadline.

Verdict types: DEADLINE_MISSED, CPU_BOUND_PREPROCESS, INFERENCE_ON_CPU, MEMORY_TRANSFER_BOUND,
GPU_BOUND, THERMAL, MEMORY_BOUND, BALANCED.
"""

from ..core.schema import SCHEMA_VERSION

# Categorical evidence strength cut-offs on the internal rank_score.
STRONG_AT = 0.75
MODERATE_AT = 0.5

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

# Worst-case headroom at which bottleneck findings become informational.
HEADROOM_DEMOTE_PCT = 50.0
BOTTLENECK_TYPES = {"CPU_BOUND_PREPROCESS", "MEMORY_TRANSFER_BOUND", "GPU_BOUND",
                    "INFERENCE_ON_CPU", "BALANCED"}

TELEMETRY_DEPENDENT_TYPES = {"THERMAL", "MEMORY_BOUND", "CPU_BOUND_PREPROCESS", "GPU_BOUND"}
# INFERENCE_ON_CPU is deliberately NOT telemetry-dependent: it's decided
# from the provider recorded in pipeline_source, not from sampled load.


def strength_label(score):
    """Categorical evidence strength from the internal ordering score.
    The score is a heuristic for ranking findings, NOT a probability, so it
    is never shown as a percentage."""
    if score >= STRONG_AT:
        return "strong"
    if score >= MODERATE_AT:
        return "moderate"
    return "weak"


def _stage_roles(benchmark_result):
    """{stage_name: role} from the result's pipeline description, falling
    back to name-based inference for pre-v0.1.0 files."""
    from ..core.pipeline import infer_role
    from ..packs.vision import VisionPack
    described = (benchmark_result.get("pipeline") or {}).get("stages")
    if described:
        return {s["name"]: s.get("role") or infer_role(s["name"]) for s in described}
    vision = VisionPack()
    return {n: vision.role_for(n) or infer_role(n)
            for n in (benchmark_result.get("stage_avg_ms") or {})}


def diagnose(benchmark_result):
    stages = benchmark_result.get("stage_avg_ms", {})
    total = sum(stages.values()) or 1e-9
    stage_pct = {k: (v / total) * 100 for k, v in stages.items()}
    roles = _stage_roles(benchmark_result)
    role_pct = {}
    for name, pct in stage_pct.items():
        role = roles.get(name, "other")
        role_pct[role] = role_pct.get(role, 0.0) + pct

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

    # Rules reason about ROLES, so a sensor pipeline's "filter"+"fft" count
    # as preprocessing exactly like a camera pipeline's "preprocess".
    pre_pct = role_pct.get("preprocess", 0)
    transfer_pct = role_pct.get("transfer", 0)
    infer_pct = role_pct.get("inference", 0)
    transfer_names = [n for n in stage_pct if roles.get(n) == "transfer"]

    findings = []

    if max_temp >= THERMAL_LIMIT_C:
        findings.append({
            "type": "THERMAL",
            "rank_score": round(min(0.95, 0.5 + (max_temp - THERMAL_LIMIT_C) / 40), 2),
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
            "rank_score": round(min(0.9, 0.5 + (mem_pct - MEMORY_LIMIT_PCT) / 30), 2),
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
            "rank_score": round(min(0.95, 0.4 + pre_pct / 100), 2),
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
            "rank_score": round(min(0.9, 0.4 + infer_pct / 200), 2),
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
            "rank_score": round(min(0.85, 0.35 + transfer_pct / 100), 2),
            "detail": f"Host<->device memory transfers consume {transfer_pct:.1f}% of "
                      f"total latency.",
            "recommendation": "Use pinned/zero-copy memory, or keep pre/postprocessing "
                              "on-GPU to avoid repeated CPU<->GPU copies.",
            "evidence": {
                "transfer_stage_pct": round(transfer_pct, 1),
                "per_stage_pct": {n: round(stage_pct[n], 1) for n in transfer_names},
                "threshold_pct": TRANSFER_STAGE_PCT_THRESHOLD,
            },
        })

    if infer_pct >= INFERENCE_STAGE_PCT_THRESHOLD and gpu >= GPU_BUSY_THRESHOLD:
        findings.append({
            "type": "GPU_BOUND",
            "rank_score": round(min(0.9, 0.35 + gpu / 100), 2),
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

    dl = benchmark_result.get("deadline")
    if dl and dl.get("misses"):
        worst_stage = max(stage_pct, key=stage_pct.get) if stage_pct else None
        lat = benchmark_result.get("latency") or {}
        findings.append({
            "type": "DEADLINE_MISSED",
            "rank_score": round(min(0.95, 0.6 + dl["miss_ratio"] * 3.5), 2),
            "detail": f"{dl['misses']} of {dl['iterations']} iterations "
                      f"({dl['miss_ratio']*100:.2f}%) exceeded the {dl['deadline_ms']} ms "
                      f"deadline; worst {dl['worst_ms']} ms, longest burst "
                      f"{dl['max_consecutive_misses']} in a row. Largest share of the "
                      f"budget: '{worst_stage}' ({stage_pct.get(worst_stage, 0):.1f}%).",
            "recommendation": "Fix the bottleneck findings listed with this one first; if "
                              "misses are rare bursts, look for stalls in the trace "
                              "(thermal, other processes, GC) rather than average speed.",
            "evidence": {
                "deadline_ms": dl["deadline_ms"], "misses": dl["misses"],
                "miss_ratio": dl["miss_ratio"], "worst_ms": dl["worst_ms"],
                "max_consecutive_misses": dl["max_consecutive_misses"],
                "latency_p99_ms": lat.get("p99"), "dominant_stage": worst_stage,
            },
        })

    # A requirement that is met with plenty of headroom changes what the
    # bottleneck findings MEAN: "where the time goes / where to optimise for
    # energy", not "your pipeline has a problem". Demote them and say so.
    if dl and not dl.get("misses") and dl.get("deadline_ms"):
        worst = dl.get("worst_ms") or 0.0
        headroom_pct = 100.0 * (1 - worst / dl["deadline_ms"])
        if headroom_pct >= HEADROOM_DEMOTE_PCT:
            for f in findings:
                if f["type"] in BOTTLENECK_TYPES:
                    f["rank_score"] = round(f["rank_score"] * 0.5, 2)
                    f["detail"] += (f" [Not a problem for the stated requirement: the "
                                    f"{dl['deadline_ms']} ms deadline is met with "
                                    f"{headroom_pct:.0f}% headroom even in the worst "
                                    f"iteration. Optimise here only for more headroom or "
                                    f"lower energy.]")
        findings.append({
            "type": "DEADLINE_MET",
            "rank_score": 0.8 if headroom_pct >= 20 else 0.6,
            "detail": f"All {dl['iterations']} iterations finished within the "
                      f"{dl['deadline_ms']} ms deadline; worst {worst} ms "
                      f"({headroom_pct:.0f}% headroom).",
            "recommendation": ("Validate under sustained load and at the target power "
                               "mode before relying on this margin." if headroom_pct >= 20
                               else "The margin is thin: check the tail (p99.9/max) under "
                                    "longer runs and thermal load."),
            "evidence": {"deadline_ms": dl["deadline_ms"], "iterations": dl["iterations"],
                         "worst_ms": worst, "headroom_pct": round(headroom_pct, 1),
                         "latency_p99_ms": (benchmark_result.get("latency") or {}).get("p99")},
        })

    pack_name = (benchmark_result.get("pipeline") or {}).get("pack")
    if pack_name:
        from ..packs import get_pack
        try:
            findings.extend(get_pack(pack_name).findings(benchmark_result))
        except ValueError:
            pass   # result written by an unknown/third-party pack not installed here

    if not findings:
        findings.append({
            "type": "BALANCED",
            "rank_score": 0.5,
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
                f["rank_score"] = round(f["rank_score"] * LOW_SAMPLE_STRENGTH_PENALTY, 2)
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

    for f in findings:
        f["evidence_strength"] = strength_label(f["rank_score"])
    findings.sort(key=lambda f: f["rank_score"], reverse=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "data_quality": {
            "iterations": benchmark_result.get("iterations"),
            "telemetry_sample_count": sample_count,
            "telemetry_reliable": telemetry_reliable,
            "mode": benchmark_result.get("mode"),
        },
        "role_pct": {k: round(v, 1) for k, v in role_pct.items()},
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

from edgelens.diagnose.engine import diagnose


def _bench(stages, cpu=45, gpu=50, mem=40, max_temp=55, sample_count=50):
    return {
        "stage_avg_ms": stages,
        "telemetry": {
            "cpu_percent_mean": cpu,
            "gpu_percent_mean": gpu,
            "mem_percent_mean": mem,
            "max_temp_c": max_temp,
            "sample_count": sample_count,
        },
    }


def test_cpu_bound_preprocess_detected():
    result = _bench(
        stages={"capture": 3, "preprocess": 18, "h2d_copy": 2, "inference": 10,
                "d2h_copy": 1, "postprocess": 2},
        cpu=92, gpu=35, max_temp=55,
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "CPU_BOUND_PREPROCESS"
    assert "evidence_strength" in verdict["primary"]
    assert "confidence" not in verdict["primary"]
    assert verdict["primary"]["evidence"]["cpu_percent_mean"] == 92.0


def test_thermal_takes_priority_when_hot():
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 20,
                "d2h_copy": 1, "postprocess": 2},
        cpu=50, gpu=60, max_temp=85,
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "THERMAL"
    assert verdict["primary"]["evidence"]["peak_temperature_c"] == 85.0


def test_memory_transfer_bound_detected():
    result = _bench(
        stages={"capture": 2, "preprocess": 4, "h2d_copy": 10, "inference": 10,
                "d2h_copy": 8, "postprocess": 2},
        cpu=40, gpu=50, max_temp=55,
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "MEMORY_TRANSFER_BOUND"


def test_gpu_bound_when_inference_dominates():
    result = _bench(
        stages={"capture": 1, "preprocess": 2, "h2d_copy": 1, "inference": 25,
                "d2h_copy": 1, "postprocess": 1},
        cpu=30, gpu=95, max_temp=55,
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "GPU_BOUND"


def test_balanced_when_nothing_dominates():
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 8,
                "d2h_copy": 1, "postprocess": 4},
        cpu=45, gpu=50, max_temp=55,
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "BALANCED"


def test_memory_bound_detected():
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 8,
                "d2h_copy": 1, "postprocess": 4},
        cpu=50, gpu=50, mem=90, max_temp=55,
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "MEMORY_BOUND"


def test_evidence_strength_is_within_unit_range():
    result = _bench(
        stages={"capture": 3, "preprocess": 18, "h2d_copy": 2, "inference": 10,
                "d2h_copy": 1, "postprocess": 2},
        cpu=92, gpu=35, max_temp=55,
    )
    verdict = diagnose(result)
    for finding in verdict["all_findings"]:
        assert finding["evidence_strength"] in ("weak", "moderate", "strong")
        assert 0.0 <= finding["rank_score"] <= 1.0


def test_low_sample_count_demotes_telemetry_dependent_findings():
    # Reproduces the exact real-world case: a benchmark that finishes
    # faster than one telemetry sampling interval gets only 1 sample,
    # and a THERMAL finding from that single reading must be demoted
    # and clearly caveated, not presented as a confident verdict.
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 8,
                "d2h_copy": 1, "postprocess": 4},
        cpu=50, gpu=50, max_temp=93.0, sample_count=1,
    )
    verdict = diagnose(result)
    assert verdict["telemetry_reliable"] is False
    assert verdict["primary"]["type"] == "THERMAL"
    assert "LOW CONFIDENCE" in verdict["primary"]["detail"]
    assert verdict["primary"]["evidence"]["telemetry_reliable"] is False
    # Strength must be lower than the un-demoted value would have been
    # (0.5 + (93-80)/40 = 0.825, capped/rounded to 0.82 before demotion)
    assert verdict["primary"]["rank_score"] < 0.82
    assert verdict["primary"]["evidence_strength"] == "weak"   # demoted from strong


def test_sufficient_sample_count_is_not_demoted():
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 8,
                "d2h_copy": 1, "postprocess": 4},
        cpu=50, gpu=50, max_temp=93.0, sample_count=50,
    )
    verdict = diagnose(result)
    assert verdict["telemetry_reliable"] is True
    assert "LOW CONFIDENCE" not in verdict["primary"]["detail"]


# --- Regression tests from real Jetson Nano runs (JetPack 4.6, 4 cores) ---

def test_nano_cpu_provider_run_is_inference_on_cpu_not_balanced():
    # Real numbers: CPUExecutionProvider, inference 91% of frame, CPU 100%.
    result = _bench(
        stages={"capture": 0.01, "preprocess": 4.00, "h2d_copy": 0.25,
                "inference": 44.63, "d2h_copy": 0.01, "postprocess": 0.07},
        cpu=100.0, gpu=0.0, mem=77, max_temp=54.5, sample_count=21,
    )
    result["pipeline_source"] = "onnxruntime:CPUExecutionProvider (small_cnn.onnx)"
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "INFERENCE_ON_CPU"


def test_nano_cuda_run_flags_single_threaded_preprocess():
    # Real numbers: CUDAExecutionProvider, preprocess 33.6%, CPU mean 33% on
    # 4 cores (= one saturated core), GPU 48%. Used to come out BALANCED.
    result = _bench(
        stages={"capture": 0.01, "preprocess": 2.50, "h2d_copy": 0.61,
                "inference": 4.08, "d2h_copy": 0.18, "postprocess": 0.07},
        cpu=33.11, gpu=48.19, mem=77, max_temp=53.0, sample_count=21,
    )
    result["telemetry"]["cpu_count"] = 4
    result["pipeline_source"] = "onnxruntime:CUDAExecutionProvider (small_cnn.onnx)"
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "CPU_BOUND_PREPROCESS"
    assert "single-threaded" in verdict["primary"]["detail"]


def test_single_core_rule_needs_cpu_count():
    # Older JSON without cpu_count keeps the original all-cores threshold.
    result = _bench(
        stages={"capture": 0.01, "preprocess": 2.50, "h2d_copy": 0.61,
                "inference": 4.08, "d2h_copy": 0.18, "postprocess": 0.07},
        cpu=33.11, gpu=48.19, sample_count=21,
    )
    assert diagnose(result)["primary"]["type"] == "BALANCED"


def test_gpu_provider_never_reports_inference_on_cpu():
    result = _bench(
        stages={"capture": 1, "preprocess": 2, "h2d_copy": 1, "inference": 30,
                "d2h_copy": 1, "postprocess": 1},
        cpu=30, gpu=95, sample_count=50,
    )
    result["pipeline_source"] = "onnxruntime:TensorrtExecutionProvider (m.onnx)"
    assert diagnose(result)["primary"]["type"] != "INFERENCE_ON_CPU"


# --- v0.1.0: roles, deadlines, packs, categorical strength ---

def test_rules_use_roles_so_sensor_pipelines_get_preprocess_verdicts():
    result = _bench(stages={"acquire": 0.2, "filter": 3.0, "fft": 4.0, "inference": 2.0},
                    cpu=92, gpu=10)
    result["pipeline"] = {"pack": "custom", "stages": [
        {"name": "acquire", "role": "input"}, {"name": "filter", "role": "preprocess"},
        {"name": "fft", "role": "preprocess"}, {"name": "inference", "role": "inference"}]}
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "CPU_BOUND_PREPROCESS"
    assert verdict["role_pct"]["preprocess"] > 70


def test_deadline_missed_is_reported_with_dominant_stage():
    result = _bench(stages={"capture": 1, "preprocess": 2, "inference": 20})
    result["iterations"] = 100
    result["deadline"] = {"deadline_ms": 20, "iterations": 100, "misses": 30,
                          "miss_ratio": 0.3, "worst_ms": 31.0, "max_consecutive_misses": 4}
    verdict = diagnose(result)
    f = next(f for f in verdict["all_findings"] if f["type"] == "DEADLINE_MISSED")
    assert f["evidence"]["dominant_stage"] == "inference"
    assert f["evidence_strength"] == "strong"


def test_strength_labels():
    from edgelens.diagnose.engine import strength_label
    assert [strength_label(x) for x in (0.2, 0.5, 0.74, 0.75, 0.95)] == \
        ["weak", "moderate", "moderate", "strong", "strong"]


def test_diagnosis_has_schema_version():
    from edgelens.core.schema import SCHEMA_VERSION
    v = diagnose(_bench(stages={"capture": 1, "inference": 1}))
    assert v["schema_version"] == SCHEMA_VERSION

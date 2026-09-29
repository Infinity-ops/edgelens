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
        assert 0.0 <= finding["evidence_strength"] <= 1.0


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
    assert verdict["primary"]["evidence_strength"] < 0.82


def test_sufficient_sample_count_is_not_demoted():
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 8,
                "d2h_copy": 1, "postprocess": 4},
        cpu=50, gpu=50, max_temp=93.0, sample_count=50,
    )
    verdict = diagnose(result)
    assert verdict["telemetry_reliable"] is True
    assert "LOW CONFIDENCE" not in verdict["primary"]["detail"]

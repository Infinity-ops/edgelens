from edgelens.diagnose.engine import diagnose


def _bench(stages, telemetry):
    return {"stage_avg_ms": stages, "telemetry": telemetry}


def test_cpu_bound_preprocess_detected():
    result = _bench(
        stages={"capture": 3, "preprocess": 18, "h2d_copy": 2, "inference": 10,
                "d2h_copy": 1, "postprocess": 2},
        telemetry={"cpu_percent": 92, "gpu_percent": 35, "mem_percent": 40,
                   "temps_c": {"CPU-therm": 55}},
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "CPU_BOUND_PREPROCESS"


def test_thermal_takes_priority_when_hot():
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 20,
                "d2h_copy": 1, "postprocess": 2},
        telemetry={"cpu_percent": 50, "gpu_percent": 60, "mem_percent": 40,
                   "temps_c": {"CPU-therm": 85}},
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "THERMAL"


def test_memory_transfer_bound_detected():
    result = _bench(
        stages={"capture": 2, "preprocess": 4, "h2d_copy": 10, "inference": 10,
                "d2h_copy": 8, "postprocess": 2},
        telemetry={"cpu_percent": 40, "gpu_percent": 50, "mem_percent": 40,
                   "temps_c": {"CPU-therm": 55}},
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "MEMORY_TRANSFER_BOUND"


def test_gpu_bound_when_inference_dominates():
    result = _bench(
        stages={"capture": 1, "preprocess": 2, "h2d_copy": 1, "inference": 25,
                "d2h_copy": 1, "postprocess": 1},
        telemetry={"cpu_percent": 30, "gpu_percent": 95, "mem_percent": 40,
                   "temps_c": {"CPU-therm": 55}},
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "GPU_BOUND"


def test_balanced_when_nothing_dominates():
    result = _bench(
        stages={"capture": 3, "preprocess": 5, "h2d_copy": 2, "inference": 8,
                "d2h_copy": 1, "postprocess": 4},
        telemetry={"cpu_percent": 45, "gpu_percent": 50, "mem_percent": 40,
                   "temps_c": {"CPU-therm": 55}},
    )
    verdict = diagnose(result)
    assert verdict["primary"]["type"] == "BALANCED"

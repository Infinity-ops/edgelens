"""
edgelens.hardware.readiness
-----------------------------
"Is this board in a state that gives trustworthy benchmark numbers?"

Shown by `edgelens doctor` as *Benchmark readiness*. Every check comes from
something that actually went wrong on a real Jetson Nano:

* clocks not pinned  -> p99 148 ms vs 9.4 ms pinned, same model and code;
* onnxruntime from PyPI over NVIDIA's GPU build -> CUDA/TensorRT silently
  gone ("Available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']");
* power sensor root-only (JetPack 4.6) -> no power/energy without a chmod.

`assess()` is pure (data in, findings out) so it is unit-tested with those
real values; `gather()` collects the live data.
"""

OK, WARN, INFO = "ok", "warn", "info"

GPU_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")


def _check(name, status, value, advice=None, fix=None):
    """advice: one short sentence for the table. fix: an exact, copyable
    command (printed in full under the table, never truncated)."""
    return {"name": name, "status": status, "value": value, "advice": advice, "fix": fix}


def assess(is_jetson, power_mode, cpu_pinned, gpu_pinned, providers, power_probe,
           unloadable=()):
    """Return a list of readiness checks.

    is_jetson    bool
    power_mode   {"id": int, "name": str} or None
    cpu_pinned / gpu_pinned   True / False / None (unknown)
    providers    onnxruntime execution providers ([] when not installed)
    unloadable   listed GPU providers whose libraries cannot load (CUDA/cuDNN
                 missing or mismatched): they would silently fall back to CPU
    power_probe  telemetry.probe_sources()["power_ina3221"]
    """
    checks = []
    if is_jetson:
        if power_mode:
            label = power_mode.get("name") or f"mode {power_mode.get('id')}"
            checks.append(_check(
                "Power mode", INFO, label,
                "Results are only comparable within the same power mode; "
                "`compare` warns when it differs."))
        else:
            checks.append(_check("Power mode", INFO, "not detected"))

        if cpu_pinned is False or gpu_pinned is False:
            unpinned = [n for n, v in (("CPU", cpu_pinned), ("GPU", gpu_pinned)) if v is False]
            checks.append(_check(
                "Clocks pinned", WARN, f"no ({', '.join(unpinned)} scaling)",
                "Clock scaling inflates the latency tail (Nano: p99 148 ms unpinned "
                "vs 9.4 ms pinned). Pin for benchmarks; undo with "
                "`sudo jetson_clocks --restore`.",
                fix="sudo jetson_clocks --store && sudo jetson_clocks"))
        elif cpu_pinned and gpu_pinned:
            checks.append(_check("Clocks pinned", OK, "yes (CPU and GPU)"))
        else:
            checks.append(_check("Clocks pinned", INFO, "unknown"))

    if not providers:
        checks.append(_check(
            "ONNX Runtime", INFO, "not installed",
            "Needed only for `benchmark --model`. On Jetson install NVIDIA's "
            "onnxruntime-gpu wheel; elsewhere: pip install \"edgelens[onnx]\"."))
    else:
        gpu = [p.replace("ExecutionProvider", "") for p in providers
               if p in GPU_PROVIDERS and p not in unloadable]
        if unloadable:
            from ..benchmark.onnx_pipeline import CUDA_UNLOADABLE_ADVICE
            checks.append(_check(
                "GPU inference", WARN,
                "listed, cannot load (" + ", ".join(
                    p.replace("ExecutionProvider", "") for p in unloadable) + ")",
                CUDA_UNLOADABLE_ADVICE))
        elif gpu:
            checks.append(_check("GPU inference", OK, ", ".join(gpu)))
        elif is_jetson:
            checks.append(_check(
                "GPU inference", WARN, "CPU only",
                "onnxruntime has no CUDA/TensorRT, often because the PyPI build "
                "replaced NVIDIA's. Reinstall the JetPack-matched onnxruntime-gpu "
                "wheel (Jetson Zoo) after uninstalling.",
                fix="pip uninstall -y onnxruntime"))
        else:
            checks.append(_check("GPU inference", INFO, "CPU only (no NVIDIA GPU provider)"))

    if is_jetson:
        pw = power_probe or {}
        if pw.get("total_w") is not None:
            checks.append(_check("Power sensor", OK, f"{pw['total_w']:.2f} W ({pw.get('method')})"))
        elif pw.get("permission_denied"):
            checks.append(_check("Power sensor", WARN, "root-only",
                                 "No power or energy until the sensor files are "
                                 "readable (resets at reboot).", fix=pw.get("fix")))
        else:
            checks.append(_check("Power sensor", INFO, "not found",
                                 "Power and energy will not be reported on this board."))
    return checks


def summary(checks, is_jetson=True):
    warnings = [c for c in checks if c["status"] == WARN]
    if not is_jetson:
        if warnings:   # e.g. GPU providers listed but unloadable on a laptop
            return WARN, (f"{len(warnings)} item(s) can distort or limit results: "
                          + ", ".join(c["name"] for c in warnings) + ".")
        return INFO, ("Not a Jetson: no board-level checks apply. Latency, tail and "
                      "deadline results are valid for this host.")
    if not warnings:
        return OK, ("Ready: clocks pinned, GPU inference available, power readable. "
                    "Results from this board are comparable run to run.")
    return WARN, (f"{len(warnings)} item(s) can distort or limit results: "
                  + ", ".join(c["name"] for c in warnings) + ".")


def gather(is_jetson, power_probe):
    """Collect live inputs for assess() on this host."""
    from ..benchmark.onnx_pipeline import available_providers, unloadable_providers
    from ..core.identity import detect_clock_locking, detect_power_mode

    cpu, gpu = detect_clock_locking() if is_jetson else (None, None)
    return assess(is_jetson=is_jetson,
                  power_mode=detect_power_mode() if is_jetson else None,
                  cpu_pinned=cpu, gpu_pinned=gpu,
                  providers=available_providers(), power_probe=power_probe,
                  unloadable=unloadable_providers())

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
           unloadable=(), gpu_reason=None, gpu_kind=None, gpu_cached_at=None, storage=None,
           clock_unset=None, jetson_clocks=True):
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
            from .platform_hints import power_mode_unreadable_reason
            checks.append(_check("Power mode", INFO, "not detected",
                                 power_mode_unreadable_reason(True)))

        if cpu_pinned is False or gpu_pinned is False:
            unpinned = [n for n, v in (("CPU", cpu_pinned), ("GPU", gpu_pinned)) if v is False]
            checks.append(_check(
                "Clocks pinned", WARN, f"no ({', '.join(unpinned)} scaling)",
                "Clock scaling inflates the latency tail (Nano: p99 148 ms unpinned "
                "vs 9.4 ms pinned). Pin for benchmarks; undo with "
                "`sudo jetson_clocks --restore`.",
                fix=_pin_fix(jetson_clocks)))
        elif cpu_pinned and gpu_pinned:
            checks.append(_check("Clocks pinned", OK, "yes (CPU and GPU)"))
        else:
            checks.append(_check("Clocks pinned", INFO, "unknown"))

    if clock_unset:
        checks.append(_check(
            "System clock", WARN, f"never set ({clock_unset})",
            "No RTC time and no NTP: result timestamps, run ids and evidence files carry this "
            "date (ids stay unique). Comparing runs by date across reboots means nothing.",
            fix="date -s 'YYYY-MM-DD HH:MM'   (until reboot), or enable NTP / chrony / "
                "systemd-timesyncd in the image"))

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
            # the check's own finding: "missing libraries" advice only when that
            # is what it found (a crash or a hang has a different cause and fix)
            advice = (gpu_reason or CUDA_UNLOADABLE_ADVICE) if gpu_kind in ("crash", "hang") \
                else ((gpu_reason + ". " if gpu_reason else "") + CUDA_UNLOADABLE_ADVICE)
            checks.append(_check(
                "GPU inference", WARN,
                "listed, cannot load (" + ", ".join(
                    p.replace("ExecutionProvider", "") for p in unloadable) + ")",
                advice, fix="edgelens doctor --no-cache    # after fixing: test again"))
        elif gpu:
            note = (f"verified with a CUDA session; cached from "
                    f"{gpu_cached_at.replace('T', ' ')} — rerun with --no-cache"
                    if gpu_cached_at else "verified with a CUDA session just now"
                    if gpu_kind == "ok" else None)
            checks.append(_check("GPU inference", OK, ", ".join(gpu), note))
        elif is_jetson:
            checks.append(_check(
                "GPU inference", WARN, "CPU only",
                "onnxruntime has no CUDA/TensorRT, often because the PyPI build "
                "replaced NVIDIA's. Reinstall the JetPack-matched onnxruntime-gpu "
                "wheel (Jetson Zoo) after uninstalling.",
                fix="pip uninstall -y onnxruntime"))
        else:
            checks.append(_check("GPU inference", INFO, "CPU only (no NVIDIA GPU provider)"))

    if storage is not None:
        from .storage import STORAGE_ADVICE
        if storage.get("count"):
            checks.append(_check("Storage", WARN,
                                 f"{storage['count']} I/O error(s) in the kernel log",
                                 STORAGE_ADVICE + " Last: " + storage["lines"][-1][:160]))
        elif storage.get("remembered"):
            r = storage["remembered"][0]
            checks.append(_check(
                "Storage", WARN,
                f"this card ({r['device']}) logged {r['count']} I/O error(s) "
                f"(last {r.get('last_seen', '?').replace('T', ' ')}, an earlier boot)",
                STORAGE_ADVICE + " A reboot clears the kernel log, not the card. Last: "
                + (r.get("last_line") or "")[:160],
                fix="edgelens doctor --forget-storage-errors   # only after testing or replacing the card"))
        elif storage.get("readable"):
            checks.append(_check("Storage", OK, "no I/O errors in the kernel log since boot"))
        elif is_jetson:
            checks.append(_check("Storage", INFO, "kernel log not readable",
                                 "Can't check the SD card / disk for errors without root: "
                                 "sudo dmesg | grep -i 'I/O error'"))

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
    from ..benchmark.onnx_pipeline import (available_providers, cuda_probe, cuda_probe_cached_at,
                                           cuda_probe_kind, unloadable_providers)
    from ..core.identity import detect_clock_locking, detect_power_mode
    from .storage import kernel_storage_errors

    cpu, gpu = detect_clock_locking() if is_jetson else (None, None)
    providers = available_providers()
    reason = kind = cached_at = None
    if any(p in GPU_PROVIDERS for p in providers):
        reason, kind, cached_at = cuda_probe()[1], cuda_probe_kind(), cuda_probe_cached_at()
    return assess(is_jetson=is_jetson,
                  power_mode=detect_power_mode() if is_jetson else None,
                  cpu_pinned=cpu, gpu_pinned=gpu,
                  providers=providers, power_probe=power_probe,
                  unloadable=unloadable_providers(), gpu_reason=reason, gpu_kind=kind,
                  gpu_cached_at=cached_at, storage=kernel_storage_errors(),
                  clock_unset=_clock_unset(), jetson_clocks=_has_jetson_clocks())


def _clock_unset():
    from .platform_hints import clock_unset
    return clock_unset()


def _has_jetson_clocks():
    from .platform_hints import jetson_clocks_available
    return jetson_clocks_available()


def _pin_fix(jetson_clocks):
    from .platform_hints import pin_clocks_fix
    return ("sudo jetson_clocks --store && sudo jetson_clocks" if jetson_clocks
            else pin_clocks_fix(available=False))

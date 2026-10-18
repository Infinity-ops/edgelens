"""Layer 2 findings: output equivalence (sweep), duty-cycle penalty
(periodic), and soak findings (drift, throttling, memory)."""


def _f(type_, score, detail, rec, evidence):
    return {"type": type_, "rank_score": score, "detail": detail, "recommendation": rec,
            "evidence": evidence}


def findings(result):
    out = []
    eq = result.get("equivalence")
    if eq and not eq.get("reference") and eq.get("verdict") != "EQUIVALENT":
        backend = (result.get("backend") or {}).get("name", "this backend")
        flips = eq.get("decisions_flipped") or 0
        if eq["verdict"] == "DECISIONS_DIFFER":
            detail = (f"{backend} flipped {flips} of {eq.get('decisions_compared')} decisions "
                      f"compared with {eq.get('reference_backend')} (max |error| "
                      f"{eq['max_abs_error']:.3g}). It is faster or slower — but it is not the "
                      f"same model any more.")
            score = 0.95
        else:
            agree = eq.get("decision_agreement")
            detail = (f"{backend} outputs differ from {eq.get('reference_backend')} by up to "
                      f"{eq['max_abs_error']:.3g} "
                      + ("(no decisions compared: the outputs have none — use --decision / "
                         "--threshold)." if agree is None else
                         f"(decisions still agree: {agree * 100:.2f}%)."))
            score = 0.6
        out.append(_f("OUTPUT_DRIFT", score, detail,
                      "Check the drift on REAL inputs (--inputs), set a tolerance the "
                      "application can live with (validate --max-output-error / "
                      "--min-decision-agreement), or keep FP32 for the layers that drift.",
                      {"verdict": eq["verdict"], "max_abs_error": eq["max_abs_error"],
                       "decision_agreement": eq.get("decision_agreement"),
                       "inputs": eq.get("inputs")}))
    per = result.get("periodic")
    if per and per.get("duty_cycle_penalty"):
        b = per["back_to_back"]
        worst = max(per["periods"], key=lambda r: r.get("penalty_ratio_p50") or 0)
        rail = per.get("gpu_railgate")
        env = result.get("environment") or {}
        pinned = env.get("cpu_clocks_locked") and env.get("gpu_clocks_locked")
        out.append(_f(
            "DUTY_CYCLE_PENALTY", 0.85,
            f"Back to back an iteration takes {b['p50_ms']:.3f} ms (p50); released every "
            f"{worst['period_ms']:g} ms it takes {worst['p50_ms']:.3f} ms "
            f"({worst['penalty_ratio_p50']:.1f}x). Between inferences the board saves power and "
            f"each inference pays to wake it up.",
            ("Measure with clocks pinned (sudo jetson_clocks) and compare both periodic files: "
             "the trade-off is lower latency for higher idle power." if not pinned else
             "Clocks are already pinned: the remaining penalty is caches / GPU wake-up"
             + (" (GPU railgate is ON)" if rail else "") + "; keep the model warm with a "
             "cheaper heartbeat inference only if the latency matters more than the energy."),
            {"back_to_back_p50_ms": b["p50_ms"], "periodic_p50_ms": worst["p50_ms"],
             "period_ms": worst["period_ms"], "gpu_railgate": rail,
             "clocks_pinned": bool(pinned)}))
    s = result.get("soak")
    if s:
        for k in ("p50", "p99"):
            d = s["drift"][k]
            if d.get("drifting"):
                out.append(_f("LATENCY_DRIFT", 0.85,
                              f"{k} latency drifted {d['change_pct']:+.1f}% over the run "
                              f"({d['start']:.3f} → {d['end']:.3f} ms, significant trend).",
                              "Look at the throttling and memory rows of the same run: drift "
                              "usually follows heat or memory growth.",
                              dict(d)))
                break
        th = s["throttling"]
        if th["throttled_windows"]:
            out.append(_f("THERMAL_THROTTLING", 0.9,
                          f"Thermal throttling in {th['throttled_windows']} of {s['windows']} "
                          f"windows (peak {s.get('max_temp_c')} C). First: "
                          f"{'; '.join(th['windows'][0]['reasons'])}.",
                          "Improve cooling (heatsink, fan, airflow, enclosure), or choose a "
                          "lower power mode that the cooling can sustain.",
                          {"throttled_windows": th["throttled_windows"],
                           "max_temp_c": s.get("max_temp_c")}))
        m = s["memory"]
        if m.get("leak_suspected"):
            out.append(_f("MEMORY_LEAK", 0.9,
                          f"Process memory grows {m['process_rss_mb_per_h']:+.1f} MB/h "
                          f"(significant trend; now {m.get('rss_mb_last')} MB).",
                          "Find what accumulates per iteration (lists, caches, tensors kept "
                          "alive); `edgelens soak` again after the fix shows the slope.",
                          {"rss_mb_per_h": m["process_rss_mb_per_h"]}))
        if m.get("gpu_memory_growth_suspected"):
            out.append(_f("GPU_MEMORY_GROWTH", 0.85,
                          f"Jetson GPU memory (NvMap) grows {m['gpu_nvmap_mb_per_h']:+.1f} MB/h — "
                          f"invisible in the process RSS.",
                          "Look for CUDA/TensorRT buffers allocated per iteration and never "
                          "freed (new IOBindings, engines, contexts).",
                          {"nvmap_mb_per_h": m["gpu_nvmap_mb_per_h"]}))
        if m.get("hours_until_out_of_memory") is not None and m["hours_until_out_of_memory"] < 24:
            out.append(_f("OUT_OF_MEMORY_RISK", 0.9,
                          f"System available memory falls at {m['available_mb_per_h']:.1f} MB/h; "
                          f"at this rate it runs out in about {m['hours_until_out_of_memory']} h.",
                          "A robot that runs for a shift will be killed by the OOM killer: find "
                          "the growth before deployment.",
                          {"hours_until_out_of_memory": m["hours_until_out_of_memory"]}))
        if (m.get("swap_grew_mb") or 0) > 1:
            out.append(_f("SWAP_IN_USE", 0.7,
                          f"Swap grew by {m['swap_grew_mb']} MB during the run: memory pressure "
                          f"pushes pages to disk, which causes long latency spikes.",
                          "Reduce memory use, or disable swap (zram) for real-time workloads.",
                          {"swap_grew_mb": m["swap_grew_mb"]}))
    for f in out:
        f["evidence_strength"] = "strong" if f["rank_score"] >= 0.75 else "moderate"
    return out

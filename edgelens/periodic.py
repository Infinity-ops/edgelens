"""
edgelens.periodic
-------------------
"Your benchmark says 7 ms. At 1 Hz it's N ms."

Benchmarks run inferences back to back, so clocks are high, caches warm and
the GPU awake. A robot or an IoT node runs ONE inference per second (a
condition-monitoring LSTM, say) and idles in between — and between inferences the board saves power:
CPU/GPU clocks ramp down (DVFS), the GPU may be power-gated (Jetson
"railgate"), caches go cold. Each inference then pays a wake-up cost that a
back-to-back benchmark never shows.

`edgelens periodic` measures exactly that:
  1. idle power baseline (nothing running), before and after the rows,
  2. back-to-back latency and energy (the classic benchmark),
  3. for each period (e.g. 1000 ms): inferences released on that period —
     the SERVICE time of each (the pipeline's own work once it runs; the
     first is excluded: it follows the warm-up) kept apart from the OS
     WAKE-UP delay (release time -> loop actually running), energy
     per PERIOD including the idle time (what the battery pays) and the extra
     energy per inference above idle (what the model costs).

Evidence notes, never hidden:
  * A short inference (ms) is far below the power sampling interval (100 ms),
    so energy per inference is derived from average power over many periods,
    not observed per inference: needs >= MIN_PERIODS_FOR_ENERGY periods.
  * The trade-off of pinning clocks is shown both ways by `edgelens compare`
    on two periodic runs: lower latency, higher idle power.
"""

import glob


from .core.stats import latency_stats

PENALTY_RATIO = 1.5          # paced p50 / back-to-back p50 that counts as a penalty
PENALTY_MIN_MS = 0.5         # ... and at least this many ms more
MIN_PERIODS_FOR_ENERGY = 30
# mean power above the idle baseline that a board power sensor can resolve
IDLE_RESOLUTION_W = 0.02        # INA3221: a few mW per LSB, plus baseline drift
IDLE_RESOLUTION_REL = 0.01      # 1% of idle power
# Back-to-back power is the ceiling for the periodic rows (below), so it
# needs a real power average: at 10 samples/s, 2 s is 20 samples.
B2B_MIN_POWER_S = 2.0

_RAILGATE_GLOBS = ("/sys/devices/gpu.0/railgate_enable",
                   "/sys/devices/platform/gpu.0/railgate_enable",
                   "/sys/devices/platform/bus@0/*.gpu/railgate_enable",
                   "/sys/devices/platform/*.gpu/railgate_enable",
                   "/sys/devices/*.gpu/railgate_enable")


def gpu_railgate():
    """True/False when the Jetson GPU may power off between inferences."""
    for pattern in _RAILGATE_GLOBS:
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as f:
                    return f.read().strip() == "1"
            except OSError:
                continue
    return None


def _energy_row(result, period_ms, idle_w, n, idle_drift_w=None, ceiling_w=None):
    """Energy per period (measured) and extra energy per inference above idle
    (derived: (mean power - idle) x period).

    The derived figure is only as good as the idle baseline. AGX Orin, 1 Hz,
    a 0.3 ms LSTM: three runs on a quiet board gave 57, 114 and 376 mJ "per
    inference"; 376 mJ in a 4 ms inference would be ~100 W on a board that
    draws 4-9 W. Mean power had moved with the board, not with the model.
    So two checks, both from this run's own measurements:
      * idle_drift_w: idle measured before AND after the rows; an excess
        smaller than that drift (or the sensor resolution) is only a bound;
      * ceiling_w: mean power of the same work run back to back (100% duty
        cycle). A periodic row cannot draw more on average than running the
        work continuously; if it does, the excess is not this model's.
    """
    en = result.get("energy") or {}
    if not en.get("available"):
        return {"energy_per_period_mj": None, "power_w_mean": None,
                "dynamic_energy_per_inference_mj": None}
    # mean power x period (W x ms = mJ). Not energy / releases: the run has
    # one more release than full periods, which would bias it low by 1/(n+1).
    pw = en.get("power_w_mean")
    row = {"power_w_mean": pw,
           "energy_per_period_mj": round(pw * period_ms, 3) if pw is not None else None}
    row["dynamic_energy_per_inference_mj"] = None
    if idle_w is not None and pw is not None:
        # Fourth Nano run: at 1 Hz a 9 ms inference raised mean power by less
        # than the sensor resolves, and "max(0, P - idle)" printed a confident
        # 0.0 mJ. Below the resolution there is no measurement, only a bound.
        res_w = max(IDLE_RESOLUTION_W, IDLE_RESOLUTION_REL * idle_w)
        floor_w = max(res_w, idle_drift_w or 0.0)
        if ceiling_w is not None and pw > ceiling_w + res_w:
            row["dynamic_energy_not_attributable"] = (
                f"mean power {pw:.3f} W is above the {ceiling_w:.3f} W the board drew running "
                f"the same work back to back: the excess over idle is not this model's (idle "
                f"baseline drift or other activity on the board)")
        elif pw - idle_w < floor_w:
            row["dynamic_energy_below_mj"] = round(floor_w * period_ms, 1)
            row["dynamic_energy_below_reason"] = (
                "sensor resolution" if floor_w == res_w else
                f"idle baseline drift ({idle_drift_w:.3f} W between the idle measurements "
                f"before and after)")
        else:
            row["dynamic_energy_per_inference_mj"] = round((pw - idle_w) * period_ms, 3)
    if n < MIN_PERIODS_FOR_ENERGY or en.get("low_confidence"):
        row["energy_low_confidence"] = (
            f"only {n} periods: energy per period/inference is an average over the run and "
            f"needs >= {MIN_PERIODS_FOR_ENERGY} periods to be trustworthy")
    return row


def run_periodic(pipe, periods_ms, iterations=30, warm_iterations=200, idle_baseline_s=10.0,
                 deadline_ms=None, telemetry_interval_s=0.1, pipeline_source=None,
                 model_path=None, progress=None):
    """Returns the result document of the LONGEST period (the real rate; so
    validate / compare work on it) with a `periodic` block for every period."""
    from .core.harness import measure_idle_power, run_harness
    from .core.identity import attach_identity
    from .hardware import power as power_mod

    say = progress or (lambda m: None)
    reader = power_mod.PowerReader()
    has_power = reader.available() and bool(reader.read())
    idle_w = None
    if has_power and idle_baseline_s:
        say(f"Measuring idle power for {idle_baseline_s:g} s (nothing running)...")
        idle_w = measure_idle_power(idle_baseline_s, reader)

    say(f"Back-to-back: {warm_iterations} iterations...")
    b2b = run_harness(pipe, source=getattr(pipe, "source", None),
                              iterations=warm_iterations, warmup=10,
                      telemetry_interval_s=telemetry_interval_s, attach_ids=False,
                      pipeline_source=pipeline_source)
    b2b_lat = b2b["latency"]
    if has_power and b2b_lat.get("mean"):
        # 200 x 0.3 ms is 60 ms: less than one power sample. Run long enough
        # for a power average (it is the ceiling of every periodic row).
        need = int(B2B_MIN_POWER_S * 1000.0 / b2b_lat["mean"]) + 1
        if need > warm_iterations:
            say(f"Back-to-back: {need} iterations for a {B2B_MIN_POWER_S:g} s power average...")
            b2b = run_harness(pipe, source=getattr(pipe, "source", None),
                              iterations=min(need, 1_000_000), warmup=10,
                              telemetry_interval_s=telemetry_interval_s, attach_ids=False,
                              pipeline_source=pipeline_source,
                              max_seconds=3 * B2B_MIN_POWER_S)
            b2b_lat = b2b["latency"]
    b2b_inference_ms = b2b_lat.get("p50")

    rows, results, steady_counts = [], [], []
    for period in periods_ms:
        say(f"Periodic: {iterations} inferences every {period:g} ms "
            f"(~{iterations * period / 1000:.0f} s)...")
        res = run_harness(pipe, source=getattr(pipe, "source", None),
                              iterations=iterations + 1, warmup=3, period_ms=period,
                          pace=True, deadline_ms=deadline_ms,
                          telemetry_interval_s=telemetry_interval_s, attach_ids=False,
                          pipeline_source=pipeline_source)
        # Two different costs, kept apart:
        #   service   = the pipeline's own work when it starts after idle
        #               (cold clocks/caches/GPU): the duty-cycle penalty
        #   wake-up   = how late the OS woke the loop after the release time
        resp = res.get("latency_series_ms") or []
        serv = res.get("service_series_ms") or resp
        steady = serv[1:] if len(serv) > 1 else serv           # #0 follows the warm-up
        wake = [r - s for r, s in zip(resp[1:], serv[1:])]
        st = latency_stats(steady)
        wk = latency_stats(wake) if wake else {}
        rt = latency_stats(resp[1:]) if len(resp) > 1 else {}
        row = {"period_ms": period, "inferences": len(steady),
               "p50_ms": st.get("p50"), "p99_ms": st.get("p99"), "max_ms": st.get("max"),
               "mean_ms": st.get("mean"),
               "wake_delay_p50_ms": wk.get("p50"), "wake_delay_max_ms": wk.get("max"),
               "response_p99_ms": rt.get("p99"), "response_max_ms": rt.get("max"),
               "first_after_warmup_ms": round(serv[0], 3) if serv else None}
        if b2b_inference_ms and st.get("p50") is not None:
            row["penalty_ratio_p50"] = round(st["p50"] / b2b_inference_ms, 3)
            row["penalty_ms_p50"] = round(st["p50"] - b2b_inference_ms, 3)
        rows.append(row)
        results.append(res)
        steady_counts.append(len(steady))

    # Idle again after the rows: the drift between the two is the uncertainty
    # of every "extra above idle" figure; their mean is the baseline.
    idle_before, idle_after, idle_drift = idle_w, None, None
    if idle_w is not None:
        say(f"Measuring idle power again for {idle_baseline_s:g} s...")
        idle_after = measure_idle_power(idle_baseline_s, reader)
        if idle_after is not None:
            idle_drift = abs(idle_after - idle_before)
            idle_w = (idle_before + idle_after) / 2.0
    b2b_en = b2b.get("energy") or {}
    ceiling_w = (b2b_en.get("power_w_mean")
                 if b2b_en.get("available") and not b2b_en.get("low_confidence") else None)
    for row, res, n in zip(rows, results, steady_counts):
        row.update(_energy_row(res, row["period_ms"], idle_w, n, idle_drift_w=idle_drift,
                               ceiling_w=ceiling_w))

    # The LONGEST period is the reference ("primary"): the real, slowest rate
    # (e.g. 1000 ms) — compare / validate / the trade-off use it.
    primary_idx = max(range(len(rows)), key=lambda k: rows[k]["period_ms"])
    primary_result = results[primary_idx]
    sample_ms = telemetry_interval_s * 1000.0
    notes = []
    if has_power and b2b_inference_ms is not None and b2b_inference_ms < sample_ms:
        notes.append(f"An inference ({b2b_inference_ms:.3f} ms) is shorter than the power "
                     f"sampling interval ({sample_ms:.0f} ms): energy per inference is derived "
                     f"from average power over the periods, not observed per inference "
                     f"(moderate evidence; >= {MIN_PERIODS_FOR_ENERGY} periods).")
    bad = [r for r in rows if r.get("dynamic_energy_not_attributable")]
    if bad:
        which = ", ".join(f"every {r['period_ms']:g} ms" for r in bad)
        notes.append(f"Extra energy per inference not reported ({which}): "
                     f"{bad[0]['dynamic_energy_not_attributable']}. Energy per period is "
                     f"still measured.")
    if not has_power:
        notes.append("No readable power sensor: latency only (Jetson: see `edgelens doctor`).")
    railgate = gpu_railgate()
    penalized = [r for r in rows if (r.get("penalty_ratio_p50") or 0) >= PENALTY_RATIO
                 and (r.get("penalty_ms_p50") or 0) >= PENALTY_MIN_MS]
    primary_result["periodic"] = {
        "back_to_back": {"iterations": b2b["iterations"], "p50_ms": b2b_lat.get("p50"),
                         "p99_ms": b2b_lat.get("p99"), "mean_ms": b2b_lat.get("mean"),
                         "energy_per_inference_mj": (
                             round(b2b["energy"]["energy_per_iteration_j"] * 1000.0, 3)
                             if (b2b.get("energy") or {}).get("available") else None),
                         "power_w_mean": b2b_en.get("power_w_mean"),
                         "power_sample_count": b2b_en.get("power_sample_count")},
        "idle_power_w": round(idle_w, 3) if idle_w is not None else None,
        "idle_power_before_w": round(idle_before, 3) if idle_before is not None else None,
        "idle_power_after_w": round(idle_after, 3) if idle_after is not None else None,
        "idle_drift_w": round(idle_drift, 3) if idle_drift is not None else None,
        "periods": rows,
        "primary": rows[primary_idx],
        "duty_cycle_penalty": bool(penalized),
        "gpu_railgate": railgate,
        "power_sample_interval_ms": sample_ms,
        "notes": notes,
    }
    attach_identity(primary_result, model_path=model_path)
    return primary_result


def tradeoff(before, after):
    """'pinning cuts p99 from X to Y ms but costs Z mJ per period' — both
    sides of a change, from two periodic results. None if not comparable."""
    pb = (before.get("periodic") or {}).get("primary") or {}
    pa = (after.get("periodic") or {}).get("primary") or {}
    if not pb or not pa or pb.get("period_ms") != pa.get("period_ms"):
        return None
    out = {"period_ms": pb["period_ms"],
           "p99_ms": [pb.get("p99_ms"), pa.get("p99_ms")],
           "p50_ms": [pb.get("p50_ms"), pa.get("p50_ms")],
           "energy_per_period_mj": [pb.get("energy_per_period_mj"),
                                    pa.get("energy_per_period_mj")],
           "power_w_mean": [pb.get("power_w_mean"), pa.get("power_w_mean")]}
    eb, ea = out["energy_per_period_mj"]
    if eb is not None and ea is not None:
        out["energy_delta_mj_per_period"] = round(ea - eb, 3)
        out["energy_delta_mj_per_s"] = round((ea - eb) * 1000.0 / pb["period_ms"], 3)
    lb, la = out["p99_ms"] if None not in out["p99_ms"] else out["p50_ms"]
    out["latency_basis"] = "p99" if None not in out["p99_ms"] else "p50"
    if lb is not None and la is not None:
        out["latency_delta_ms"] = round(la - lb, 3)
    return out


def tradeoff_sentence(t, change="this change"):
    if not t or t.get("latency_delta_ms") is None:
        return None
    lb, la = t[f"{t['latency_basis']}_ms"]
    s = (f"At one inference every {t['period_ms']:g} ms, {change} moves "
         f"{t['latency_basis']} latency from {lb:.3f} to {la:.3f} ms")
    if t.get("energy_delta_mj_per_period") is not None:
        d = t["energy_delta_mj_per_period"]
        verb = "costs" if d > 0 else "saves"
        s += (f" and {verb} {abs(d):.1f} mJ per period ({abs(t['energy_delta_mj_per_s']):.1f} "
              f"mJ per second, idle included)")
    else:
        s += " (energy not measured: no power sensor)"
    return s + "."

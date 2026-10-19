"""
edgelens.stream.findings
--------------------------
Diagnosis rules for windowed time-series pipelines (results of `edgelens
replay`, or any result whose pipeline config states `trained_step_ms`).

    CADENCE_MISMATCH        the model is called at another step than it was
                            trained with, or the data arrive at another rate
    DECISION_LATENCY        the model is fast, the decision is not: where the
                            seconds between event and alarm come from
    WARMUP_PADDING          the first window is padding, not data
    OUT_OF_ENVELOPE         the model answers about inputs it never saw
    DATA_GAPS               the recording stalls; windows span the gap
    DEAD_INPUTS             inputs that never varied in training
"""

CADENCE_TOLERANCE = 0.10       # 10 % between trained step and real step
ENVELOPE_MIN_PCT = 99.0


def _f(type_, score, detail, rec, evidence):
    return {"type": type_, "rank_score": round(score, 2), "detail": detail,
            "recommendation": rec, "evidence": evidence}


def cadence(result):
    """(trained_step_ms, real_step_ms, how) or None."""
    cfg = (result.get("pipeline") or {}).get("config") or {}
    trained = cfg.get("trained_step_ms")
    if not trained:
        return None
    real = cfg.get("deployment_period_ms")
    how = "deployment period in the spec"
    if not real and not result.get("stream"):
        # an application recorded with --period-ms (FORTIS, a ROS node)
        real = (result.get("requirements") or {}).get("period_ms")
        how = "the run's period"
    if not real:
        return None
    return trained, float(real), how


def findings(result):
    out = []
    st = result.get("stream") or {}
    dec = result.get("decision") or {}
    if dec.get("aggregate") and dec.get("recordings"):
        # an `edgelens detect` result: explain the worst recording
        def worst(s):
            o = s.get("output") or {}
            return (1 if o.get("missed") else 0, o.get("time_to_detect_s") or -1e18)
        dec = max(dec["recordings"], key=worst)
    audit = st.get("audit") or {}

    c = cadence(result)
    if c and abs(c[1] - c[0]) > CADENCE_TOLERANCE * c[0]:
        trained, real, how = c
        cfg = (result.get("pipeline") or {}).get("config") or {}
        seq = (cfg.get("window") or [None])[0]
        hist = (f" A {seq}-step window then covers {seq * real / 1000:g} s instead of "
                f"{seq * trained / 1000:g} s," if seq else "")
        out.append(_f(
            "CADENCE_MISMATCH", 0.85,
            f"The model was trained with one step every {trained:g} ms, but runs every "
            f"{real:g} ms ({how}).{hist} and every per-step constant (slew limit, smoothing "
            f"length) changes its meaning in seconds. The model's accuracy on the training "
            f"cadence says nothing about this one.",
            "Run the model on the trained step (buffer the input and call it every "
            f"{trained:g} ms), or retrain with blocks of {real:g} ms of data; then replay "
            f"with the deployed period to measure the difference.",
            {"trained_step_ms": trained, "deployed_step_ms": real, "source": how}))
    if audit.get("rate_mismatch"):
        out.append(_f(
            "CADENCE_MISMATCH", 0.8,
            f"The recording arrives at {audit.get('rate_hz_measured')} Hz; the pipeline "
            f"assumes {audit.get('rate_hz_expected')} Hz, so each block holds a different "
            f"span of time than in training.",
            "Resample to the trained rate before the window is formed, or fix the merge "
            "that produces the rows.",
            {"rate_hz_measured": audit.get("rate_hz_measured"),
             "rate_hz_expected": audit.get("rate_hz_expected")}))

    o = dec.get("output") or {}
    b = dec.get("budget") or {}
    ttd = o.get("time_to_detect_s")
    if ttd is not None and b.get("compute_p99_ms") is not None:
        raw_ttd = (dec.get("raw") or {}).get("time_to_detect_s")
        parts = []
        if raw_ttd is not None:
            parts.append(f"the raw model output crosses the threshold {raw_ttd:g} s after the "
                         f"event (the model's own response)")
        if b.get("post_processing_cost_s") is not None:
            parts.append(f"post-processing adds {b['post_processing_cost_s']:+g} s")
        built = [f"one block {b['step_s']:g} s"]
        if b.get("slew_floor_s"):
            built.append(f"the slew limit needs >= {b['slew_floor_s']:g} s to move from "
                         f"{b.get('slew_from_level')} to the threshold")
        if b.get("mean_delay_s"):
            built.append(f"the output mean delays ~{b['mean_delay_s']:g} s")
        out.append(_f(
            "DECISION_LATENCY", 0.9 if ttd > 1.0 else 0.5,
            f"A step is computed in {b['compute_p99_ms']:g} ms (p99), but the alarm comes "
            f"{ttd:g} s after the event: " + "; ".join(parts) + ". Built into the pipeline: "
            + ", ".join(built) + ".",
            "Judge the system on time to detect, not on inference latency. Tune the "
            "post-processing on validation data (`edgelens detect --mean-steps/--slew-per-step "
            "--thresholds`, recomputed without re-running the model); a slow raw response is "
            "the model's: more training data near the threshold, or a stateful (streaming) "
            "model that updates every step.",
            dict({"time_to_detect_s": ttd, "raw_time_to_detect_s": raw_ttd}, **b)))

    if b.get("warmup_s") and (dec.get("guard") or {}).get("warming_up"):
        out.append(_f(
            "WARMUP_PADDING", 0.55,
            f"For the first {b['warmup_s']:g} s the window is padded with zeros (in scaled "
            f"space: the training centre), so the model reports a 'typical' state whatever "
            f"the robot does. A fault present at start-up is invisible for that long.",
            "Treat warm-up outputs as 'no decision' (the guard marks them warming_up), or "
            "start the window from real data held over from the previous run.",
            {"warmup_s": b["warmup_s"], "steps": dec["guard"]["warming_up"]}))

    env = dec.get("in_envelope_pct")
    if env is not None and env < ENVELOPE_MIN_PCT:
        g = dec.get("guard") or {}
        out.append(_f(
            "OUT_OF_ENVELOPE", 0.75 if env < 90 else 0.6,
            f"Only {env:g}% of the predictions saw inputs inside the training envelope "
            f"(degraded {g.get('degraded', 0)}, invalid {g.get('invalid', 0)} steps). The "
            f"model's test accuracy does not cover the rest.",
            "`edgelens envelope check` names the inputs that leave it; position or map "
            "features are a frequent cause (they describe WHERE the robot is, not its "
            "health).",
            {"in_envelope_pct": env, "guard": g}))

    if audit.get("gaps"):
        gaps = audit["gaps"]
        out.append(_f(
            "DATA_GAPS", 0.6,
            f"The recording has {len(gaps)} gap(s), {audit.get('gap_time_s')} s in total "
            f"(longest {max(g['length_s'] for g in gaps):g} s): a block that spans a gap "
            f"mixes data from before and after it.",
            "Split the recording at gaps (or mark them invalid) before training and replay.",
            {"gaps": gaps[:10]}))
    dead = audit.get("dead_columns") or []
    ncols = ((result.get("pipeline") or {}).get("config") or {}).get("n_columns") or "the"
    if dead:
        out.append(_f(
            "DEAD_INPUTS", 0.45,
            f"{len(dead)} of {ncols} "
            f"input columns never vary in the training data ({', '.join(dead[:5])}"
            f"{'...' if len(dead) > 5 else ''}): they carry no information, and any change at "
            f"run time is an input the model never saw.",
            "Drop them from the model inputs (smaller, faster model) or record data where "
            "they vary.",
            {"dead_columns": dead}))
    return out

"""
edgelens.core.integrity
-------------------------
"Can these numbers be trusted?" — checks on the measurement itself, before
anyone reads the measurement. Used by `validate` (an issue that BLOCKS turns
a PASS into NOT_MEASURED; a FAIL stays a FAIL, because what was observed
really happened) and by `diagnose` (listed next to the findings).

  MARKER_EVENTS_LOST      spans produced but never received: hidden misses
                          are possible, so a latency PASS can't be confirmed
  GPU_TIMING_NOT_SYNCED   inference spans ended without GPU sync while the
                          GPU was busy: they may measure the enqueue only
  LATENCY_NOT_MEASURED    attach mode without markers: resources only
  TIMING_FROM_LOG         the application timed itself; EdgeLens parsed it
  CLOCK_NOT_MONOTONIC     harness timestamps not on CLOCK_MONOTONIC (non-Linux)
  TRACE_TRUNCATED         trace capped; statistics still use every sample
  CROSS_MACHINE_CLOCK_UNKNOWN  ROS data age across machines without clock bounds
  NEGATIVE_DATA_AGE       messages "arrive before they were stamped": clocks disagree
  DUPLICATE_SEQUENCE      two senders shared an id: loss counts unreliable
  SENDER_NOT_FINISHED_CLEANLY  no final message (killed): tail losses unseen
  SPANS_NOT_STORED        spans beyond the collector's storage limit
  ITERATIONS_AMBIGUOUS    several threads/processes without @iteration spans
  ROS_MESSAGES_MISSING    unmatched outputs / topic gaps in chain latency
  FOREIGN_SPANS_EXCLUDED  other processes' spans on a shared socket, dropped
"""

BLOCKS, WARN, INFO = "blocks_pass", "warn", "info"

# Requirements whose PASS depends on having seen every iteration.
LATENCY_REQUIREMENTS = ("p50_ms", "p95_ms", "p99_ms", "p99_9_ms", "max_latency_ms",
                        "max_miss_ratio", "min_fps")

GPU_BUSY_PCT = 5.0


def _issue(code, severity, detail, affects=()):
    return {"code": code, "severity": severity, "detail": detail, "affects": list(affects)}


def check(result):
    issues = []
    m = result.get("markers") or {}
    if m:
        lost = m.get("lost") or 0
        if lost:
            issues.append(_issue(
                "MARKER_EVENTS_LOST", BLOCKS,
                f"{lost} marker span(s) were lost ({(m.get('lost_ratio') or 0) * 100:.3f}%). "
                f"A lost span may have been a slow one, so a latency PASS cannot be "
                f"confirmed; a FAIL still stands.", LATENCY_REQUIREMENTS))
        if m.get("duplicate_seq"):
            issues.append(_issue(
                "DUPLICATE_SEQUENCE", BLOCKS,
                f"{m['duplicate_seq']} span(s) reused another span's sequence number (two "
                f"senders shared one id, e.g. after fork() in an old client). Loss counts "
                f"are unreliable, so a latency PASS can't be confirmed.", LATENCY_REQUIREMENTS))
        unfinished = [x for x in m.get("senders") or [] if not x.get("final_seen")]
        if unfinished:
            obs = result.get("observation") or {}
            target = obs.get("target") or {}
            still_running = (target.get("kind") == "pid" and
                             obs.get("ended_by") in ("duration reached", "interrupted (Ctrl-C)"))
            pressured = [x for x in unfinished if x.get("backpressure")]
            # Spans can sit in a sender's pending batch without the collector
            # ever seeing a batch, so "no backpressure seen" proves nothing
            # once the process is gone. Only a target that is still running
            # (watch --pid with --duration) is merely "not finished yet".
            blocks = bool(pressured) or not still_running
            issues.append(_issue(
                "SENDER_NOT_FINISHED_CLEANLY", BLOCKS if blocks else WARN,
                f"{len(unfinished)} sender(s) (pid {', '.join(str(x['pid']) for x in unfinished)}) "
                f"sent no final message"
                + (" — the target was still running when the observation ended; spans "
                   "after that point are simply not part of this run."
                   if not blocks else
                   " (killed by a signal or crashed): spans still pending inside the process "
                   "at that moment can't be counted, so a latency PASS can't be confirmed. "
                   "Let the application exit normally (handle SIGINT by leaving its loop "
                   "and returning from main)."),
                LATENCY_REQUIREMENTS))
        unverified = m.get("foreign_unverified") or 0
        if unverified:
            issues.append(_issue(
                "FOREIGN_SPANS_UNVERIFIED", BLOCKS,
                f"{unverified} span(s) came from processes that had already exited when "
                f"EdgeLens checked whether they belong to the target; they were excluded, "
                f"but may have been the target's own short-lived workers.",
                LATENCY_REQUIREMENTS))
        if m.get("not_stored"):
            issues.append(_issue(
                "SPANS_NOT_STORED", BLOCKS,
                f"{m['not_stored']} span(s) arrived after the collector's storage limit and are "
                f"missing from the statistics.", LATENCY_REQUIREMENTS))
        if (m.get("iteration_streams") or 0) > 1 and \
                str(m.get("iterations_from", "")).startswith("name repetition"):
            issues.append(_issue(
                "ITERATIONS_AMBIGUOUS", BLOCKS,
                f"Spans came from {m['iteration_streams']} threads/processes and no "
                f"@iteration spans mark what one end-to-end iteration is, so each stream was "
                f"cut into iterations separately. Stage timings are fine; end-to-end latency "
                f"and deadline misses are not confirmed. Add EL_ITERATION() / "
                f"mark.iteration() around one full cycle.", LATENCY_REQUIREMENTS))
        if m.get("foreign_spans"):
            issues.append(_issue(
                "FOREIGN_SPANS_EXCLUDED", INFO,
                f"{m['foreign_spans']} span(s) from other processes (pid "
                f"{', '.join(map(str, m.get('foreign_pids') or []))}) on the shared socket "
                f"were excluded from this run."))
        if m.get("source") == "log":
            issues.append(_issue(
                "TIMING_FROM_LOG", WARN,
                "Latency was measured by the application itself and parsed from its log "
                f"(timestamps: {m.get('timestamps')}); no loss accounting.",
                LATENCY_REQUIREMENTS))
        g = m.get("gpu_timing") or {}
        gpu = (result.get("telemetry") or {}).get("gpu_percent_mean")
        if g.get("unsynced_inference_spans") and gpu is None:
            issues.append(_issue(
                "GPU_TIMING_NOT_SYNCED", WARN,
                f"{g['unsynced_inference_spans']} inference span(s) ended without GPU "
                f"synchronisation and GPU load is unknown on this host: if they launch "
                f"asynchronous CUDA/TensorRT work they measure the enqueue only.",
                ("p50_ms", "p95_ms", "p99_ms", "p99_9_ms", "max_latency_ms")))
        elif g.get("unsynced_inference_spans") and gpu > GPU_BUSY_PCT:
            issues.append(_issue(
                "GPU_TIMING_NOT_SYNCED", WARN,
                f"{g['unsynced_inference_spans']} inference span(s) "
                f"({', '.join(g.get('unsynced_inference_stages') or [])}) ended without GPU "
                f"synchronisation while the GPU was {gpu:.0f}% busy: if they launch "
                f"asynchronous CUDA/TensorRT work they measure the enqueue, not the "
                f"inference.", ("p50_ms", "p95_ms", "p99_ms", "p99_9_ms", "max_latency_ms")))
    r2 = result.get("ros2") or {}
    if r2.get("latency_basis") == "chain":
        ch = r2.get("chain") or {}
        missing = sum(((r2.get(k) or {}).get("estimated_missing_messages") or 0)
                      for k in ("from", "to"))
        if ch.get("unmatched") or missing:
            issues.append(_issue(
                "ROS_MESSAGES_MISSING", BLOCKS,
                f"{ch.get('unmatched', 0)} output message(s) had no matching input and about "
                f"{missing} message(s) are missing from the topics' rate: those predictions "
                f"may have been the slow ones, so a latency PASS can't be confirmed.",
                LATENCY_REQUIREMENTS))
    if r2.get("latency_basis") == "data_age":
        age = r2.get("data_age") or {}
        if not age.get("valid"):
            issues.append(_issue(
                "CROSS_MACHINE_CLOCK_UNKNOWN", BLOCKS,
                f"Latency here is data age (arrival - header.stamp), which compares two "
                f"machines' clocks: {age.get('uncertainty_basis')}. A PASS can't be "
                f"confirmed; use chain latency (output copies the input stamp) or give the "
                f"clock bounds.", LATENCY_REQUIREMENTS))
        elif age.get("uncertainty_ms"):
            issues.append(_issue(
                "CLOCK_UNCERTAINTY", WARN,
                f"Data age carries ±{age['uncertainty_ms']} ms clock uncertainty "
                f"({age.get('uncertainty_basis')}).", LATENCY_REQUIREMENTS))
        neg = ((age.get("at_output_ms") or {}).get("p50") or 0) < 0 or \
            ((age.get("from_ms") or {}).get("p50") or 0) < 0
        if neg:
            issues.append(_issue(
                "NEGATIVE_DATA_AGE", BLOCKS,
                "Messages arrive 'before' they were stamped: the clocks of the sensor machine "
                "and this machine disagree. Fix time sync (chrony) before trusting data age.",
                LATENCY_REQUIREMENTS))
    if result.get("mode") in ("attach", "markers") and not result.get("iterations"):
        issues.append(_issue(
            "LATENCY_NOT_MEASURED", INFO,
            "No marker spans: this run measured resources only (attach mode). Add "
            "edgelens_mark.h / edgelens.mark spans for per-inference latency.",
            LATENCY_REQUIREMENTS))
    c = result.get("clock") or {}
    if c and c.get("perf_counter_is_monotonic") is False:
        issues.append(_issue(
            "CLOCK_NOT_MONOTONIC", INFO,
            "This host's perf_counter is not CLOCK_MONOTONIC (non-Linux): stage timings "
            "are fine on their own but don't line up with markers from other processes."))
    if (result.get("trace") or {}).get("truncated"):
        issues.append(_issue(
            "TRACE_TRUNCATED", INFO,
            "The event trace was capped; statistics still use every sample, but the "
            "timeline is incomplete."))
    return issues


def blocking(issues, requirement_key):
    """The first issue that blocks a PASS of this requirement, or None."""
    for i in issues:
        if i["severity"] == BLOCKS and requirement_key in i["affects"]:
            return i
    return None

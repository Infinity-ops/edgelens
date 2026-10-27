"""
edgelens.timeline
-------------------
`edgelens export --perfetto`: one timeline from one or more result files
(v0.2 Layer 3, E2). Opens in https://ui.perfetto.dev (and chrome://tracing).

What ends up on the timeline:

* stage / marker spans, per process and thread (several processes from one
  `edgelens record`, or several result files recorded at the same time),
* end-to-end iterations on their own track, deadline misses as markers,
* board counters: CPU / GPU load, GPU clock, EMC, memory, max temperature,
  board power and every power rail,
* per-process counters of `watch` / `record` runs (CPU % of one core, RSS).

Format: the Chrome JSON trace format, which Perfetto imports natively — no
protobuf dependency. Every timestamp in EdgeLens is CLOCK_MONOTONIC, so
files recorded on the same boot line up exactly; files from different boots
(or machines) are aligned at their start instead, and the trace says so.
"""

import json

ITERATION = "@iteration"
BOARD_PID_BASE = 2_000_000_000          # synthetic pids, far above Linux pid_max
PIPE_PID_BASE = 2_000_100_000
SAME_BOOT_TOLERANCE_NS = 2_000_000_000  # wall-minus-mono differs by NTP corrections only

# telemetry_series field -> (counter name, series name). Perfetto names a
# JSON counter track "<name> <arg>", so this gives "CPU load %", "GPU clock MHz".
COUNTERS = (
    ("cpu_percent", "CPU", "load %"),
    ("gpu_percent", "GPU", "load %"),
    ("gpu_freq_mhz", "GPU", "clock MHz"),
    ("emc_percent", "EMC", "load %"),
    ("mem_percent", "Memory", "used %"),
    ("max_temp_c", "Temperature", "max C"),
    ("power_w", "Board", "power W"),
)
PROCESS_COUNTERS = (("cpu_percent", "CPU % of one core"), ("rss_mb", "RSS MB"),
                    ("threads", "threads"))


class TimelineError(ValueError):
    pass


def _origin(result):
    clk = result.get("clock") or {}
    tel = result.get("telemetry") or {}
    return clk.get("trace_origin_mono_ns") or tel.get("origin_mono_ns")


def _wall_offset(result):
    return ((result.get("clock") or {}).get("wall") or {}).get("wall_minus_mono_ns")


def _first_abs(result):
    """Earliest absolute CLOCK_MONOTONIC timestamp in the result."""
    cands = []
    o = _origin(result)
    events = (result.get("trace") or {}).get("events") or []
    if o is not None and events:
        cands.append(o + min(e["start_ns"] for e in events))
    for s in result.get("telemetry_series") or []:
        if s.get("t_mono_ns") is not None:
            cands.append(s["t_mono_ns"])
            break
    for s in result.get("process_series") or []:
        if s.get("t_mono_ns") is not None:
            cands.append(s["t_mono_ns"])
            break
    return min(cands) if cands else None


GAP_NOTE_NS = 60 * 10 ** 9


def _length_ns(result):
    """Rough length of a result on the timeline (spans or telemetry)."""
    ends = []
    o = _origin(result)
    ev = (result.get("trace") or {}).get("events") or []
    if o is not None and ev:
        ends.append(max(o + e["start_ns"] + e["dur_ns"] for e in ev))
    ser = [s["t_mono_ns"] for s in result.get("telemetry_series") or [] if s.get("t_mono_ns")]
    if ser:
        ends.append(max(ser))
    first = _first_abs(result)
    return (max(ends) - first) if ends and first is not None else 0


def _alignment(items, align):
    """'mono' when every file comes from the same boot, else 'start'."""
    if align in ("mono", "start"):
        return align, None
    offs = [_wall_offset(r) for _, r in items]
    if len(items) == 1:
        return "mono", None
    if any(o is None for o in offs):
        return "start", ("some files carry no clock information (older EdgeLens): each "
                         "file is aligned at its own start")
    if max(offs) - min(offs) > SAME_BOOT_TOLERANCE_NS:
        return "start", ("the files come from different boots or machines: their "
                         "monotonic clocks can't be compared, so each file is aligned at "
                         "its own start")
    return "mono", None


def build(items, align="auto"):
    """items: [(label, result_dict)] -> (trace dict, notes)."""
    if not items:
        raise TimelineError("No result files given.")
    mode, note = _alignment(items, align)
    notes = [note] if note else []
    firsts = [_first_abs(r) for _, r in items]
    if all(f is None for f in firsts):
        raise TimelineError("Nothing to put on a timeline: the files have no spans and no "
                            "telemetry series (summary-only results, e.g. a soak, keep "
                            "per-window data in their .jsonl instead).")
    base_all = min(f for f in firsts if f is not None)
    if mode == "mono" and len(items) > 1:
        spans = sorted((f, f + _length_ns(r)) for f, (_, r) in zip(firsts, items)
                       if f is not None)
        gap = max((b0 - a1 for (_, a1), (b0, _) in zip(spans, spans[1:])), default=0)
        if gap > GAP_NOTE_NS:
            notes.append(f"the files were recorded {gap / 1e9:.0f} s apart (same boot, real "
                         f"time kept); --align start puts them side by side")

    ev = []
    sort_index = [0]

    def meta(kind, pid, name, tid=None):
        e = {"ph": "M", "name": kind, "pid": pid, "args": {"name": name}}
        if tid is not None:
            e["tid"] = tid
        ev.append(e)

    def sort(pid):
        sort_index[0] += 1
        ev.append({"ph": "M", "name": "process_sort_index", "pid": pid,
                   "args": {"sort_index": sort_index[0]}})

    seen_pids = set()
    for i, (label, r) in enumerate(items):
        first = firsts[i]
        if first is None:
            notes.append(f"{label}: no spans and no telemetry series; skipped")
            continue
        base = base_all if mode == "mono" else first

        def us(abs_ns, base=base):
            return round((abs_ns - base) / 1000.0, 3)

        origin = _origin(r) or first
        pname = (r.get("pipeline") or {}).get("name") or "pipeline"
        events = (r.get("trace") or {}).get("events") or []
        if (r.get("trace") or {}).get("truncated"):
            notes.append(f"{label}: the result kept only the first {len(events)} spans "
                         f"(the rest are in the statistics, not on the timeline)")

        # -------- spans: real pids from markers; one synthetic process otherwise
        tids = {}

        def tid_of(pid, raw):
            key = (pid, raw)
            if key not in tids:
                # real Linux tids are small; Python thread idents are not
                tids[key] = raw if isinstance(raw, int) and 0 < raw < 2 ** 31 \
                    else len([k for k in tids if k[0] == pid]) + 1
            return tids[key]

        synth = PIPE_PID_BASE + i
        iter_tid = 2 ** 31 - 1 - i          # own track; distinct per file (tids are global)
        by_iter = {}
        pids_here = []
        for e in events:
            pid = e.get("pid") or synth
            if pid not in pids_here:
                pids_here.append(pid)
            start = origin + e["start_ns"]
            if e["stage"] == ITERATION:
                ev.append({"ph": "X", "name": "iteration", "cat": "iteration", "pid": pid,
                           "tid": iter_tid, "ts": us(start), "dur": round(e["dur_ns"] / 1000.0, 3),
                           "args": {"latency_ms": round(e["dur_ns"] / 1e6, 4)}})
                continue
            args = {"iteration": e.get("iter"), "ms": round(e["dur_ns"] / 1e6, 4)}
            if e.get("flags"):
                args["gpu_synced"] = "g" in e["flags"]
            ev.append({"ph": "X", "name": e["stage"], "cat": "stage", "pid": pid,
                       "tid": tid_of(pid, e.get("thread") or 0), "ts": us(start),
                       "dur": round(e["dur_ns"] / 1000.0, 3), "args": args})
            if e.get("iter") is not None and not e.get("pid"):
                lo, hi = by_iter.get(e["iter"], (start, start + e["dur_ns"]))
                by_iter[e["iter"]] = (min(lo, start), max(hi, start + e["dur_ns"]))

        # end-to-end iterations for harness runs (markers carry @iteration spans)
        lat = r.get("latency_series_ms") or []
        dl = (r.get("deadline") or {}).get("deadline_ms")
        for it, (lo, hi) in sorted(by_iter.items()):
            e2e = lat[it] if it < len(lat) else None
            args = {"service_ms": round((hi - lo) / 1e6, 4)}
            if e2e is not None:
                args["latency_ms"] = e2e      # includes queueing in paced runs
            args["iteration"] = it
            ev.append({"ph": "X", "name": "iteration", "cat": "iteration", "pid": synth,
                       "tid": iter_tid, "ts": us(lo), "dur": round((hi - lo) / 1000.0, 3),
                       "args": args})
            if dl is not None and e2e is not None and e2e > dl:
                ev.append({"ph": "i", "s": "p", "name": f"deadline miss ({e2e:.3f} > {dl:g} ms)",
                           "cat": "deadline", "pid": synth, "tid": iter_tid, "ts": us(hi)})
        if dl is not None and not by_iter:
            for e in events:            # markers: misses on @iteration spans
                if e["stage"] == ITERATION and e["dur_ns"] / 1e6 > dl:
                    pid = e.get("pid") or synth
                    ev.append({"ph": "i", "s": "p", "name": f"deadline miss "
                               f"({e['dur_ns'] / 1e6:.3f} > {dl:g} ms)", "cat": "deadline",
                               "pid": pid, "tid": iter_tid,
                               "ts": us(origin + e["start_ns"] + e["dur_ns"])})

        target = ((r.get("observation") or {}).get("target") or {})
        for pid in pids_here:
            if pid == synth:
                meta("process_name", pid, f"{pname} — {label}")
            elif pid not in seen_pids:
                who = " ".join(target.get("command") or []) if target.get("pid") == pid else ""
                meta("process_name", pid, f"pid {pid}{' ' + who if who else ''} — {label}")
            if pid not in seen_pids:
                sort(pid)
                seen_pids.add(pid)
            meta("thread_name", pid, "end-to-end iterations", iter_tid)
        for (pid, raw), t in tids.items():
            if pid == synth:
                meta("thread_name", pid, "stages" if len(tids) == 1 else f"thread {t}", t)

        # -------- per-process counters (watch / record of an application)
        ps = r.get("process_series") or []
        if ps:
            ppid = target.get("pid") or synth
            if ppid not in seen_pids:
                meta("process_name", ppid, f"pid {ppid} — {label}")
                sort(ppid)
                seen_pids.add(ppid)
            for s in ps:
                if s.get("t_mono_ns") is None:
                    continue
                args = {name: s[k] for k, name in PROCESS_COUNTERS if s.get(k) is not None}
                if args:
                    ev.append({"ph": "C", "name": "Process", "pid": ppid,
                               "ts": us(s["t_mono_ns"]), "args": args})

        # -------- board counters
        series = r.get("telemetry_series") or []
        if series:
            bpid = BOARD_PID_BASE + i
            board = (r.get("environment") or {}).get("board") or "board"
            meta("process_name", bpid, f"Board telemetry: {board} — {label}")
            sort(bpid)
            for s in series:
                if s.get("t_mono_ns") is None:
                    continue
                ts = us(s["t_mono_ns"])
                groups = {}
                for k, name, arg in COUNTERS:
                    if s.get(k) is not None:
                        groups.setdefault(name, {})[arg] = s[k]
                for name, args in groups.items():
                    ev.append({"ph": "C", "name": name, "pid": bpid, "ts": ts, "args": args})
                rails = s.get("rails_w") or {}
                if rails:
                    ev.append({"ph": "C", "name": "Rail", "pid": bpid, "ts": ts,
                               "args": {f"{k} W": v for k, v in sorted(rails.items())}})

    trace = {"traceEvents": ev, "displayTimeUnit": "ms",
             "metadata": {"generator": "edgelens export --perfetto", "alignment": mode,
                          "files": [label for label, _ in items], "notes": notes,
                          "clock": "CLOCK_MONOTONIC"}}
    return trace, notes


def export(paths, out_path, align="auto"):
    items = []
    for p in paths:
        try:
            with open(p) as f:
                items.append((str(p), json.load(f)))
        except (OSError, ValueError) as e:
            raise TimelineError(f"Cannot read {p}: {e}") from None
    trace, notes = build(items, align)
    with open(out_path, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    spans = sum(1 for e in trace["traceEvents"] if e["ph"] == "X")
    counters = sum(1 for e in trace["traceEvents"] if e["ph"] == "C")
    return {"spans": spans, "counter_samples": counters, "notes": notes,
            "alignment": trace["metadata"]["alignment"]}

"""
edgelens.attach.runner
------------------------
One observation session for systems EdgeLens doesn't run itself:

  * a marker collector (spans from C/C++/Python inside the application),
  * board telemetry (CPU, GPU, memory, temperature, power),
  * per-process sampling of the target (attach mode),

all on CLOCK_MONOTONIC, so a span, a power sample and a context-switch
burst can be lined up exactly. Used by `edgelens record` (markers first)
and `edgelens watch` (process first); both produce a standard result
document.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time

from ..core.clock import now_ns
from ..core.harness import measure_idle_power
from ..core.identity import attach_identity
from ..core.result import build_result
from ..hardware import telemetry as telemetry_mod
from ..markers import protocol
from ..markers.collector import MarkerCollector
from ..markers.session import spans_to_result
from .procsampler import ProcessSampler, TargetError, find_process

SETTLE_S = 0.25          # let the last datagrams arrive after the target stops


class ObservedRun:
    def __init__(self):
        self.spans = []
        self.marker_stats = None
        self.telemetry_summary = {}
        self.telemetry_series = []
        self.process = None
        self.process_series = []
        self.origin_ns = None
        self.exit_code = None
        self.interrupted = False
        self.ended_by = None
        self.idle_power_w = None
        self.target = None
        self.target_pids = None
        self.private_socket = False
        self.process_note = None
        self.pid_verdicts = {}         # sender pid -> True / False / None (gone before check)


def observe(cmd=None, pid=None, name=None, duration_s=None, socket_path=None, markers=True,
            telemetry=True, telemetry_interval_s=0.1, process_interval_s=0.5,
            children=True, idle_baseline_s=None, poll_s=0.05):
    run = ObservedRun()
    if idle_baseline_s:
        run.idle_power_w = measure_idle_power(idle_baseline_s)
    if name is not None and pid is None and not cmd:
        pid = find_process(name)

    # A started command gets a PRIVATE socket (inherited via EDGELENS_SOCKET),
    # so markers from any other process on the board can't mix into its run.
    private_dir = None
    if markers and cmd and socket_path is None:
        private_dir = tempfile.mkdtemp(prefix="edgelens-")
        socket_path = os.path.join(private_dir, "markers.sock")
    collector = MarkerCollector(socket_path) if markers else None
    if collector is not None and pid is not None and not cmd:
        # Shared socket: decide whether each sender belongs to the target the
        # moment its first span arrives (it is alive then), by its ancestry —
        # short-lived workers would be missed by periodic sampling.
        collector.add_listener(_ancestry_checker(pid, run.pid_verdicts, children))
    if collector is not None:
        collector.start()
    recorder = None
    if telemetry:
        recorder = telemetry_mod.TelemetryRecorder(interval_s=telemetry_interval_s)
        recorder.start()
    run.origin_ns = now_ns()

    child = None
    sampler = None
    try:
        if cmd:
            env = os.environ.copy()
            if collector is not None:
                env[protocol.ENV_SOCKET] = collector.path
            try:
                child = subprocess.Popen(cmd, env=env)
            except OSError as e:
                raise RuntimeError(f"Cannot start {cmd[0]}: {e.strerror or e}") from e
            pid = child.pid
            run.target = {"kind": "command", "command": cmd, "pid": pid}
        elif pid is not None:
            run.target = {"kind": "pid", "pid": pid}
        if pid is not None and recorder is not None and not cmd:
            recorder.track_pid(pid)          # the target is workload, not background
        if pid is not None:
            try:
                sampler = ProcessSampler(pid, interval_s=process_interval_s,
                                         children=children).start()
            except TargetError:
                # A command EdgeLens started can finish (and be a zombie)
                # before process sampling begins: its markers are still
                # valid. Anything else (a --pid that is gone) stays an error.
                if child is None or child.poll() is None:
                    raise
                run.process_note = ("the command exited before process sampling started; "
                                    "markers and board telemetry are complete")

        end_at = time.monotonic() + duration_s if duration_s else None
        try:
            while True:
                if child is not None and child.poll() is not None:
                    run.ended_by = "command exited"
                    break
                if child is None and sampler is not None and sampler.exited_mono_ns:
                    run.ended_by = "process exited"
                    break
                if end_at is not None and time.monotonic() >= end_at:
                    run.ended_by = "duration reached"
                    break
                time.sleep(poll_s)
        except KeyboardInterrupt:
            run.interrupted = True
            run.ended_by = "interrupted (Ctrl-C)"
    finally:
        if child is not None:
            if run.interrupted and child.poll() is None:
                # Ctrl-C already reached the child (same process group): give it
                # time to exit and flush, instead of interrupting its cleanup.
                try:
                    child.wait(timeout=5)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    pass
            if child.poll() is None:
                # Ask politely first, so the app can flush its markers.
                try:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                except KeyboardInterrupt:
                    child.kill()
                    child.wait()
            run.exit_code = child.returncode
        time.sleep(SETTLE_S)
        if sampler is not None:
            run.process = sampler.stop()
            run.process_series = sampler.series(run.origin_ns)
        if recorder is not None:
            run.telemetry_summary = recorder.stop()
            run.telemetry_series = recorder.series()
        if collector is not None:
            run.spans, run.marker_stats = collector.stop()
        if private_dir:
            shutil.rmtree(private_dir, ignore_errors=True)
        if sampler is not None:
            run.target_pids = sorted(sampler.pids_seen)
        run.private_socket = private_dir is not None
    return run


def _ancestry_checker(target_pid, verdicts, children=True):
    import psutil

    def check(msg):
        p = msg["pid"]
        if p in verdicts:
            return
        if p == target_pid:
            verdicts[p] = True
            return
        if not children:                 # --no-children: the target itself only
            verdicts[p] = False
            return
        try:
            proc = psutil.Process(p)
            if target_pid in {x.pid for x in proc.parents()}:
                verdicts[p] = True
            elif proc.ppid() == 1:
                # re-parented to init (its own parent exited): could be the
                # target's grandchild, can't tell -> unverified, blocks a PASS
                verdicts[p] = None
            else:
                verdicts[p] = False
        except psutil.Error:
            verdicts[p] = None
    return check


def to_result(run, *, name, pack="custom", deadline_ms=None, period_ms=None, mode="markers"):
    extra = {"observation": {"target": run.target, "ended_by": run.ended_by,
                             "exit_code": run.exit_code, "interrupted": run.interrupted}}
    if run.process is not None:
        extra["process"] = run.process
    elif run.process_note:
        extra["process_note"] = run.process_note
        extra["process_series"] = run.process_series
    spans, foreign, unverified = run.spans, [], []
    own = set()
    if run.target_pids and not run.private_socket:
        # Shared socket (watch --pid/--name): keep only the target's own spans
        # (its tree as sampled, plus senders whose ancestry was confirmed).
        own = set(run.target_pids) | {p for p, v in run.pid_verdicts.items() if v}
        foreign = [sp for sp in run.spans if sp["pid"] not in own]
        unverified = [sp for sp in foreign if run.pid_verdicts.get(sp["pid"]) is None]
        spans = [sp for sp in run.spans if sp["pid"] in own]
    if run.marker_stats is not None and foreign:
        senders = [x for x in run.marker_stats.get("senders", []) if x["pid"] in own]
        received = sum(x["received"] for x in senders)
        lost = sum(x["lost"] for x in senders)
        run.marker_stats = dict(run.marker_stats, senders=senders, received=received,
                                lost=lost, duplicate_seq=sum(x["duplicate_seq"] for x in senders),
                                lost_ratio=round(lost / (received + lost), 6)
                                if received + lost else 0.0)
    if run.marker_stats is not None:
        run.marker_stats = dict(run.marker_stats,
                                private_socket=run.private_socket,
                                foreign_spans=len(foreign),
                                foreign_unverified=len(unverified),
                                foreign_pids=sorted({sp["pid"] for sp in foreign}))
    if spans:
        return spans_to_result(
            spans, run.marker_stats, name=name, pack=pack, deadline_ms=deadline_ms,
            period_ms=period_ms, telemetry_summary=run.telemetry_summary,
            telemetry_series=run.telemetry_series, origin_ns=run.origin_ns, mode=mode,
            pipeline_source=_source(run, name), extra=extra, idle_power_w=run.idle_power_w)
    # No spans: resources only. Latency is NOT measured and says so.
    extra["markers"] = dict(run.marker_stats or {"received": 0},
                            note="No marker spans received: latency was not measured. "
                                 "Add edgelens_mark.h / edgelens.mark spans for "
                                 "per-inference timing.")
    result = build_result(
        pipeline_desc={"name": name, "pack": pack, "config": {}, "stages": []},
        stage_samples_ms={}, e2e_ms=[], mode=mode, pipeline_source=_source(run, name),
        telemetry_summary=run.telemetry_summary, telemetry_series=run.telemetry_series,
        deadline_ms=deadline_ms, period_ms=period_ms, trace_origin_ns=run.origin_ns,
        idle_power_w=run.idle_power_w, extra=extra)
    attach_identity(result)
    return result


def _source(run, name):
    t = run.target or {}
    if t.get("kind") == "command":
        return "command:" + " ".join(t["command"])[:200]
    if t.get("kind") == "pid":
        return f"pid:{t['pid']}"
    return f"markers:{name}"

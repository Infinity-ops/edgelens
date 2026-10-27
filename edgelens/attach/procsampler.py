"""
edgelens.attach.procsampler
-----------------------------
Per-process resource sampling from outside the process (attach mode). No
code changes in the observed application, no ptrace, no root for your own
processes: everything comes from /proc via psutil.

Per sample (summed over the process and, optionally, its descendants):
  CPU time (-> % of one core), resident memory, threads, voluntary and
  involuntary context switches.

Involuntary context switches are the real-time signal: the scheduler took
the CPU away from the process (preemption). A rising rate during inference
means something else competes for the cores.

What attach mode can NOT see: per-inference latency (add markers for that)
and per-process GPU usage (Jetson's integrated GPU has no per-process
accounting; board-level GPU % is in the telemetry series).
"""

import threading
import time

import psutil

from ..core.clock import now_ns


class TargetError(RuntimeError):
    pass


def find_process(name):
    """The one process whose name or command line matches `name`."""
    me = psutil.Process().pid
    hits = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == me:
            continue
        cmd = " ".join(p.info.get("cmdline") or [])
        if p.info.get("name") == name or (name in cmd and "edgelens" not in cmd):
            hits.append(p)
    exact = [p for p in hits if p.info.get("name") == name]
    pick = exact or hits
    if not pick:
        raise TargetError(f"No running process matches '{name}'.")
    if len(pick) > 1:
        listing = ", ".join(f"{p.info['pid']} ({p.info.get('name')})" for p in pick[:8])
        raise TargetError(f"'{name}' matches {len(pick)} processes: {listing}. "
                          f"Use --pid to pick one.")
    return pick[0].info["pid"]


def _collect(root, children):
    procs = [root]
    if children:
        try:
            procs += root.children(recursive=True)
        except psutil.Error:
            pass
    cpu = rss = threads = vol = invol = 0
    alive = 0
    pids = set()
    for p in procs:
        try:
            if p.status() == psutil.STATUS_ZOMBIE:
                continue          # exited, not yet reaped by its parent: not running
            with p.oneshot():
                t = p.cpu_times()
                m = p.memory_info()
                c = p.num_ctx_switches()
                n = p.num_threads()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            continue
        cpu += t.user + t.system
        rss += m.rss
        threads += n
        vol += c.voluntary
        invol += c.involuntary
        alive += 1
        pids.add(p.pid)
    return alive, cpu, rss, threads, vol, invol, pids


class ProcessSampler:
    def __init__(self, pid, interval_s=0.5, children=True):
        try:
            self.proc = psutil.Process(pid)
            self.name = self.proc.name()
            self.cmdline = " ".join(self.proc.cmdline())[:500]
        except psutil.NoSuchProcess:
            raise TargetError(f"No process with pid {pid}.")
        except psutil.AccessDenied:
            raise TargetError(f"Not allowed to read process {pid} (another user's process: "
                              f"run EdgeLens as that user or with sudo).")
        self.pid = pid
        self.interval_s = interval_s
        self.children = children
        self.raw = []
        self.pids_seen = {pid}            # the target and every descendant ever sampled
        self.exited_mono_ns = None
        self._costs = []
        self._stop = threading.Event()
        self._thread = None
        self.start_mono_ns = None

    def start(self):
        self.start_mono_ns = now_ns()
        self._thread = threading.Thread(target=self._run, name="edgelens-proc", daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        c0 = time.thread_time()
        alive, cpu, rss, threads, vol, invol, pids = _collect(self.proc, self.children)
        self.pids_seen |= pids
        self._costs.append((time.thread_time() - c0) * 1000.0)
        if not alive:
            if self.exited_mono_ns is None:
                self.exited_mono_ns = now_ns()
            return False
        self.raw.append((now_ns(), cpu, rss, threads, vol, invol, alive))
        return True

    def _run(self):
        while not self._stop.is_set():
            if not self._sample():
                break
            self._stop.wait(self.interval_s)

    def alive(self):
        return self.exited_mono_ns is None and self.proc.is_running() and \
            self.proc.status() != psutil.STATUS_ZOMBIE

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.exited_mono_ns is None:
            self._sample()
        return self.summary()

    # ---- results ----

    def series(self, origin_ns=None):
        origin = self.start_mono_ns if origin_ns is None else origin_ns
        out = []
        for a, b in zip(self.raw, self.raw[1:]):
            dt = (b[0] - a[0]) / 1e9
            if dt <= 0:
                continue
            out.append({
                "t_s": round((b[0] - origin) / 1e9, 4), "t_mono_ns": b[0],
                "cpu_percent": round((b[1] - a[1]) / dt * 100.0, 1),     # % of one core
                "rss_mb": round(b[2] / 1e6, 2),
                "threads": b[3],
                "ctx_voluntary_per_s": round((b[4] - a[4]) / dt, 1),
                "ctx_involuntary_per_s": round((b[5] - a[5]) / dt, 1),
                "processes": b[6],
            })
        return out

    def summary(self):
        s = self.series()
        base = {"pid": self.pid, "name": self.name, "cmdline": self.cmdline,
                "include_children": self.children, "samples": len(s),
                "sampler_cpu_ms_mean": (round(sum(self._costs) / len(self._costs), 3)
                                        if self._costs else None),
                "exited": self.exited_mono_ns is not None,
                "interval_s": self.interval_s,
                # CPU time is counted in kernel ticks (10 ms on Linux/Jetson), so
                # one sample's CPU% is only good to about this many points:
                "cpu_percent_resolution": round(1.0 / self.interval_s, 1),
                "gpu": "not attributable per process on Jetson; board-level GPU % is in "
                       "the telemetry series"}
        if not s:
            base["note"] = "Fewer than 2 samples: the process ended too quickly to measure."
            return base

        def mean(k):
            return round(sum(x[k] for x in s) / len(s), 2)

        # A process that exits by itself frees memory while it shuts down
        # (Python's finalizer returning large buffers to the OS). A sample in
        # that moment is real, but it is not the running process: an ~80 MB
        # child was reported as "17 -> 9 MB", and the trailing drop bends the
        # growth slope down. Leave falling trailing samples of an exited
        # process out of start/end/slope; report them as rss_mb_at_exit.
        running = s
        if self.exited_mono_ns is not None:
            while len(running) >= 3 and running[-1]["rss_mb"] < 0.9 * running[-2]["rss_mb"]:
                running = running[:-1]
        rss = [x["rss_mb"] for x in running]
        ts = [x["t_s"] for x in running]
        span_s = (self.raw[-1][0] - self.raw[0][0]) / 1e9
        base.update({
            "duration_s": round(span_s, 2),
            "cpu_percent_mean": mean("cpu_percent"),
            "cpu_percent_peak": max(x["cpu_percent"] for x in s),
            "cpu_cores_used_mean": round(mean("cpu_percent") / 100.0, 2),
            "rss_mb_start": rss[0], "rss_mb_end": rss[-1], "rss_mb_peak": max(rss),
            "rss_growth_mb_per_min": _slope_per_min(ts, rss),
            "rss_mb_at_exit": s[-1]["rss_mb"] if len(running) < len(s) else None,
            "threads_min": min(x["threads"] for x in s),
            "threads_max": max(x["threads"] for x in s),
            "ctx_voluntary_per_s_mean": mean("ctx_voluntary_per_s"),
            "ctx_involuntary_per_s_mean": mean("ctx_involuntary_per_s"),
            "ctx_involuntary_per_s_peak": max(x["ctx_involuntary_per_s"] for x in s),
        })
        return base


def _slope_per_min(ts, ys):
    """Least-squares slope, units per minute (memory growth rate)."""
    n = len(ts)
    if n < 3 or ts[-1] - ts[0] <= 0:
        return None
    mt, my = sum(ts) / n, sum(ys) / n
    num = sum((t - mt) * (y - my) for t, y in zip(ts, ys))
    den = sum((t - mt) ** 2 for t in ts)
    return round(num / den * 60.0, 3) if den else None

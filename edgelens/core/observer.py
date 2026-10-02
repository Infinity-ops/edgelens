"""
edgelens.core.observer
------------------------
Observer mode: keep your own loop, add a few lines, get the same result
document as harness mode.

    import edgelens as el

    with el.trace("detector", pack="vision", deadline_ms=33) as t:
        while running:
            with t.iteration():                 # optional, see below
                with t.stage("capture"):  frame = cam.read()
                with t.stage("inference"): dets = model(frame)
                with t.stage("nms"):       out = nms(dets)
    t.result            # dict; t.save("run.json") writes it

Iteration boundaries: use `with t.iteration():` for exact end-to-end
latency (including the time between stages). Without it, a new iteration
starts automatically whenever a stage name repeats, and end-to-end latency
is measured from the first stage's start to the last stage's end.

Thread-safe: stages may be recorded from several threads; each event carries
its thread id. Iterations are tracked per tracer, so use one tracer per
logical stream.
"""

import json
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from ..hardware import telemetry as telemetry_mod
from ..packs import get_pack
from .harness import measure_idle_power
from .pipeline import infer_role
from .result import MAX_TRACE_EVENTS, build_result


class Tracer:
    def __init__(self, name="pipeline", pack="custom", deadline_ms=None, period_ms=None,
                 telemetry=True, telemetry_interval_s=0.1, idle_baseline_s=None,
                 save=None, **config):
        self.name = name
        self.pack = get_pack(pack)
        self.config = self.pack.configure(dict(config))
        self.deadline_ms = (deadline_ms if deadline_ms is not None
                            else self.pack.default_deadline_ms(self.config))
        self.period_ms = (period_ms if period_ms is not None
                          else self.pack.default_period_ms(self.config))
        self._telemetry = telemetry
        self._telemetry_interval_s = telemetry_interval_s
        self._idle_baseline_s = idle_baseline_s
        self._save = save
        self._lock = threading.Lock()
        self._stage_order = OrderedDict()
        self._samples = {}
        self._events = []
        self._truncated = False
        self._e2e = []
        self._iter = -1
        self._iter_seen = set()
        self._iter_bounds = None            # [first_start, last_end] for auto mode
        self._explicit_depth = 0
        self._recorder = None
        self._idle_w = None
        self.result = None

    # ---- lifecycle ----

    def __enter__(self):
        if self._idle_baseline_s:
            self._idle_w = measure_idle_power(self._idle_baseline_s)
        if self._telemetry:
            self._recorder = telemetry_mod.TelemetryRecorder(interval_s=self._telemetry_interval_s)
            self._recorder.start()
        self._origin = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close_auto_iteration()
        tel = self._recorder.stop() if self._recorder else {}
        series = self._recorder.series() if self._recorder else []
        desc = {
            "name": self.name, "pack": self.pack.name, "config": self.config,
            "stages": [{"name": n, "role": self.pack.role_for(n) or infer_role(n)}
                       for n in self._stage_order],
        }
        self.result = build_result(
            pipeline_desc=desc, stage_samples_ms=self._samples, e2e_ms=self._e2e,
            mode="hardware", pipeline_source=f"observer:{self.name}",
            telemetry_summary=tel, telemetry_series=series, events=self._events,
            events_truncated=self._truncated, deadline_ms=self.deadline_ms,
            period_ms=self.period_ms, paced=False, idle_power_w=self._idle_w,
            pack=self.pack,
        )
        from .identity import attach_identity
        attach_identity(self.result)
        if self._save:
            self.save(self._save)
        return False

    def save(self, path):
        if self.result is None:
            raise RuntimeError("Tracer.save() is available after the `with` block ends.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.result, indent=2))
        return str(path)

    # ---- recording ----

    def _close_auto_iteration(self):
        if self._iter_bounds is not None:
            start, end = self._iter_bounds
            self._e2e.append((end - start) / 1e6)
            self._iter_bounds = None

    @contextmanager
    def iteration(self):
        with self._lock:
            self._close_auto_iteration()
            self._iter += 1
            self._iter_seen = set()
            self._explicit_depth += 1
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            t1 = time.perf_counter_ns()
            with self._lock:
                self._explicit_depth -= 1
                self._e2e.append((t1 - t0) / 1e6)

    @contextmanager
    def stage(self, name):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            t1 = time.perf_counter_ns()
            self.record(name, t0, t1)

    def record(self, name, start_ns, end_ns):
        """Record a stage interval measured elsewhere (perf_counter_ns clock)."""
        with self._lock:
            if self._explicit_depth == 0:
                if self._iter < 0 or name in self._iter_seen:
                    self._close_auto_iteration()
                    self._iter += 1
                    self._iter_seen = set()
                if self._iter_bounds is None:
                    self._iter_bounds = [start_ns, end_ns]
                else:
                    self._iter_bounds[1] = max(self._iter_bounds[1], end_ns)
            self._iter_seen.add(name)
            self._stage_order.setdefault(name, None)
            self._samples.setdefault(name, []).append((end_ns - start_ns) / 1e6)
            if len(self._events) < MAX_TRACE_EVENTS:
                self._events.append({"iter": self._iter, "stage": name,
                                     "start_ns": start_ns - self._origin,
                                     "dur_ns": end_ns - start_ns,
                                     "thread": threading.get_ident()})
            else:
                self._truncated = True


def trace(name="pipeline", pack="custom", **kwargs):
    """Observer-mode entry point; see the module docstring."""
    return Tracer(name=name, pack=pack, **kwargs)

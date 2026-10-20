"""
edgelens.core.gcwatch
-----------------------
Records Python garbage-collector pauses during a timed loop, and controls
the collector for the run (--gc normal | freeze | off).

A full (generation-2) collection stops every Python thread for as long as
it takes to scan all tracked objects. Found on an AGX Orin 32GB: one
gen-2 collection took 27.4 ms and caused the only deadline miss in 30 000
iterations of a 0.18 ms pipeline. Per-stage timing alone can't explain
that — the pause lands in whichever stage happens to be running (it was
`inference` in one run and `preprocess` in the next). So each pause is
recorded with the iteration it hit, and diagnose names it (GC_PAUSE).

Modes:
  normal  the collector runs as it would in the application (default).
  freeze  after warmup: gc.collect(); gc.freeze() — start-up objects
          (imports, the loaded model) are never scanned again. This is what a
          tuned real-time Python application does; pauses that remain come
          from the loop's own allocations.
  off     collector disabled for the timed loop: a lower bound, not a
          deployable configuration (memory in reference cycles is never freed).
"""

import gc
from array import array

from .clock import now_ns

GC_MODES = ("normal", "freeze", "off")
MIN_RECORDED_PAUSE_MS = 1.0
MAX_RECORDED_PAUSES = 1000


class GcWatch:
    def __init__(self, mode="normal", min_pause_ms=MIN_RECORDED_PAUSE_MS,
                 max_pauses=MAX_RECORDED_PAUSES):
        if mode not in GC_MODES:
            raise ValueError(f"gc mode must be one of {', '.join(GC_MODES)}, not {mode!r}")
        self.mode = mode
        self.current_iter = -1           # set by the loop; read by the callback
        self._min_ns = int(min_pause_ms * 1e6)
        self._max = max_pauses
        self._counts = [0, 0, 0]
        self._total_ns = [0, 0, 0]
        self._max_ns = 0
        self._p_start = array("q")
        self._p_dur = array("q")
        self._p_gen = array("b")
        self._p_iter = array("q")
        self._dropped = 0
        self._t0 = 0
        self._origin = 0
        self._was_enabled = True
        self._frozen = False
        self._installed = False

    # ---- lifecycle (around the timed loop only) ----

    def start(self, origin_ns):
        self._origin = origin_ns
        self._was_enabled = gc.isenabled()
        if self.mode in ("freeze", "off"):
            gc.collect()
        if self.mode == "freeze" and hasattr(gc, "freeze"):
            gc.freeze()
            self._frozen = True
        if self.mode == "off":
            gc.disable()
        gc.callbacks.append(self._callback)
        self._installed = True

    def stop(self):
        if self._installed:
            try:
                gc.callbacks.remove(self._callback)
            except ValueError:
                pass
            self._installed = False
        if self._frozen:
            gc.unfreeze()
            self._frozen = False
        if self.mode == "off" and self._was_enabled:
            gc.enable()

    def _callback(self, phase, info):
        # Runs inside the collector: integers and array appends only.
        if phase == "start":
            self._t0 = now_ns()
            return
        dur = now_ns() - self._t0
        gen = info.get("generation", 0)
        if 0 <= gen <= 2:
            self._counts[gen] += 1
            self._total_ns[gen] += dur
        if dur > self._max_ns:
            self._max_ns = dur
        if dur >= self._min_ns:
            if len(self._p_dur) < self._max:
                self._p_start.append(self._t0 - self._origin)
                self._p_dur.append(dur)
                self._p_gen.append(gen)
                self._p_iter.append(self.current_iter)
            else:
                self._dropped += 1

    # ---- result ----

    def summary(self, e2e_ms=None, deadline_ms=None):
        """The result document's "gc" block."""
        e2e_ms = e2e_ms or []
        pauses, missed_iters = [], set()
        for t0, d, g, it in zip(self._p_start, self._p_dur, self._p_gen, self._p_iter):
            p = {"start_ns": t0, "dur_ms": round(d / 1e6, 3), "generation": g,
                 "iter": it if it >= 0 else None}
            if 0 <= it < len(e2e_ms):
                p["iter_latency_ms"] = round(e2e_ms[it], 3)
                if deadline_ms:
                    p["deadline_missed"] = e2e_ms[it] > deadline_ms
                    if p["deadline_missed"]:
                        missed_iters.add(it)
            pauses.append(p)
        misses = sum(1 for v in e2e_ms if deadline_ms and v > deadline_ms)
        return {
            "mode": self.mode,
            "collector_enabled_during_run": self.mode != "off" and self._was_enabled,
            "thresholds": list(gc.get_threshold()),
            "collections": {f"gen{g}": self._counts[g] for g in range(3)},
            "pause_ms_total": {f"gen{g}": round(self._total_ns[g] / 1e6, 3) for g in range(3)},
            "max_pause_ms": round(self._max_ns / 1e6, 3),
            "min_recorded_pause_ms": self._min_ns / 1e6,
            "pauses": pauses,
            "pauses_not_recorded": self._dropped,
            "deadline_misses": misses if deadline_ms else None,
            "deadline_misses_during_gc": len(missed_iters) if deadline_ms else None,
        }

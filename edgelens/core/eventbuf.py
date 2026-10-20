"""
edgelens.core.eventbuf
------------------------
Trace storage that creates no Python container objects while recording.

Why: the trace used to be one dict per stage per iteration. Dicts are
tracked by Python's garbage collector, so a 30 000-iteration, 6-stage run
added 180 000 tracked objects to the measured process. On an AGX Orin
(CPU at 729 MHz, clocks not pinned) that made one full (generation-2)
collection take 27.4 ms in the middle of the timed loop and miss a 10 ms
deadline that the pipeline itself never misses (worst 1.3 ms with the
collector off). The measurement was disturbing what it measured.

Here every event is a few machine integers appended to `array.array`
columns; stage names are interned once. The dicts the result document
needs are built by to_dicts() AFTER the timed loop.
"""

from array import array

from .result import MAX_TRACE_EVENTS


class EventBuffer:
    """Columns of (iter, stage, start_ns, dur_ns, thread); bounded."""

    __slots__ = ("limit", "truncated", "_iter", "_stage", "_start", "_dur", "_thread",
                 "_names", "_index")

    def __init__(self, limit=MAX_TRACE_EVENTS):
        self.limit = limit
        self.truncated = False
        self._iter = array("q")
        self._stage = array("i")
        self._start = array("q")
        self._dur = array("q")
        self._thread = array("Q")
        self._names = []
        self._index = {}

    def __len__(self):
        return len(self._iter)

    def add(self, iteration, stage, start_ns, dur_ns, thread):
        if len(self._iter) >= self.limit:
            self.truncated = True
            return
        idx = self._index.get(stage)
        if idx is None:
            idx = self._index[stage] = len(self._names)
            self._names.append(stage)
        self._iter.append(iteration)
        self._stage.append(idx)
        self._start.append(start_ns)
        self._dur.append(dur_ns)
        self._thread.append(thread)

    def to_dicts(self):
        """The result document's trace.events shape (unchanged since 0.1.0)."""
        names = self._names
        return [{"iter": i, "stage": names[s], "start_ns": t0, "dur_ns": d, "thread": th}
                for i, s, t0, d, th in zip(self._iter, self._stage, self._start,
                                           self._dur, self._thread)]

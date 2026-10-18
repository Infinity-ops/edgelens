"""
edgelens.mark
---------------
Inference markers from Python, for code EdgeLens does not run itself (a
ROS 2 node, a service, a script started by someone else). Spans go to an
EdgeLens collector (`edgelens record` / `edgelens watch`) over a UNIX
datagram socket; the C/C++ header edgelens_mark.h speaks the same protocol.

    from edgelens import mark

    while running:
        with mark.iteration():                    # whole iteration (end-to-end)
            with mark.span("build_window"):
                x = window()
            with mark.span("lstm", sync=torch.cuda.synchronize):
                y = model(x)                       # GPU work: sync before the end stamp

Cost: two clock reads and one non-blocking sendto(); `edgelens selftest`
measures it next to the kernel's own sendto() cost. The application never
waits for the collector: when the collector's queue is full, spans join a
small overflow batch (sent in 4 KB chunks after a 200 us back-off); only
when that is full too is a span dropped — and counted, because every span
gets a sequence number whether or not it was sent.

Fork-safe: a child process becomes a new sender with its own sequence.
"""

import atexit
import itertools
import os
import socket
import threading
import time

from .markers import protocol

_now = time.monotonic_ns
_native_id = getattr(threading, "get_native_id", threading.get_ident)

_PENDING_MAX = 32768
_CHUNK = 4096
_BACKOFF_NS = 200_000
_RETRY_NS = 1_000_000_000            # re-probe a missing collector once per second


class _Sender:
    """Per-process sender state (replaced in a forked child)."""

    def __init__(self):
        self.pid = os.getpid()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.path = protocol.socket_path()
        self.src = f"py{self.pid:x}{id(self) & 0xffff:04x}"
        self.seq = itertools.count()
        self.last_seq = None
        self.lock = threading.Lock()
        self.pending = []
        self.pending_bytes = 0
        self.flush_after = 0
        self.retry_at = 0
        self.sent = 0
        self.dropped = 0
        self.tls = threading.local()
        self.prefix = f"EL1 X {self.src} ".encode()

    def _try(self, payload):
        try:
            self.sock.sendto(payload, self.path)
            return 0
        except (BlockingIOError, InterruptedError):
            return 1                    # queue full
        except OSError:
            return 2                    # no collector

    def _flush_locked(self):
        """Send pending lines in chunks <= 4 KB; 0 when everything went out."""
        while self.pending:
            chunk, size = [], 0
            while self.pending and (not chunk or size + len(self.pending[0]) + 1 <= _CHUNK):
                line = self.pending.pop(0)
                chunk.append(line)
                size += len(line) + 1
            st = self._try(b"\n".join(chunk))
            if st:
                self.pending[0:0] = chunk
                return st
            self.sent += len(chunk)
            self.pending_bytes -= size
        return 0

    def record(self, name, start_ns, end_ns, gpu_synced):
        seq = next(self.seq)
        self.last_seq = seq
        if self.retry_at and end_ns < self.retry_at:
            self.dropped += 1
            return False
        tid = getattr(self.tls, "tid", None)
        if tid is None:
            tid = self.tls.tid = _native_id()
        name = str(name)
        if "\n" in name or "\r" in name:
            name = name.replace("\n", " ").replace("\r", " ")
        if not name.strip():
            name = "?"                             # an empty name would break the line format
        line = self.prefix + (f"{seq} {self.pid} {tid} {start_ns} {end_ns} "
                              f"{'g' if gpu_synced else '-'} {name[:protocol.MAX_NAME]}"
                              ).encode("utf-8", "replace")
        if not self.pending:                       # fast path: no lock, one syscall
            st = self._try(line)
            if st == 0:
                self.sent += 1
                return True
        else:
            st = 1
        with self.lock:
            if self.pending and end_ns >= self.flush_after:
                st = self._flush_locked()
            if not self.pending and st == 0:
                st = self._try(line)
                if st == 0:
                    self.sent += 1
                    self.retry_at = 0
                    return True
            if st == 1:
                if not self.pending or end_ns >= self.flush_after:
                    self.flush_after = end_ns + _BACKOFF_NS
                if self.pending_bytes + len(line) + 1 <= _PENDING_MAX:
                    self.pending.append(line)
                    self.pending_bytes += len(line) + 1
                    return True
        self.dropped += 1
        if st == 2:
            self.retry_at = _now() + _RETRY_NS
        return False

    def flush(self, timeout_s=0.2):
        if self.last_seq is None:
            return
        end = time.monotonic() + timeout_s
        with self.lock:
            while self.pending:
                st = self._flush_locked()
                if st == 0:
                    break
                if st == 2 or time.monotonic() > end:
                    self.dropped += len(self.pending)        # counted lost via the final seq
                    self.pending = []
                    self.pending_bytes = 0
                    break
                time.sleep(0.001)
        final = protocol.format_final(self.src, self.last_seq, self.pid)
        for _ in range(100):
            if self._try(final) != 1:
                break
            time.sleep(0.001)


_sender = None
_create_lock = threading.Lock()


def _get():
    s = _sender
    if s is None or s.pid != os.getpid():
        return _new_sender()
    return s


def _new_sender():
    global _sender
    with _create_lock:
        if _sender is None or _sender.pid != os.getpid():
            _sender = _Sender()
    return _sender


def configure(path=None):
    """Use a specific collector socket (default: $EDGELENS_SOCKET or
    /tmp/edgelens.sock)."""
    s = _get()
    s.path = protocol.socket_path(path)
    s.retry_at = 0


def record(name, start_ns, end_ns, gpu_synced=False):
    """Send one finished span measured with time.monotonic_ns()."""
    return _get().record(name, start_ns, end_ns, gpu_synced)


class span:
    """Time a block:  with mark.span("lstm"): ...

    Pass sync=<callable> (torch.cuda.synchronize, a stream's synchronize)
    when the block launches asynchronous GPU work, so the end timestamp is
    taken after the GPU finished."""

    __slots__ = ("name", "sync", "t0")

    def __init__(self, name, sync=None):
        self.name = name
        self.sync = sync

    def __enter__(self):
        self.t0 = _now()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync = self.sync
        if sync is not None:
            sync()
        _get().record(self.name, self.t0, _now(), sync is not None)
        return False


def iteration(sync=None):
    """Mark one whole iteration: end-to-end latency (incl. gaps between spans)."""
    return span(protocol.ITERATION, sync=sync)


def stats():
    """sent / dropped of this process. `sent` is approximate when several
    threads send at once (unlocked fast path); the collector's sequence-gap
    count is the exact loss figure."""
    s = _get()
    return {"sent": s.sent, "dropped": s.dropped, "socket": s.path}


def flush():
    """Send pending spans and the last sequence number, so losses at the end
    of the run are counted too. Called automatically at interpreter exit."""
    s = _sender
    if s is not None and s.pid == os.getpid():
        s.flush()


def _after_fork_in_child():
    """A forked child is a new sender: fresh id, sequence, batch and lock."""
    global _sender, _create_lock
    _create_lock = threading.Lock()
    _sender = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
atexit.register(flush)

__all__ = ["span", "iteration", "record", "configure", "stats", "flush"]

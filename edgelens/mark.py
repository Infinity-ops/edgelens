"""
edgelens.mark
---------------
Inference markers from Python, for code EdgeLens does not run itself (a
ROS 2 node, a service, a script started by someone else). Spans go to an
EdgeLens collector (`edgelens record` / `edgelens watch --markers`) over a
UNIX datagram socket; the C/C++ header edgelens_mark.h speaks the same
protocol.

    from edgelens import mark

    while running:
        with mark.iteration():                    # whole iteration (end-to-end)
            with mark.span("build_window"):
                x = window()
            with mark.span("lstm", sync=torch.cuda.synchronize):
                y = model(x)                       # GPU work: sync before the end stamp

Cost: one clock read at each end and one non-blocking sendto(); measured by
`edgelens selftest`. A send NEVER blocks: when no collector is listening or
its buffer is full, the span is dropped and counted (the sequence number
still advances, so the collector reports the loss exactly).
"""

import atexit
import itertools
import os
import socket
import threading
import time
from contextlib import contextmanager

from .markers import protocol

_now = time.monotonic_ns
_lock = threading.Lock()
_state = {"sock": None, "path": None, "src": None, "retry_at": 0, "pid": None}
_seq = itertools.count()
_sent = [0]
_dropped = [0]
_RETRY_NS = 1_000_000_000          # re-probe a missing collector once per second
_tls = threading.local()
_native_id = getattr(threading, "get_native_id", threading.get_ident)


def _socket():
    pid = os.getpid()
    if _state["pid"] != pid:                       # first use, or after fork()
        with _lock:
            if _state["pid"] != pid:
                if _state["sock"] is not None:
                    try:
                        _state["sock"].close()
                    except OSError:
                        pass
                s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                s.setblocking(False)
                _tls.__dict__.clear()            # thread ids differ in a forked child
                _state.update(sock=s, path=protocol.socket_path(), pid=pid,
                              src=f"py{pid:x}{id(s) & 0xffff:04x}", retry_at=0,
                              last_seq=None)
    return _state["sock"]


def configure(path=None):
    """Use a specific collector socket (default: $EDGELENS_SOCKET or
    /tmp/edgelens.sock)."""
    _socket()
    _state["path"] = protocol.socket_path(path)
    _state["retry_at"] = 0


_PENDING_MAX = 32768
_pending = []            # overflow batch: encoded lines waiting for queue space
_pending_bytes = [0]


def _try(payload):
    try:
        _state["sock"].sendto(payload, _state["path"])
        return 0
    except (BlockingIOError, InterruptedError):
        return 1                        # queue full
    except OSError:
        return 2                        # no collector


def record(name, start_ns, end_ns, gpu_synced=False):
    """Send one finished span measured with time.monotonic_ns().

    Never blocks. If the collector's queue is full, the span joins a pending
    batch that is sent as one datagram on the next call (bursts get through
    short queues); only when that batch is full too is the span dropped —
    and counted."""
    if _state["pid"] != os.getpid():
        _socket()
    seq = next(_seq)
    _state["last_seq"] = seq
    if _state["retry_at"] and end_ns < _state["retry_at"]:
        _dropped[0] += 1
        return False
    tid = getattr(_tls, "tid", None)
    if tid is None:
        tid = _tls.tid = _native_id()
    line = protocol.format_span(_state["src"], seq, _state["pid"], tid, start_ns, end_ns,
                                "g" if gpu_synced else "-", name)
    if not _pending:                    # fast path: no lock, one syscall
        status = _try(line)
        if status == 0:
            _sent[0] += 1
            if _state["retry_at"]:
                _state["retry_at"] = 0
            return True
    with _lock:
        status = 0
        if _pending:
            status = _try(b"\n".join(_pending))
            if status == 0:
                _sent[0] += len(_pending)
                del _pending[:]
                _pending_bytes[0] = 0
        if not _pending and status == 0:
            status = _try(line)
            if status == 0:
                _sent[0] += 1
                _state["retry_at"] = 0
                return True
        if status == 1 and _pending_bytes[0] + len(line) + 1 <= _PENDING_MAX:
            _pending.append(line)
            _pending_bytes[0] += len(line) + 1
            return True
    _dropped[0] += 1
    if status == 2:                     # no collector listening: back off for a second
        _state["retry_at"] = _now() + _RETRY_NS
    return False


@contextmanager
def span(name, sync=None):
    """Time a block. Pass sync=<callable> (e.g. torch.cuda.synchronize,
    or a stream's synchronize) when the block launches asynchronous GPU
    work, so the end timestamp is taken after the GPU finished."""
    t0 = _now()
    try:
        yield
    finally:
        if sync is not None:
            sync()
        record(name, t0, _now(), gpu_synced=sync is not None)


def iteration(sync=None):
    """Mark one whole iteration: end-to-end latency (incl. gaps between spans)."""
    return span(protocol.ITERATION, sync=sync)


def stats():
    return {"sent": _sent[0], "dropped": _dropped[0], "socket": _state["path"]}


def flush(timeout_s=0.2):
    """Send pending spans and tell the collector the last sequence number,
    so losses at the end of the run are counted too. Called automatically
    at interpreter exit."""
    if _state["sock"] is None or _state["pid"] != os.getpid() or _state.get("last_seq") is None:
        return
    end = time.monotonic() + timeout_s
    with _lock:
        while _pending:
            st = _try(b"\n".join(_pending))
            if st == 0:
                _sent[0] += len(_pending)
                del _pending[:]
                _pending_bytes[0] = 0
            elif st == 2 or time.monotonic() > end:
                _dropped[0] += len(_pending)
                del _pending[:]
                break
            else:
                time.sleep(0.001)
    final = protocol.format_final(_state["src"], _state["last_seq"], _state["pid"])
    for _ in range(100):
        if _try(final) != 1:
            break
        time.sleep(0.001)


def _after_fork_in_child():
    """A forked child is a new sender: fresh id, sequence and (empty) batch,
    and a fresh lock (another parent thread may have held it at fork)."""
    global _lock, _seq
    _lock = threading.Lock()
    _seq = itertools.count()
    del _pending[:]
    _pending_bytes[0] = 0
    _sent[0] = _dropped[0] = 0
    _tls.__dict__.clear()
    _state.update(pid=None, last_seq=None, retry_at=0)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
atexit.register(flush)

__all__ = ["span", "iteration", "record", "configure", "stats", "flush"]

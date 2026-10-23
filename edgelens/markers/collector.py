"""
edgelens.markers.collector
----------------------------
Receives marker datagrams on a UNIX socket, on a background thread, and
keeps exact loss accounting per sender.

Loss accounting: each sender numbers every span it tries to send. Per
sender, lost = (highest seq - lowest seq + 1) - spans received. A sender's
"F" (final) message extends the highest seq, so drops at the very end of a
run are counted too. Spans a sender produced before the collector existed
are not losses (the collector never asked for them); they are reported as
`seq_before_first` so nobody mistakes a late start for a clean run.
"""

import os
import socket
import stat
import threading

from ..core.clock import now_ns
from . import protocol

DEFAULT_MAX_SPANS = 1_000_000          # ~100 MB worst case; then counted, not stored.
                                       # Longer runs: `edgelens soak` (windows, constant memory)
RAW_PARSE_AT = 50_000                  # unparsed datagrams before the receiver parses itself


class CollectorError(RuntimeError):
    pass


class _Sender:
    __slots__ = ("src", "pid", "received", "min_seq", "max_seq", "final_seq", "batched")

    def __init__(self, src, pid):
        self.src, self.pid = src, pid
        self.received = 0
        self.min_seq = None
        self.max_seq = None
        self.final_seq = None
        self.batched = False          # sent overflow batches: the queue was full at times

    def see(self, seq):
        self.received += 1
        self.min_seq = seq if self.min_seq is None else min(self.min_seq, seq)
        self.max_seq = seq if self.max_seq is None else max(self.max_seq, seq)

    def lost(self):
        if self.min_seq is None:
            return 0
        top = self.max_seq if self.final_seq is None else max(self.max_seq, self.final_seq)
        return max(0, (top - self.min_seq + 1) - self.received)

    def duplicates(self):
        """More spans than sequence numbers: two senders share an id (e.g. a
        fork without a fresh id). Loss counts are then unreliable."""
        if self.min_seq is None:
            return 0
        return max(0, self.received - (self.max_seq - self.min_seq + 1))

    def describe(self):
        return {"src": self.src, "pid": self.pid, "received": self.received,
                "first_seq": self.min_seq, "last_seq": self.max_seq,
                "final_seq": self.final_seq, "final_seen": self.final_seq is not None,
                "lost": self.lost(), "seq_before_first": self.min_seq or 0,
                "duplicate_seq": self.duplicates(), "backpressure": self.batched}


def _remove_stale(path):
    """A socket file left by a crashed collector blocks bind(). Remove it only
    when nobody is listening; refuse if another collector is live."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(st.st_mode):
        raise CollectorError(f"{path} exists and is not a socket; refusing to remove it. "
                             f"Pick another path with --socket.")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        probe.connect(path)
    except (ConnectionRefusedError, FileNotFoundError):
        os.unlink(path)
        return
    except OSError:
        os.unlink(path)
        return
    finally:
        probe.close()
    raise CollectorError(f"Another EdgeLens collector is already listening on {path}. "
                         f"Stop it, or use a different --socket (and EDGELENS_SOCKET for "
                         f"the application).")


class MarkerCollector:
    """Background receiver. Usage:

        c = MarkerCollector("/tmp/edgelens.sock"); c.start()
        ... application sends spans ...
        spans, stats = c.stop()
    """

    def __init__(self, path=None, rcvbuf_bytes=8 << 20, max_spans=DEFAULT_MAX_SPANS,
                 keep_spans=True):
        self.path = protocol.socket_path(path)
        self.rcvbuf_bytes = rcvbuf_bytes
        self.max_spans = max_spans
        self.keep_spans = keep_spans      # False: drain() hands spans out, nothing kept
        self._sock = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._raw = []                    # datagrams received, not parsed yet
        self._seen_pids = set()
        self.spans = []
        self.senders = {}
        self.malformed = 0
        self.not_stored = 0
        self.datagrams = 0
        self.started_mono_ns = None
        self.stopped_mono_ns = None
        self.rcvbuf_effective = None
        self._listeners = []

    # ---- lifecycle ----

    def start(self):
        _remove_stale(self.path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_bytes)
        except OSError:
            pass
        try:
            sock.bind(self.path)
        except OSError as e:
            sock.close()
            raise CollectorError(f"Cannot listen on {self.path}: {e.strerror or e}") from e
        try:
            # Let processes of other users (e.g. an inference service running as root) send.
            os.chmod(self.path, 0o666)
        except OSError:
            pass
        sock.settimeout(0.2)
        self.rcvbuf_effective = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self._sock = sock
        self.started_mono_ns = now_ns()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="edgelens-markers",
                                        daemon=True)
        self._thread.start()
        return self

    def add_listener(self, fn):
        """fn({"pid": pid}) is called on the receiver thread the first time a
        process sends anything (e.g. to check its ancestry while it is alive)."""
        self._listeners.append(fn)

    def _run(self):
        # Hot loop: receive and keep raw bytes only. Parsing happens later
        # (drain / stop), so the collector keeps up with fast senders even on
        # a slow CPU (Jetson Nano: parsing on arrival capped it far below
        # what a C application can send).
        sock = self._sock
        raw = self._raw
        listeners = self._listeners
        seen = self._seen_pids
        while not self._stop.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            raw.append(data)
            if listeners:
                self._notify(data, seen)
            if len(raw) > RAW_PARSE_AT:
                # Nobody drains (a long run without drain()): parse here so the
                # raw backlog stays bounded. Costs receive speed only in this case.
                self._parse_pending()
        # drain what is already queued so a fast stop loses nothing
        sock.setblocking(False)
        while True:
            try:
                data = sock.recv(65536)
            except (BlockingIOError, OSError):
                break
            raw.append(data)
            if listeners:
                self._notify(data, seen)

    def _notify(self, data, seen):
        parts = data.split(b" ", 5)
        if len(parts) < 5:
            return
        try:
            pid = int(parts[4])
        except ValueError:
            return
        if pid in seen:
            return
        seen.add(pid)
        for fn in self._listeners:
            try:
                fn({"pid": pid})
            except Exception:
                pass

    def _parse_pending(self):
        """Parse everything received so far. Returns the new spans."""
        with self._lock:
            # The receiver thread appends concurrently: take exactly the first n
            # items and delete exactly those (appends land after them).
            n = len(self._raw)
            batch = self._raw[:n]
            del self._raw[:n]
            new = []
            for data in batch:
                self.datagrams += 1
                if b"\n" in data.rstrip(b"\n"):
                    for line in data.split(b"\n"):
                        if line.strip():
                            self._handle_one(line, new, batched=True)
                else:
                    self._handle_one(data, new)
            return new

    def _handle_one(self, data, new, batched=False):
        msg = protocol.parse(data)
        if msg is None:
            self.malformed += 1
            return
        key = (msg["src"], msg["pid"])
        snd = self.senders.get(key)
        if snd is None:
            snd = self.senders[key] = _Sender(msg["src"], msg["pid"])
        if msg["kind"] == "F":
            snd.final_seq = msg["seq"]
            return
        snd.see(msg["seq"])
        if batched:
            snd.batched = True
        if not self.keep_spans:
            new.append(msg)
        elif len(self.spans) < self.max_spans:
            self.spans.append(msg)
            new.append(msg)
        else:
            self.not_stored += 1

    def drain(self):
        """Spans received since the last drain (parsed now). With
        keep_spans=False this is the only place spans go: long runs stream
        them out instead of holding them in memory."""
        return self._parse_pending()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.stopped_mono_ns = now_ns()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        try:
            os.unlink(self.path)
        except OSError:
            pass
        self._parse_pending()
        return self.spans, self.stats()

    # ---- results ----

    def stats(self):
        self._parse_pending()
        with self._lock:
            senders = [s.describe() for s in self.senders.values()]
        received = sum(s["received"] for s in senders)
        lost = sum(s["lost"] for s in senders)
        total = received + lost
        return {
            "socket": self.path,
            "received": received,
            "lost": lost,
            "lost_ratio": round(lost / total, 6) if total else 0.0,
            "duplicate_seq": sum(s["duplicate_seq"] for s in senders),
            "malformed": self.malformed,
            "not_stored": self.not_stored,
            "datagrams": self.datagrams,
            "senders": senders,
            "rcvbuf_bytes": self.rcvbuf_effective,
            "started_mono_ns": self.started_mono_ns,
            "stopped_mono_ns": self.stopped_mono_ns,
        }

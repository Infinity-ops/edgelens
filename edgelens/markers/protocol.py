"""
edgelens.markers.protocol
---------------------------
Wire format for inference markers: one UNIX *datagram* per finished span.
Plain text, so any language (C, C++, Rust, Go, a shell with socat) can send
it without a library.

    EL1 X <src> <seq> <pid> <tid> <start_ns> <end_ns> <flags> <name>
    EL1 F <src> <seq> <pid>                                  (final, optional)

  src       sender id, unique per process/translation unit (hex or decimal)
  seq       per-src sequence number, incremented for EVERY span the sender
            tried to send — including ones it dropped. Gaps = lost events,
            counted exactly, never silently.
  pid, tid  OS process id and thread id
  start_ns, end_ns
            CLOCK_MONOTONIC nanoseconds (clock_gettime(CLOCK_MONOTONIC) in C,
            time.monotonic_ns() in Python)
  flags     "-" or letters: g = end taken after GPU synchronisation
  name      span name; may contain spaces; "@iteration" marks one whole
            iteration (end-to-end latency)
  F         "I'm done, my last seq was <seq>": makes losses at the very end
            of a run visible too.

Batching: a datagram may hold several such lines separated by "\n". Senders
use this only when the collector's queue is full (Linux limits a UNIX
datagram queue by message COUNT, net.unix.max_dgram_qlen, which can be as
low as 10): the overflow is sent as one datagram on the next call instead of
being dropped.

Why datagrams: a send never blocks the measured application (MSG_DONTWAIT),
each message arrives whole or not at all, and there is no connection to
manage. Why one message per finished span (not begin + end): half the
messages, and a lost message loses one span cleanly instead of corrupting
the pairing of every later one.
"""

import os

VERSION_TAG = "EL1"
ITERATION = "@iteration"
MAX_NAME = 200
DEFAULT_SOCKET = "/tmp/edgelens.sock"
ENV_SOCKET = "EDGELENS_SOCKET"


def socket_path(path=None):
    return path or os.environ.get(ENV_SOCKET) or DEFAULT_SOCKET


def format_span(src, seq, pid, tid, start_ns, end_ns, flags, name):
    name = str(name).replace("\n", " ").replace("\r", " ")[:MAX_NAME] or "?"
    return (f"{VERSION_TAG} X {src} {seq} {pid} {tid} {int(start_ns)} {int(end_ns)} "
            f"{flags or '-'} {name}").encode("utf-8", "replace")


def format_final(src, seq, pid):
    return f"{VERSION_TAG} F {src} {seq} {pid}".encode()


def parse(data):
    """Parse one datagram. Returns a dict, or None when malformed."""
    try:
        text = data.decode("utf-8", "replace").strip()
    except Exception:
        return None
    parts = text.split(" ", 9)
    if len(parts) < 5 or parts[0] != VERSION_TAG:
        return None
    kind = parts[1]
    try:
        if kind == "X" and len(parts) == 10:
            start, end = int(parts[6]), int(parts[7])
            if end < start:
                return None
            return {"kind": "X", "src": parts[2], "seq": int(parts[3]), "pid": int(parts[4]),
                    "tid": int(parts[5]), "start_ns": start, "end_ns": end,
                    "flags": "" if parts[8] == "-" else parts[8], "name": parts[9]}
        if kind == "F":
            return {"kind": "F", "src": parts[2], "seq": int(parts[3]), "pid": int(parts[4])}
    except ValueError:
        return None
    return None

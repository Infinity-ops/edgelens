"""
edgelens.markers.logimport
----------------------------
Timing from logs an application already writes, when you can't add markers:

    edgelens import-log fortis.log --regex 'LSTM inference took (?P<ms>[0-9.]+) ms'

Named groups the regex may use:
  duration (one required unless start/end given):
      (?P<ms>...)  (?P<us>...)  (?P<ns>...)  (?P<s>...)
  timestamps (optional, CLOCK_MONOTONIC or any monotonic ns/us/ms):
      (?P<start_ns>...) (?P<end_ns>...)   — duration taken from them if no ms/us
  stage name (optional):  (?P<stage>...)  else --stage (default "inference")

Evidence is weaker than markers: the application measured its own timing,
EdgeLens only parses it, and there is no loss counting. The result says so
(markers.source = "log").
"""

import re

_UNITS = (("ns", 1), ("us", 1_000), ("ms", 1_000_000), ("s", 1_000_000_000))


class LogImportError(ValueError):
    pass


def compile_regex(pattern):
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise LogImportError(f"Invalid --regex: {e}") from e
    groups = set(rx.groupindex)
    has_dur = any(u in groups for u, _ in _UNITS)
    has_ts = {"start_ns", "end_ns"} <= groups
    if not (has_dur or has_ts):
        raise LogImportError("The regex needs a named duration group (?P<ms>...), "
                             "(?P<us>...), (?P<ns>...) or (?P<s>...), or both "
                             "(?P<start_ns>...) and (?P<end_ns>...).")
    return rx


def parse_lines(lines, pattern, stage="inference"):
    """Returns (spans, info). Spans use synthetic, back-to-back timestamps
    when the log has none, so durations are exact but gaps are unknown."""
    rx = compile_regex(pattern)
    spans, matched, skipped = [], 0, 0
    t = 0
    has_ts = {"start_ns", "end_ns"} <= set(rx.groupindex)
    for line in lines:
        m = rx.search(line)
        if not m:
            continue
        g = m.groupdict()
        try:
            dur_ns = None
            for unit, mult in _UNITS:
                if g.get(unit) not in (None, ""):
                    dur_ns = int(round(float(g[unit]) * mult))
                    break
            if has_ts and g.get("start_ns") and g.get("end_ns"):
                start, end = int(g["start_ns"]), int(g["end_ns"])
                if dur_ns is None:
                    dur_ns = end - start
            else:
                start, end = t, t + (dur_ns or 0)
                t = end
        except ValueError:
            skipped += 1
            continue
        if dur_ns is None or dur_ns < 0:
            skipped += 1
            continue
        matched += 1
        spans.append({"kind": "X", "src": "log", "seq": matched - 1, "pid": 0, "tid": 0,
                      "start_ns": start, "end_ns": start + dur_ns, "flags": "",
                      "name": (g.get("stage") or stage)})
    info = {"source": "log", "received": matched, "lost": None, "lost_ratio": None,
            "malformed": skipped, "senders": [],
            "timestamps": "from log" if has_ts else "synthetic (durations only)",
            "note": "Timing measured by the application itself and parsed from its log; "
                    "no loss accounting. Prefer markers (edgelens_mark.h) for evidence."}
    return spans, info

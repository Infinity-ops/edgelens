"""
edgelens.selftest
-------------------
`edgelens selftest`: before you trust EdgeLens's numbers on a board, measure
EdgeLens itself on that board.

  clock                 one clock (CLOCK_MONOTONIC), resolution, read cost
  harness_accuracy      a reference workload of KNOWN duration, measured by
                        the harness: reported - true = EdgeLens's own error
  python_marker_cost    cost of one Python marker span (edgelens.mark)
  c_marker_cost         cost of one C marker span (edgelens_mark.h), if a
                        C compiler is available
  marker_delivery       spans at a realistic rate: none may be lost; a burst:
                        every span must be received or counted as lost
  dgram_queue           the kernel's UNIX datagram queue limit
  sampler_cpu           CPU the telemetry sampler takes (% of one core)
  telemetry_disturbance the reference workload with telemetry on vs off
  chrony                clock sync state (needed only across machines)

What the reference workload proves: it busy-waits for D ms on the same
clock EdgeLens reads, so the difference between reported and true duration
is the harness's own overhead and jitter (scheduling, Python, bookkeeping) —
not the accuracy of the hardware clock itself, which no software can check
against itself.

Limits are initial, deliberately generous values; they are tightened from
measured Nano / AGX Orin rows in docs/jetson.md.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import __version__
from .core import clock

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

LIMITS = {
    "clock_read_ns": 2_000,                # Python read of CLOCK_MONOTONIC
    "harness_error_us": 50.0,              # |median reported - true| per stage
    "harness_error_pct": 2.0,              # ... or this % of the duration, if larger
    "python_marker_us": 30.0,              # one Python span, collector listening
    "c_marker_us": 5.0,                    # one C span
    "sampler_cpu_pct_of_core": 2.0,
    "telemetry_shift_pct": 3.0,            # mean latency shift caused by telemetry
    "min_dgram_qlen": 64,
}


def _row(check, status, value, unit, limit=None, detail="", fix=None):
    return {"check": check, "status": status, "value": value, "unit": unit, "limit": limit,
            "detail": detail, "fix": fix}


def _spin_ns(ns):
    end = clock.now_ns() + ns
    while clock.now_ns() < end:
        pass


# ---------------------------------------------------------------- checks

def check_clock():
    res = clock.resolution_ns()
    mono = clock.perf_counter_is_monotonic()
    cost = clock.read_cost_ns()
    rows = [_row("clock", PASS if mono else WARN, clock.CLOCK_NAME, "",
                 detail=f"resolution {res} ns; perf_counter on the same clock: {mono}",
                 fix=None if mono else "Non-Linux host: markers from other processes won't "
                                       "line up with stage timings.")]
    rows.append(_row("clock_read_cost", PASS if cost <= LIMITS["clock_read_ns"] else WARN,
                     cost, "ns", LIMITS["clock_read_ns"], "median cost of one clock read "
                     "from Python"))
    return rows


def _reference_run(duration_ms, iterations, telemetry):
    import edgelens as el
    from .core.harness import run_harness
    ns = int(duration_ms * 1e6)
    p = el.Pipeline("selftest-reference")
    p.add_stage("reference", lambda x=None: _spin_ns(ns))
    return run_harness(p, iterations=iterations, warmup=5, telemetry=telemetry,
                       attach_ids=False)


def check_harness(durations_ms, iterations):
    rows = []
    for d in durations_ms:
        r = _reference_run(d, iterations, telemetry=False)
        st = r["stage_stats_ms"]["reference"]
        e2e = r["latency"]
        err_us = (st["p50"] - d) * 1000.0
        tail_us = ((st.get("p99") or st["max"]) - d) * 1000.0
        e2e_extra_us = (e2e["p50"] - st["p50"]) * 1000.0
        limit = max(LIMITS["harness_error_us"], LIMITS["harness_error_pct"] / 100 * d * 1000)
        status = PASS if abs(err_us) <= limit else FAIL
        rows.append(_row("harness_accuracy", status, round(err_us, 2), "us", round(limit, 1),
                         f"{d} ms reference: median error {err_us:+.2f} us, p99 error "
                         f"{tail_us:+.1f} us (preemption shows here), per-iteration "
                         f"bookkeeping {e2e_extra_us:.2f} us"))
    return rows


PY_BENCH = r"""
import socket, sys, time
from edgelens import mark
from edgelens.markers import protocol
path, n = sys.argv[1], int(sys.argv[2])
mark.configure(path)
raw = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
raw.setblocking(False)
msg = protocol.format_span("raw", 0, 0, 0, 0, 1, "-", "selftest")
def median_ns(fn, count):
    t = []
    for i in range(count):
        a = time.perf_counter_ns(); fn(); t.append(time.perf_counter_ns() - a)
        if i % 8 == 7:
            time.sleep(0.0002)
    t.sort()
    return t[len(t) // 2]
def one_span():
    with mark.span("selftest"):
        pass
def one_raw():
    try:
        raw.sendto(msg, path)
    except BlockingIOError:
        pass
span_ns = median_ns(one_span, n)
raw_ns = median_ns(one_raw, min(n, 2000))
mark.flush()
print(span_ns, raw_ns)
"""


def check_python_markers(n):
    """Runs the sender in its own process, like a real application (an
    in-process collector would compete for the GIL and inflate the cost)."""
    from .markers.collector import MarkerCollector
    d = tempfile.mkdtemp(prefix="edgelens-selftest-")
    path = os.path.join(d, "s.sock")
    rows = []
    try:
        script = os.path.join(d, "bench.py")
        with open(script, "w") as f:
            f.write(PY_BENCH)
        c = MarkerCollector(path).start()
        try:
            out = subprocess.run([sys.executable, script, path, str(n)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, timeout=300)
        finally:
            time.sleep(0.3)
            _, st = c.stop()
        if out.returncode != 0 or len(out.stdout.split()) < 2:
            return [_row("python_marker_cost", FAIL, None, "us", LIMITS["python_marker_us"],
                         "benchmark process failed: " + out.stderr.strip()[-300:])]
        span_ns, raw_ns = (float(x) for x in out.stdout.split())
        cost_us, raw_us = span_ns / 1000.0, raw_ns / 1000.0
        rows.append(_row("python_marker_cost", PASS if cost_us <= LIMITS["python_marker_us"]
                         else WARN, round(cost_us, 2), "us", LIMITS["python_marker_us"],
                         f"median cost of one `with mark.span(...)` in a separate process; "
                         f"of which the kernel's sendto() alone is {raw_us:.2f} us on this "
                         f"machine"))
        # the raw datagrams are a separate sender ('raw'); judge only the marker sender
        senders = [x for x in st["senders"] if x["src"] != "raw"]
        received = sum(x["received"] for x in senders)
        lost = sum(x["lost"] for x in senders)
        rows.append(_row("marker_delivery", PASS if lost == 0 and received == n else FAIL,
                         lost, "lost", 0, f"Python, {n} spans at a realistic rate: received "
                         f"{received}, lost {lost}"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return rows


C_BENCH = r"""
#include "edgelens_mark.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(int argc, char **argv) {
    long n = atol(argv[1]); long pause_every = atol(argv[2]);
    uint64_t total = 0;
    for (long i = 0; i < n; i++) {
        uint64_t t0 = el_now_ns();
        el_span_t s = el_begin("selftest"); el_end(s);
        total += el_now_ns() - t0;
        if (pause_every && i % pause_every == pause_every - 1) {
            struct timespec ts = {0, 200000}; nanosleep(&ts, 0);
        }
    }
    uint64_t sent, dropped; el_stats(&sent, &dropped);
    /* baseline: the kernel's sendto() alone, same message size */
    char msg[] = "EL1 X 0 0 0 0 0 1 - raw-baseline-not-a-span";
    uint64_t raw = 0; long m = n < 2000 ? n : 2000;
    for (long i = 0; i < m; i++) {
        uint64_t t0 = el_now_ns();
        sendto(el__fd, msg, 5, MSG_DONTWAIT, (const struct sockaddr *)&el__addr, el__addrlen);
        raw += el_now_ns() - t0;
        if (i % 8 == 7) { struct timespec ts = {0, 200000}; nanosleep(&ts, 0); }
    }
    printf("%.1f %llu %llu %.1f\n", (double)total / n, (unsigned long long)sent,
           (unsigned long long)dropped, (double)raw / m);
    return 0;
}
"""


def check_c_markers(n):
    from .markers import INCLUDE_DIR
    from .markers.collector import MarkerCollector
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return [_row("c_marker_cost", SKIP, None, "us", LIMITS["c_marker_us"],
                     "no C compiler found (install gcc to check the C header)")]
    d = tempfile.mkdtemp(prefix="edgelens-selftest-")
    rows = []
    try:
        src = os.path.join(d, "bench.c")
        exe = os.path.join(d, "bench")
        with open(src, "w") as f:
            f.write(C_BENCH)
        r = subprocess.run([cc, "-O2", "-I", INCLUDE_DIR, src, "-o", exe],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        if r.returncode != 0:
            return [_row("c_marker_cost", FAIL, None, "us", LIMITS["c_marker_us"],
                         "edgelens_mark.h failed to compile: " + r.stderr.strip()[:300])]
        path = os.path.join(d, "s.sock")
        env = dict(os.environ, EDGELENS_SOCKET=path)
        for label, pause in (("realistic rate", 8), ("burst", 0)):
            c = MarkerCollector(path).start()
            try:
                proc = subprocess.run([exe, str(n), str(pause)], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      universal_newlines=True, timeout=120)
            except subprocess.TimeoutExpired:
                proc = None
            finally:
                time.sleep(0.3)
                _, st = c.stop()
            out = proc.stdout.split() if proc is not None and proc.returncode == 0 else []
            if len(out) < 4:
                why = "timed out" if proc is None else \
                    f"exit {proc.returncode}: {(proc.stderr or '').strip()[-200:]}"
                rows.append(_row("c_marker_cost", FAIL, None, "us", LIMITS["c_marker_us"],
                                 f"C benchmark ({label}) failed: {why}"))
                break
            cost_ns = float(out[0])
            if label == "realistic rate":
                cost_us = cost_ns / 1000.0
                raw_us = float(out[3]) / 1000.0
                rows.append(_row("c_marker_cost", PASS if cost_us <= LIMITS["c_marker_us"]
                                 else WARN, round(cost_us, 3), "us", LIMITS["c_marker_us"],
                                 f"mean cost of one C span (el_begin + el_end); of which the "
                                 f"kernel's sendto() alone is {raw_us:.2f} us on this machine"))
                rows.append(_row("marker_delivery", PASS if st["lost"] == 0 else FAIL,
                                 st["lost"], "lost", 0,
                                 f"C, {n} spans at a realistic rate: received "
                                 f"{st['received']}, lost {st['lost']}"))
            else:
                exact = st["received"] + st["lost"] == n
                rows.append(_row("marker_accounting", PASS if exact else FAIL,
                                 st["lost"], "lost", None,
                                 f"C burst of {n} spans as fast as possible: received "
                                 f"{st['received']} + counted lost {st['lost']} "
                                 f"{'=' if exact else '!='} {n} produced"
                                 f"{'' if exact else ' — spans went missing uncounted'}"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return rows


def check_dgram_queue():
    try:
        with open("/proc/sys/net/unix/max_dgram_qlen") as f:
            q = int(f.read().strip())
    except (OSError, ValueError):
        return [_row("dgram_queue", SKIP, None, "messages", LIMITS["min_dgram_qlen"],
                     "not a Linux host")]
    ok = q >= LIMITS["min_dgram_qlen"]
    return [_row("dgram_queue", PASS if ok else WARN, q, "messages", LIMITS["min_dgram_qlen"],
                 "net.unix.max_dgram_qlen: how many marker datagrams can wait for the "
                 "collector. Senders batch on overflow, but a deeper queue absorbs bursts "
                 "without batching.",
                 fix=None if ok else "sudo sysctl -w net.unix.max_dgram_qlen=512")]


def check_sampler(seconds):
    from .hardware.telemetry import TelemetryRecorder
    rec = TelemetryRecorder(interval_s=0.1)
    rec.start()
    time.sleep(seconds)
    s = rec.stop()
    cpu_ms = s.get("sampler_cpu_ms_mean") or 0.0
    pct = cpu_ms / 100.0 * 100.0              # ms CPU per 100 ms interval -> % of a core
    return [_row("sampler_cpu", PASS if pct <= LIMITS["sampler_cpu_pct_of_core"] else WARN,
                 round(pct, 2), "% of one core", LIMITS["sampler_cpu_pct_of_core"],
                 f"{cpu_ms:.3f} ms CPU per sample at 10 samples/s "
                 f"({s.get('sample_count')} samples)")]


def check_disturbance(duration_ms, iterations, rounds):
    """Alternate telemetry off / on, so slow drifts hit both equally."""
    off, on = [], []
    for _ in range(rounds):
        off.append(_reference_run(duration_ms, iterations, telemetry=False)["latency"]["mean"])
        on.append(_reference_run(duration_ms, iterations, telemetry=True)["latency"]["mean"])
    m_off, m_on = float(np.median(off)), float(np.median(on))
    shift = 100.0 * (m_on - m_off) / m_off
    return [_row("telemetry_disturbance",
                 PASS if abs(shift) <= LIMITS["telemetry_shift_pct"] else WARN,
                 round(shift, 2), "%", LIMITS["telemetry_shift_pct"],
                 f"{duration_ms} ms reference, mean {m_off:.4f} ms without telemetry vs "
                 f"{m_on:.4f} ms with ({rounds} alternating rounds)")]


def check_chrony():
    st = clock.chrony_status()
    if not st.get("available"):
        return [_row("chrony", WARN, None, "", None,
                     f"{st.get('reason')}. Only needed for latency across machines "
                     f"(e.g. sensor on the PMC, prediction on the PMA).",
                     fix="sudo apt install chrony   (and point both machines at the same "
                         "server)")]
    return [_row("chrony", PASS if st["synchronized"] else WARN,
                 round(st["error_bound_s"] * 1000, 3), "ms bound", None,
                 f"stratum {st['stratum']}, reference {st['reference']}, "
                 f"{'synchronised' if st['synchronized'] else 'NOT synchronised'}")]


def run(quick=False, include_c=True):
    """All checks. quick=True for CI / unit tests (seconds, not a minute)."""
    from .hardware import detector
    n_markers = 2_000 if quick else 20_000
    checks = []
    checks += check_clock()
    checks += check_harness([1.0] if quick else [1.0, 10.0], 60 if quick else 300)
    checks += check_python_markers(n_markers)
    if include_c:
        checks += check_c_markers(n_markers)
    checks += check_dgram_queue()
    checks += check_sampler(1.0 if quick else 3.0)
    checks += check_disturbance(2.0, 50 if quick else 300, 1 if quick else 3)
    checks += check_chrony()
    try:
        board = (detector.detect_board() or {}).get("model")
    except Exception:
        board = None
    statuses = {c["status"] for c in checks}
    return {
        "edgelens_version": __version__,
        "python": sys.version.split()[0],
        "is_jetson": detector.is_jetson(),
        "board": board,
        "quick": quick,
        "limits": LIMITS,
        "checks": checks,
        "verdict": FAIL if FAIL in statuses else (WARN if WARN in statuses else PASS),
    }

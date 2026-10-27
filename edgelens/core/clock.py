"""
edgelens.core.clock
---------------------
Time integrity: every timestamp EdgeLens records — harness stages, observer
stages, markers from other processes and languages, telemetry samples, ROS
message arrivals — lives on ONE clock, CLOCK_MONOTONIC, in nanoseconds.

Why one clock:
  * Wall time (time.time) jumps when NTP/chrony corrects it, so a latency
    computed from wall time can be negative or wrong by the correction.
  * CLOCK_MONOTONIC is what C/C++ code gets from clock_gettime() and what
    Python's time.monotonic_ns() and (on Linux) time.perf_counter_ns() read,
    so a span measured inside a C++ process and a telemetry sample measured
    by EdgeLens line up without any conversion.

Other time bases are converted EXPLICITLY, with the uncertainty of the
conversion kept next to the value:
  * wall / ROS time -> monotonic: offset estimated from back-to-back reads;
    the uncertainty is the read bracket (typically < 1 us).
  * another machine's clock: only chrony knows how far apart two machines
    are. `chrony_status()` reads chrony's own error estimate; without it a
    cross-machine latency is reported as INCONCLUSIVE, never as a number.
"""

import shutil
import subprocess
import sys
import time

CLOCK_NAME = "CLOCK_MONOTONIC"


def now_ns():
    """The one clock: CLOCK_MONOTONIC in nanoseconds."""
    return time.monotonic_ns()


def resolution_ns():
    try:
        return int(time.clock_getres(time.CLOCK_MONOTONIC) * 1e9) or 1
    except (AttributeError, OSError):     # non-POSIX
        return None


def perf_counter_is_monotonic(samples=5, tolerance_ns=50_000):
    """True when time.perf_counter_ns() reads CLOCK_MONOTONIC (Linux), so the
    harness's perf_counter timestamps are on the same clock as markers."""
    if not sys.platform.startswith("linux"):
        return False
    worst = 0
    for _ in range(samples):
        a = time.monotonic_ns()
        p = time.perf_counter_ns()
        b = time.monotonic_ns()
        worst = max(worst, abs(p - (a + b) // 2) - (b - a) // 2)
    return worst <= tolerance_ns


def read_cost_ns(n=20000):
    """Median cost of one clock read in this process (ns)."""
    vals = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(n):
            time.monotonic_ns()
        vals.append((time.perf_counter_ns() - t0) / n)
    vals.sort()
    return round(vals[len(vals) // 2], 1)


def wall_offset(samples=21):
    """Estimate wall_ns - mono_ns. Returns (offset_ns, uncertainty_ns).

    Reads mono, wall, mono back to back and keeps the tightest bracket: the
    wall read happened somewhere inside it, so half its width bounds the
    error of the estimate."""
    best = None
    for _ in range(samples):
        m0 = time.monotonic_ns()
        w = time.time_ns()
        m1 = time.monotonic_ns()
        width = m1 - m0
        if best is None or width < best[1]:
            best = (w - (m0 + m1) // 2, width)
    return best[0], max(1, best[1] // 2)


class ClockMap:
    """Converts wall/ROS-time nanoseconds to the monotonic clock, keeping
    the uncertainty of the conversion. Re-estimate (refresh) on long runs:
    chrony slews the wall clock, so the offset drifts slowly."""

    def __init__(self):
        self.refresh()

    def refresh(self):
        self.offset_ns, self.uncertainty_ns = wall_offset()
        self.estimated_at_mono_ns = now_ns()
        return self

    def wall_to_mono(self, wall_ns):
        return wall_ns - self.offset_ns

    def mono_to_wall(self, mono_ns):
        return mono_ns + self.offset_ns

    def describe(self):
        return {"wall_minus_mono_ns": self.offset_ns,
                "conversion_uncertainty_ns": self.uncertainty_ns,
                "estimated_at_mono_ns": self.estimated_at_mono_ns}


# ---------------------------------------------------------------------------
# chrony: the only honest source for "how far apart are two machines' clocks"

# `chronyc -c tracking` CSV fields (chrony 3.x / 4.x):
#  0 ref id, 1 ref name/ip, 2 stratum, 3 ref time, 4 system time offset (s,
#  + = local fast), 5 last offset, 6 rms offset, 7 frequency ppm, 8 residual
#  freq, 9 skew, 10 root delay (s), 11 root dispersion (s), 12 update
#  interval, 13 leap status
def parse_chrony_tracking_csv(line):
    parts = line.strip().split(",")
    if len(parts) < 14:
        return None
    try:
        offset = float(parts[4])
        root_delay = float(parts[10])
        root_disp = float(parts[11])
        stratum = int(parts[2])
    except ValueError:
        return None
    leap = parts[13].strip()
    synced = leap.lower() != "not synchronised" and stratum < 16
    # chrony's documented maximum error of the local clock against true time:
    # |offset| + root_dispersion + root_delay / 2.
    bound = abs(offset) + root_disp + root_delay / 2.0
    return {
        "available": True,
        "synchronized": synced,
        "reference": parts[1],
        "stratum": stratum,
        "offset_s": offset,
        "root_delay_s": root_delay,
        "root_dispersion_s": root_disp,
        "error_bound_s": round(bound, 9),
        "leap_status": leap,
    }


def chrony_status(timeout_s=1.5):
    """chrony's own estimate of this machine's clock error.

    Two machines both synced to the same reference are at most
    bound_A + bound_B apart; that sum is the uncertainty of any latency
    computed from timestamps taken on different machines."""
    exe = shutil.which("chronyc")
    if not exe:
        return {"available": False, "reason": "chronyc not found (chrony not installed)"}
    try:
        out = subprocess.run([exe, "-c", "tracking"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True,
                             timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"available": False, "reason": f"chronyc failed: {e}"}
    if out.returncode != 0:
        return {"available": False,
                "reason": (out.stderr.strip() or "chronyc returned an error")[:200]}
    parsed = parse_chrony_tracking_csv(out.stdout.splitlines()[0] if out.stdout else "")
    return parsed or {"available": False, "reason": "unrecognised chronyc output"}


def cross_machine_bound_s(local, remote_bound_s=None):
    """Uncertainty (s) of comparing a local timestamp with one from another
    machine, or None when it can't be known (then: INCONCLUSIVE)."""
    if not local or not local.get("available") or not local.get("synchronized"):
        return None
    if remote_bound_s is None:
        return None
    return local["error_bound_s"] + remote_bound_s


def describe(include_chrony=False):
    """The `clock` block stored in every result document."""
    out = {
        "clock": CLOCK_NAME,
        "resolution_ns": resolution_ns(),
        "perf_counter_is_monotonic": perf_counter_is_monotonic(),
        "wall": ClockMap().describe(),
    }
    if include_chrony:
        out["chrony"] = chrony_status()
    return out


def format_uncertainty_ms(bound_s):
    return None if bound_s is None else round(bound_s * 1000.0, 3)


__all__ = ["CLOCK_NAME", "now_ns", "resolution_ns", "perf_counter_is_monotonic",
           "read_cost_ns", "wall_offset", "ClockMap", "parse_chrony_tracking_csv",
           "chrony_status", "cross_machine_bound_s", "describe"]

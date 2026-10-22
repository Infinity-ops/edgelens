"""
edgelens.soak
---------------
Long runs (30 minutes, hours): what a 300-iteration benchmark can't show.

    edgelens soak --duration 30m --pipeline dem2.py --period-ms 1000
    edgelens soak --duration 30m -- ./fortis --config dem2.yaml     (markers + process)

The run is cut into windows (default 60 s). After every window one JSON line
is appended to <save>.jsonl and flushed to disk, so a crash, a power cut or
Ctrl-C loses at most one window; `edgelens soak --report run.jsonl` rebuilds
the analysis from that file alone.

Per window: latency (p50/p99/max, deadline misses), power and energy, peak
temperature, CPU/GPU clock frequencies (lowest seen), thermal cooling
devices that were active, process memory, system available memory, swap,
and on Jetson the GPU's share of the unified memory (NvMapMemUsed).

Analysis over the windows (the first window is warm-up and left out of
trends when there are enough windows):
  drift       p50 / p99 trend: linear fit over time, change start -> end,
              significance of the slope (t-test); "drift" only when the
              slope is significant AND the change is >= DRIFT_PCT.
  throttling  a window is throttled when a thermal cooling device that limits
              frequency was active (not fans, Tegra voltage floors or VTS —
              see NOT_THROTTLING), or the ALLOWED maximum CPU/GPU clock (the cap thermal
              management lowers) fell below 95% of the first window's — with
              the evidence listed. Current clocks are not used: they are low
              whenever the CPU idles between inferences.
  memory      process RSS growth (MB/h), unified-memory (NvMap) growth,
              system available memory trend and the projected time until it
              runs out; swap growth.
"""

import glob
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from .core.clock import now_ns

DEFAULT_WINDOW_S = 60.0
DRIFT_PCT = 10.0
LEAK_MIN_MB_PER_H = 5.0
# A slope alone is not a leak: a healthy pipeline's RSS rises in small steps
# while the allocator and caches settle, and over a short soak that steady
# rise is "significant". Found on an AGX Orin: a 5-minute soak of a pipeline
# without a leak grew 0.9 MB (75.5 -> 76.4 MB) and was reported as
# "+11.64 MB/h LEAK SUSPECTED". A leak must also have GROWN by this much
# over the analysed windows (and by LEAK_MIN_GROWTH_PCT of the process):
LEAK_MIN_GROWTH_MB = 8.0
LEAK_MIN_GROWTH_PCT = 5.0
MAX_KEEP_SAMPLES = 2_000_000


def parse_duration(text):
    """'30m', '2h', '90s', '1h30m', '45' (seconds) -> seconds."""
    import re
    t = str(text).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return float(t)
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", t)
    if not parts or "".join(a + b for a, b in parts) != t.replace(" ", ""):
        raise ValueError(f"Can't read duration '{text}': use e.g. 30m, 2h, 90s, 1h30m.")
    return sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in parts)


# ---------------------------------------------------------------- system probe

def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _int(path):
    v = _read(path)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def meminfo():
    """MemAvailable, swap and (Jetson) NvMap usage in MB, from /proc/meminfo."""
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts:
                    out[k.strip()] = int(parts[0]) / 1024.0     # kB -> MB
    except OSError:
        return {}
    res = {"mem_available_mb": out.get("MemAvailable"),
           "swap_used_mb": (out["SwapTotal"] - out["SwapFree"])
           if "SwapTotal" in out and "SwapFree" in out else None}
    if "NvMapMemUsed" in out:
        res["nvmap_used_mb"] = out["NvMapMemUsed"]
    return res


# Cooling devices that are NOT frequency throttling (third Jetson Nano run:
# CORE-floor, DFLL-floor and vdd-gpu-vts showed state 1-2 at 50 C in every
# window). Tegra "floor" devices raise the MINIMUM voltage when the chip is cold;
# "vts" (voltage-temperature scaling) picks the voltage table for the current
# temperature band. Their state is normally non-zero; neither lowers a clock.
# Fans and PWM fans are cooling, not throttling. Anything else still counts,
# and a lowered clock cap counts regardless of the device names.
NOT_THROTTLING = ("fan", "pwm", "floor", "vts")


def throttling_cooling_device(name):
    n = (name or "").lower()
    return not any(k in n for k in NOT_THROTTLING)


class SystemProbe:
    """Clock frequencies and cooling devices, sampled once per second on a
    background thread (cheap sysfs reads); summarized per window."""

    def __init__(self, interval_s=1.0):
        self.interval_s = interval_s
        self.cpu_cur = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))
        self.cpu_max = [p.replace("scaling_cur_freq", "scaling_max_freq") for p in self.cpu_cur]
        self.gpu = self._find_gpu_devfreq()
        self.cooling, self.ignored_cooling = [], []
        for d in sorted(glob.glob("/sys/class/thermal/cooling_device*")):
            typ = _read(f"{d}/type") or ""
            if not throttling_cooling_device(typ):   # fans, voltage floors, VTS
                self.ignored_cooling.append(typ)
                continue
            if _int(f"{d}/max_state") in (None, 0):
                continue
            self.cooling.append((typ, f"{d}/cur_state"))
        self._lock = threading.Lock()
        self._reset()
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _find_gpu_devfreq():
        for d in sorted(glob.glob("/sys/class/devfreq/*")):
            name = os.path.basename(d).lower()
            dev = os.path.realpath(d).lower()
            if any(k in name or k in dev for k in ("gpu", "gv11b", "ga10b", "gp10b", "gm20b")):
                return d
        return None

    def _reset(self):
        self._w = {"cpu_min": None, "cpu_sum": 0.0, "n": 0, "gpu_min": None, "cooling": {},
                   "cpu_cap": None, "gpu_cap": None}

    def read(self):
        cpu = [v for v in (_int(p) for p in self.cpu_cur) if v]
        snap = {"cpu_freq_mhz": (sum(cpu) / len(cpu) / 1000.0) if cpu else None}
        # Throttling LOWERS the allowed maximum (cpufreq scaling_max_freq,
        # devfreq max_freq); the current clock is low whenever the CPU idles.
        caps = [v for v in (_int(p) for p in self.cpu_max) if v]
        snap["cpu_cap_mhz"] = min(caps) / 1000.0 if caps else None
        if self.gpu:
            g = _int(f"{self.gpu}/cur_freq")
            snap["gpu_freq_mhz"] = g / 1e6 if g else None
            gm = _int(f"{self.gpu}/max_freq")
            snap["gpu_cap_mhz"] = gm / 1e6 if gm else None
        snap["cooling"] = {t: _int(p) for t, p in self.cooling}
        return snap

    def limits(self):
        cpu = [v for v in (_int(p) for p in self.cpu_max) if v]
        out = {"cpu_freq_mhz_max": max(cpu) / 1000.0 if cpu else None}
        if self.gpu:
            g = _int(f"{self.gpu}/max_freq")
            out["gpu_freq_mhz_max"] = g / 1e6 if g else None
            out["gpu_devfreq"] = self.gpu
        out["cooling_devices"] = [t for t, _ in self.cooling]
        out["cooling_devices_ignored"] = sorted(set(self.ignored_cooling))
        return out

    def _run(self):
        while not self._stop.is_set():
            s = self.read()
            with self._lock:
                w = self._w
                if s["cpu_freq_mhz"]:
                    w["cpu_min"] = s["cpu_freq_mhz"] if w["cpu_min"] is None else \
                        min(w["cpu_min"], s["cpu_freq_mhz"])
                    w["cpu_sum"] += s["cpu_freq_mhz"]
                    w["n"] += 1
                if s.get("gpu_freq_mhz"):
                    w["gpu_min"] = s["gpu_freq_mhz"] if w["gpu_min"] is None else \
                        min(w["gpu_min"], s["gpu_freq_mhz"])
                for t, v in s["cooling"].items():
                    if v:
                        w["cooling"][t] = max(v, w["cooling"].get(t, 0))
                for k in ("cpu", "gpu"):
                    cap = s.get(f"{k}_cap_mhz")
                    if cap:
                        w[f"{k}_cap"] = cap if w[f"{k}_cap"] is None else min(cap, w[f"{k}_cap"])
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="edgelens-probe", daemon=True)
        self._thread.start()
        return self

    def window(self):
        """Summary since the last call, then reset."""
        with self._lock:
            w, n = self._w, self._w["n"]
            out = {"cpu_freq_mhz_min": round(w["cpu_min"], 1) if w["cpu_min"] else None,
                   "cpu_freq_mhz_mean": round(w["cpu_sum"] / n, 1) if n else None,
                   "gpu_freq_mhz_min": round(w["gpu_min"], 1) if w["gpu_min"] else None,
                   "cpu_cap_mhz_min": round(w["cpu_cap"], 1) if w["cpu_cap"] else None,
                   "gpu_cap_mhz_min": round(w["gpu_cap"], 1) if w["gpu_cap"] else None,
                   "cooling_active": dict(w["cooling"])}
            self._reset()
        return out

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


# ---------------------------------------------------------------- windows

class WindowLog:
    """Appends one JSON line per window and flushes it to disk."""

    def __init__(self, path, header):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        self.write(dict(header, kind="header"))

    def write(self, obj):
        self._f.write(json.dumps(obj) + "\n")
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError:
            pass

    def close(self):
        self._f.close()


def process_memory(pid):
    """{"rss_anon_mb", "rss_file_mb"} from /proc/<pid>/status (Linux), or {}.

    RSS counts pages of memory-mapped FILES too. Found on an AGX Orin: a soak
    that read a 146 MB memory-mapped input file one 60 KB window per second
    grew RSS by 107 MB in 30 minutes and was reported as a leak; those pages
    are file cache. The leak check uses private (anonymous) memory."""
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                key, _, val = line.partition(":")
                if key in ("RssAnon", "RssFile"):
                    out["rss_anon_mb" if key == "RssAnon" else "rss_file_mb"] = \
                        round(int(val.split()[0]) / 1024.0, 2)
    except (OSError, ValueError, IndexError):
        return {}
    return out


def window_row(i, t0_ns, t1_ns, window_result, telemetry, probe_w, mem, rss_mb, split=None):
    """One window, from the window's FULL statistics (a window result's
    latency_series is capped; its latency/deadline blocks cover every sample)."""
    st = window_result.get("latency") or {"n": 0}
    n = window_result.get("iterations") or 0
    dl = window_result.get("deadline")
    pw = (telemetry or {}).get("power") or {}
    row = {"kind": "window", "window": i, "t_start_s": round(t0_ns / 1e9, 3),
           "t_end_s": round(t1_ns / 1e9, 3), "iterations": n,
           "p50_ms": st.get("p50"), "p99_ms": st.get("p99"), "max_ms": st.get("max"),
           "mean_ms": st.get("mean"), "misses": dl["misses"] if dl else None,
           "power_w_mean": pw.get("power_w_mean") if pw.get("available") else None,
           "energy_j": pw.get("energy_j") if pw.get("available") else None,
           "max_temp_c": (telemetry or {}).get("max_temp_c") or None,
           "cpu_percent_mean": (telemetry or {}).get("cpu_percent_mean"),
           "gpu_percent_mean": (telemetry or {}).get("gpu_percent_mean"),
           "background_cpu_pct": ((telemetry or {}).get("background") or {}).get("cpu_pct"),
           "start_temp_c": (telemetry or {}).get("start_temp_c"),
           "rss_mb": rss_mb}
    row.update(split or {})                  # rss_anon_mb / rss_file_mb when known
    if row["energy_j"] is not None and n:
        row["energy_per_iteration_mj"] = round(row["energy_j"] * 1000.0 / n, 3)
    row.update(probe_w or {})
    row.update(mem or {})
    return row


# ---------------------------------------------------------------- analysis

def _fit(xs, ys):
    """(slope, intercept, significant) — least squares with a t-test on the
    slope (95%, two-sided)."""
    from .compare.stats import t975
    n = len(xs)
    if n < 4:
        return None, None, False
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return None, None, False
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = (resid ** 2).sum() / (n - 2)
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    if se == 0:
        return b, a, b != 0
    return b, a, abs(b / se) > t975(n - 2)


def _trend(rows, key, warmup_skip):
    pts = [(r["t_end_s"] / 3600.0, r[key]) for r in rows[warmup_skip:]
           if r.get(key) is not None and (r.get("iterations", 1) or key not in ("p50_ms", "p99_ms"))]
    if len(pts) < 4:
        return {"windows": len(pts), "change_pct": None, "significant": False}
    xs, ys = zip(*pts)
    b, a, sig = _fit(xs, ys)
    start, end = a + b * xs[0], a + b * xs[-1]
    change = (end - start) / start * 100.0 if start else None
    return {"windows": len(pts), "start": round(start, 4), "end": round(end, 4),
            "slope_per_h": round(b, 4), "change_pct": round(change, 2) if change is not None else None,
            "significant": bool(sig)}


LEVEL_RATIO = 2.0      # two latency levels at least this far apart (high/low p50)


def latency_levels(rows, skip=0):
    """Windows whose p50 falls into two clearly separate levels, or None.

    Drift is a trend; this is a switch. Found on an AGX Orin: in a 30-minute
    soak at one window per second, 22 one-minute windows ran at p50 ~7 ms and
    8 ran entirely at ~31 ms, scattered through the run; the trend test only
    said "+39%, not significant". Splits at the largest gap between sorted
    window p50s; reports the CPU clock of each level when it was recorded."""
    pts = [(r["window"], r["p50_ms"], r.get("cpu_freq_mhz_mean"))
           for r in rows[skip:] if r.get("p50_ms")]
    if len(pts) < 6:
        return None
    vals = sorted(p for _, p, _ in pts)
    ratios = [vals[k + 1] / vals[k] for k in range(len(vals) - 1) if vals[k] > 0]
    if not ratios:
        return None
    k = int(np.argmax(ratios))
    if ratios[k] < LEVEL_RATIO:
        return None
    cut = vals[k]
    low = [p for p in pts if p[1] <= cut]
    high = [p for p in pts if p[1] > cut]
    if len(low) < 2 or len(high) < 2:
        return None              # one or two outliers are a tail, not a level

    def freq(group):
        f = [x[2] for x in group if x[2]]
        return round(float(np.mean(f)), 1) if f else None
    switches = sum(1 for a, b in zip(pts, pts[1:]) if (a[1] > cut) != (b[1] > cut))
    return {"low_p50_ms": round(float(np.median([p[1] for p in low])), 3),
            "high_p50_ms": round(float(np.median([p[1] for p in high])), 3),
            "windows_low": len(low), "windows_high": len(high),
            "high_windows": [w for w, _, _ in high], "switches": switches,
            "cpu_freq_mhz_low_level": freq(low), "cpu_freq_mhz_high_level": freq(high)}


def analyze(rows, limits=None):
    rows = [r for r in rows if r.get("kind") == "window"]
    skip = 1 if len(rows) >= 6 else 0
    out = {"windows": len(rows), "warmup_windows_excluded": skip,
           "duration_s": round(rows[-1]["t_end_s"] - rows[0]["t_start_s"], 1) if rows else 0}
    drift = {k: _trend(rows, f"{k}_ms", skip) for k in ("p50", "p99")}
    for d in drift.values():
        d["drifting"] = bool(d["significant"] and d["change_pct"] is not None
                             and abs(d["change_pct"]) >= DRIFT_PCT)
    out["drift"] = drift
    out["latency_levels"] = latency_levels(rows, skip)
    short = [r.get("iterations") for r in rows[skip:] if r.get("iterations")]
    if short and drift["p99"].get("change_pct") is None and min(short) < 100:
        drift["p99"]["note"] = (f"p99 needs >= 100 iterations per window; these windows had "
                                f"{min(short)}-{max(short)}. Use a longer --window "
                                f"(e.g. --window 120 at 1 Hz).")
    temps = [r["max_temp_c"] for r in rows if r.get("max_temp_c")]
    out["max_temp_c"] = max(temps) if temps else None

    # throttling: a non-fan cooling device active, or the ALLOWED maximum
    # clock (the cap thermal management lowers) below the first window's.
    # Current clocks are not used: an idle CPU between 1 Hz inferences runs
    # at a low clock without any throttling.
    flagged = []
    base = rows[0] if rows else {}
    for r in rows:
        why = []
        # filtered here too, so `soak --report` of an older .jsonl is right
        active = {k: v for k, v in (r.get("cooling_active") or {}).items()
                  if throttling_cooling_device(k)}
        if active:
            why.append("cooling device active: " + ", ".join(
                f"{k} (state {v})" for k, v in active.items()))
        for k, label in (("cpu", "CPU"), ("gpu", "GPU")):
            cap, cap0 = r.get(f"{k}_cap_mhz_min"), base.get(f"{k}_cap_mhz_min")
            if r is not base and cap and cap0 and cap < 0.95 * cap0:
                why.append(f"{label} maximum clock lowered to {cap} MHz (window 0: {cap0} MHz)"
                           + (f" at {r['max_temp_c']}C" if r.get("max_temp_c") else ""))
        if why:
            flagged.append({"window": r["window"], "reasons": why})
    out["throttling"] = {"throttled_windows": len(flagged), "windows": flagged[:50],
                         "limits": limits or {}}

    # memory
    mem = {}
    for key, name in (("rss_mb", "process_rss_mb_per_h"), ("rss_anon_mb", "process_anon_mb_per_h"),
                      ("rss_file_mb", "process_file_mb_per_h"),
                      ("nvmap_used_mb", "gpu_nvmap_mb_per_h"),
                      ("mem_available_mb", "available_mb_per_h"), ("swap_used_mb", "swap_mb_per_h")):
        t = _trend(rows, key, skip)
        mem[name] = t.get("slope_per_h")
        mem[name + "_significant"] = t["significant"]
        if t.get("start") is not None and t.get("end") is not None:
            # growth along the fitted trend over the analysed windows
            mem[name.replace("_per_h", "_grown")] = round(float(t["end"] - t["start"]), 2)
            mem[name.replace("_per_h", "_trend_start")] = round(float(t["start"]), 2)
    last = rows[-1] if rows else {}
    mem["rss_mb_last"] = last.get("rss_mb")
    mem["mem_available_mb_last"] = last.get("mem_available_mb")
    mem["swap_used_mb_last"] = last.get("swap_used_mb")
    mem["nvmap_used_mb_last"] = last.get("nvmap_used_mb")
    av_slope = mem.get("available_mb_per_h")
    if av_slope is not None and av_slope < 0 and mem["available_mb_per_h_significant"] \
            and last.get("mem_available_mb"):
        mem["hours_until_out_of_memory"] = round(last["mem_available_mb"] / -av_slope, 2)
    else:
        mem["hours_until_out_of_memory"] = None
    # private memory when the rows have it (file-backed pages are cache, not a leak)
    basis = "anon" if mem.get("process_anon_mb_per_h") is not None else "rss"
    mem["leak_basis"] = ("private (anonymous) memory" if basis == "anon" else
                         "resident memory (RSS; includes memory-mapped files)")
    rss = mem.get(f"process_{basis}_mb_per_h")
    mem["leak_mb_per_h"] = rss
    last_level = last.get("rss_anon_mb" if basis == "anon" else "rss_mb") or 0
    rising = bool(rss is not None and mem[f"process_{basis}_mb_per_h_significant"]
                  and rss >= max(LEAK_MIN_MB_PER_H, 0.01 * last_level))
    grown = mem.get(f"process_{basis}_mb_grown") or 0.0
    base = mem.get(f"process_{basis}_mb_trend_start") or 0.0
    mem["leak_grown_mb"] = grown
    fgrow = mem.get("process_file_mb_grown") or 0.0
    mem["file_backed_growth"] = bool(basis == "anon" and mem.get("process_file_mb_per_h_significant")
                                     and fgrow >= LEAK_MIN_GROWTH_MB)
    enough = grown >= max(LEAK_MIN_GROWTH_MB, LEAK_MIN_GROWTH_PCT / 100.0 * base)
    mem["leak_suspected"] = bool(rising and enough)
    # rising, but by too little to tell a slow leak from warm-up: say so
    mem["rss_growing_inconclusive"] = bool(rising and not enough)
    mem["leak_rule"] = (f"significant slope >= {LEAK_MIN_MB_PER_H} MB/h (and >= 1% of RSS per "
                        f"hour) AND grown >= {LEAK_MIN_GROWTH_MB} MB and >= "
                        f"{LEAK_MIN_GROWTH_PCT}% over the analysed windows")
    nv = mem.get("gpu_nvmap_mb_per_h")
    mem["gpu_memory_growth_suspected"] = bool(nv is not None
                                              and mem["gpu_nvmap_mb_per_h_significant"]
                                              and nv >= LEAK_MIN_MB_PER_H
                                              and (mem.get("gpu_nvmap_mb_grown") or 0.0)
                                              >= LEAK_MIN_GROWTH_MB)
    swap0 = next((r.get("swap_used_mb") for r in rows if r.get("swap_used_mb") is not None), None)
    mem["swap_grew_mb"] = (round(last["swap_used_mb"] - swap0, 1)
                           if swap0 is not None and last.get("swap_used_mb") is not None else None)
    out["memory"] = mem
    return out


def read_jsonl(path):
    rows, header = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue                       # a half-written last line after a crash
            if obj.get("kind") == "header":
                header = obj
            elif obj.get("kind") == "window":
                rows.append(obj)
    return header, rows


# ---------------------------------------------------------------- runners

class _Acc:
    """Every iteration is COUNTED; samples are kept for the final statistics
    up to MAX_KEEP_SAMPLES (beyond that, per-window rows in the .jsonl are the
    full record)."""

    def __init__(self):
        self.e2e, self.stages, self.kept, self.total = [], {}, True, 0

    def add(self, e2e, stages, n_total=None):
        self.total += len(e2e) if n_total is None else n_total
        if n_total is not None and n_total > len(e2e):
            self.kept = False                  # the window's series was capped
        if len(self.e2e) + len(e2e) > MAX_KEEP_SAMPLES:
            self.kept = False
            return
        self.e2e.extend(e2e)
        for k, v in stages.items():
            self.stages.setdefault(k, []).extend(v)


def _finish(acc, rows, probe_limits, *, desc, mode, source, deadline_ms, period_ms,
            telemetry_last, jsonl, extra, model_path=None):
    from .core.identity import attach_identity
    from .core.result import build_result
    soak = analyze(rows, probe_limits)
    soak.update(jsonl=str(jsonl), all_samples_kept=acc.kept,
                window_s=round(rows[0]["t_end_s"] - rows[0]["t_start_s"], 1) if rows else None)
    energies = [r["energy_j"] for r in rows if r.get("energy_j") is not None]
    result = build_result(
        pipeline_desc=desc, stage_samples_ms=acc.stages, e2e_ms=acc.e2e, mode=mode,
        pipeline_source=source, deadline_ms=deadline_ms, period_ms=period_ms,
        # an EdgeLens-driven soak with a period released one input per period
        paced=bool(period_ms) and mode == "hardware",
        telemetry_summary=telemetry_last, extra=dict(extra or {}, soak=soak),
        keep_series=len(acc.e2e) <= 200_000)
    if energies and acc.total:
        total = sum(energies)
        result["energy"] = {"available": True, "method": "sum over soak windows",
                            "energy_j": round(total, 3),
                            "energy_per_iteration_j": round(total / acc.total, 6),
                            "power_w_mean": round(float(np.mean(
                                [r["power_w_mean"] for r in rows if r.get("power_w_mean")])), 3),
                            "iterations_per_joule": round(acc.total / total, 3)
                            if total > 0 else None}
    # iterations = every iteration of the run; latency statistics come from
    # the kept samples (all of them unless all_samples_kept is False).
    result["iterations"] = acc.total
    misses = [r["misses"] for r in rows if r.get("misses") is not None]
    if result.get("deadline") is not None and misses and not acc.kept:
        result["deadline"] = dict(result["deadline"], misses=sum(misses),
                                  iterations=acc.total,
                                  miss_ratio=round(sum(misses) / acc.total, 6),
                                  met=sum(misses) == 0,
                                  note="counted over every window (.jsonl); worst/burst "
                                       "values cover the kept samples")
    tel = result.setdefault("telemetry", {})
    tel["max_temp_c"] = soak.get("max_temp_c") or tel.get("max_temp_c")
    # Run-level conditions for assumption contracts: the board at the start
    # of the first window, background load over every window.
    tel["start_temp_c"] = rows[0].get("start_temp_c") if rows else None
    bg = [(r["background_cpu_pct"], r["t_end_s"] - r["t_start_s"]) for r in rows
          if r.get("background_cpu_pct") is not None]
    if bg:
        wsum = sum(w for _, w in bg) or 1.0
        tel["background"] = {"cpu_pct": round(sum(v * w for v, w in bg) / wsum, 2),
                             "max_window_cpu_pct": max(v for v, _ in bg),
                             "windows": len(bg),
                             "method": "time-weighted mean of the soak windows"}
    else:
        tel.pop("background", None)
    attach_identity(result, model_path=model_path)
    return result


def run_soak_pipeline(pipe, duration_s, window_s=DEFAULT_WINDOW_S, period_ms=None,
                      deadline_ms=None, jsonl_path="soak.jsonl", pipeline_source=None,
                      model_path=None, progress=None, telemetry_interval_s=0.5):
    """EdgeLens drives the loop (optionally paced) for duration_s seconds."""
    from .core.harness import run_harness
    say = progress or (lambda m: None)
    probe = SystemProbe().start()
    limits = probe.limits()
    log = WindowLog(jsonl_path, {"mode": "pipeline", "source": pipeline_source,
                                 "duration_s": duration_s, "window_s": window_s,
                                 "period_ms": period_ms, "deadline_ms": deadline_ms,
                                 "limits": limits, "started_mono_ns": now_ns()})
    acc, rows, last_tel = _Acc(), [], {}
    eff_deadline = deadline_ms
    t_origin = now_ns()
    me = psutil.Process()
    i = 0
    try:
        while (now_ns() - t_origin) / 1e9 < duration_s:
            left = duration_s - (now_ns() - t_origin) / 1e9
            w = min(window_s, left)
            if w < 1.0 and rows:
                break
            t0 = now_ns()
            res = run_harness(pipe, source=getattr(pipe, "source", None),
                              iterations=10 ** 9, warmup=10 if i == 0 else 0,
                              period_ms=period_ms, pace=bool(period_ms),
                              deadline_ms=deadline_ms, telemetry_interval_s=telemetry_interval_s,
                              attach_ids=False, pipeline_source=pipeline_source, max_seconds=w)
            t1 = now_ns()
            eff_deadline = res["requirements"]["deadline_ms"]
            e2e = res.get("latency_series_ms") or []
            stages = {}
            if not (res.get("trace") or {}).get("truncated"):
                for ev in (res.get("trace") or {}).get("events") or []:
                    stages.setdefault(ev["stage"], []).append(ev["dur_ns"] / 1e6)
            acc.add(e2e, stages, n_total=res["iterations"])
            last_tel = res.get("telemetry") or {}
            row = window_row(i, t0 - t_origin, t1 - t_origin, res, last_tel, probe.window(),
                             meminfo(), round(me.memory_info().rss / 1e6, 2),
                             process_memory(os.getpid()))
            rows.append(row)
            log.write(row)
            say(_progress_line(row, duration_s))
            i += 1
    except KeyboardInterrupt:
        say("Interrupted: analysing the windows recorded so far.")
    finally:
        probe.stop()
        log.close()
    desc = pipe.describe()
    return _finish(acc, rows, limits, desc=desc, mode="hardware",
                   source=pipeline_source or f"pipeline:{pipe.name}",
                   deadline_ms=eff_deadline, period_ms=period_ms, telemetry_last=last_tel,
                   jsonl=jsonl_path,
                   extra={"observation": {"ended_by": "duration reached" if rows else "no data"}},
                   model_path=model_path)


def run_soak_observed(duration_s, window_s=DEFAULT_WINDOW_S, cmd=None, pid=None, name=None,
                      deadline_ms=None, jsonl_path="soak.jsonl", socket_path=None,
                      progress=None, telemetry_interval_s=0.5):
    """An application EdgeLens doesn't run: markers (if it sends any), its
    process tree, and the board — window by window."""
    import shutil
    import signal
    import subprocess
    import tempfile
    from .attach.procsampler import ProcessSampler, find_process
    from .hardware.telemetry import TelemetryRecorder
    from .markers import protocol
    from .markers.collector import MarkerCollector
    from .markers.session import spans_to_result

    say = progress or (lambda m: None)
    if name and not pid and not cmd:
        pid = find_process(name)
    private = None
    if cmd and socket_path is None:
        private = tempfile.mkdtemp(prefix="edgelens-")
        socket_path = os.path.join(private, "markers.sock")
    collector = MarkerCollector(socket_path, keep_spans=False)
    verdicts = {}
    if pid and not cmd:
        # shared socket: only the target's own process tree counts (as in watch)
        from .attach.runner import _ancestry_checker
        collector.add_listener(_ancestry_checker(pid, verdicts))
    collector.start()
    probe = SystemProbe().start()
    limits = probe.limits()
    child = None
    if cmd:
        env = dict(os.environ, **{protocol.ENV_SOCKET: collector.path})
        child = subprocess.Popen(cmd, env=env)
        pid = child.pid
    sampler = ProcessSampler(pid, interval_s=1.0).start() if pid else None
    excluded = [0]

    def _own(spans):
        if cmd or not pid:
            return spans                       # private socket / no target: all spans
        own = set(sampler.pids_seen if sampler else ()) | {pid} | \
            {p for p, v in verdicts.items() if v}
        kept = [sp for sp in spans if sp["pid"] in own]
        excluded[0] += len(spans) - len(kept)
        return kept
    log = WindowLog(jsonl_path, {"mode": "observed", "target": cmd or pid,
                                 "duration_s": duration_s, "window_s": window_s,
                                 "deadline_ms": deadline_ms, "limits": limits,
                                 "started_mono_ns": now_ns()})
    acc, rows, last_tel, all_spans_n = _Acc(), [], {}, 0
    t_origin = now_ns()
    i, ended_by = 0, "duration reached"
    try:
        while (now_ns() - t_origin) / 1e9 < duration_s:
            rec = TelemetryRecorder(interval_s=telemetry_interval_s)
            target = child.pid if child is not None else pid
            if target:
                rec.track_pid(target)            # the app is workload, not background
            rec.start()
            t0 = now_ns()
            end = time.monotonic() + min(window_s, duration_s - (t0 - t_origin) / 1e9)
            while time.monotonic() < end:
                if child is not None and child.poll() is not None:
                    ended_by = "command exited"
                    break
                if child is None and sampler is not None and sampler.exited_mono_ns:
                    ended_by = "process exited"
                    break
                time.sleep(0.1)
            last_tel = rec.stop()
            t1 = now_ns()
            spans = _own(collector.drain())
            all_spans_n += len(spans)
            wr = {"latency": {"n": 0}, "iterations": 0, "deadline": None}
            e2e, stages = [], {}
            if spans:
                wr = spans_to_result(spans, {}, deadline_ms=deadline_ms, attach_ids=False)
                e2e = wr.get("latency_series_ms") or []
                if not wr["trace"]["truncated"]:
                    for ev in wr["trace"]["events"]:
                        if ev["stage"] != protocol.ITERATION:
                            stages.setdefault(ev["stage"], []).append(ev["dur_ns"] / 1e6)
            acc.add(e2e, stages, n_total=wr["iterations"])
            rss = None
            if sampler is not None and sampler.raw:
                rss = round(sampler.raw[-1][2] / 1e6, 2)
            row = window_row(i, t0 - t_origin, t1 - t_origin, wr, last_tel,
                             probe.window(), meminfo(), rss,
                             process_memory(sampler.pid) if sampler is not None else None)
            rows.append(row)
            log.write(row)
            say(_progress_line(row, duration_s))
            i += 1
            if ended_by != "duration reached":
                break
    except KeyboardInterrupt:
        ended_by = "interrupted (Ctrl-C)"
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        time.sleep(0.25)
        tail = _own(collector.drain())
        if tail:                               # spans that arrived after the last window
            wr = spans_to_result(tail, {}, deadline_ms=deadline_ms, attach_ids=False)
            acc.add(wr.get("latency_series_ms") or [], {}, n_total=wr["iterations"])
        _, mstats = collector.stop()
        if sampler is not None:
            sampler.stop()
        probe.stop()
        log.close()
        if private:
            shutil.rmtree(private, ignore_errors=True)
    names = sorted(acc.stages)
    desc = {"name": "soak", "pack": "custom", "config": {},
            "stages": [{"name": n, "role": "other"} for n in names]}
    from .core.pipeline import infer_role
    for s in desc["stages"]:
        s["role"] = infer_role(s["name"])
    mstats = dict(mstats, private_socket=private is not None, foreign_spans=excluded[0])
    extra = {"markers": mstats,
             "observation": {"target": {"kind": "command" if cmd else "pid",
                                        "command": cmd, "pid": pid},
                             "ended_by": ended_by,
                             "exit_code": child.returncode if child is not None else None}}
    if sampler is not None:
        extra["process"] = sampler.summary()
    return _finish(acc, rows, limits, desc=desc, mode="markers" if acc.e2e else "attach",
                   source="command:" + " ".join(cmd) if cmd else f"pid:{pid}",
                   deadline_ms=deadline_ms, period_ms=None, telemetry_last=last_tel,
                   jsonl=jsonl_path, extra=extra)


def _progress_line(row, duration_s):
    if row["iterations"]:
        lat = f"p50 {row['p50_ms']} ms" + (f", p99 {row['p99_ms']} ms" if row.get("p99_ms")
                                           else f", max {row['max_ms']} ms")
    else:
        lat = "no iterations"
    parts = [f"{row['t_end_s']:.0f}/{duration_s:.0f} s", f"{row['iterations']} it", lat]
    if row.get("max_temp_c"):
        parts.append(f"{row['max_temp_c']}C")
    if row.get("rss_mb") is not None:
        parts.append(f"RSS {row['rss_mb']} MB")
    if any(throttling_cooling_device(k) for k in row.get("cooling_active") or {}):
        parts.append("THROTTLING DEVICE ACTIVE")
    return "Soak: " + " · ".join(parts)


def report_from_jsonl(path):
    """Rebuild the analysis from a .jsonl alone (after a crash or power cut)."""
    from .core.result import build_result
    header, rows = read_jsonl(path)
    soak = analyze(rows, header.get("limits"))
    soak.update(jsonl=str(path), rebuilt_from_jsonl=True)
    result = build_result(
        pipeline_desc={"name": "soak (rebuilt)", "pack": "custom", "config": {}, "stages": []},
        stage_samples_ms={}, e2e_ms=[], mode="soak-report", pipeline_source=f"jsonl:{path}",
        deadline_ms=header.get("deadline_ms"), extra={"soak": soak, "soak_windows": rows})
    result["telemetry"]["max_temp_c"] = soak.get("max_temp_c")
    return result

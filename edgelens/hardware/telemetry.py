"""
edgelens.hardware.telemetry
-----------------------------
Live system telemetry, sampled with as little disturbance to the workload
being measured as possible.

v0.1.0 design (replaces the per-sample subprocess approach):

* CPU% uses psutil.cpu_percent(interval=None): non-blocking, the
  utilisation since the previous call. The old interval=0.2 blocked the
  sampler for 200 ms per sample.
* GPU% is read from the GPU devfreq ``load`` node in sysfs (per-mille) when
  it exists — one small file read, no process spawn.
* tegrastats, when needed (no sysfs GPU node, or for power rails), runs as
  ONE persistent process for the whole recording, read line by line on a
  background thread. The old code spawned a new tegrastats process for every
  sample, which cost ~1 s per sample on a Jetson Nano and stole CPU from the
  benchmark it was measuring.
* Every sample is timestamped and the full series is kept (not just
  mean/peak), so telemetry can be lined up against the event trace.
* The sampler measures its own cost per sample and reports it, so the
  observer effect is visible instead of assumed to be zero.
"""

import glob
import os
import re
import shutil
import statistics
import subprocess
import threading
import time

import psutil

from . import power as power_mod
from ..core.clock import now_ns as _now_ns

# GPU load nodes, per-mille (0..1000). Paths differ across Jetson modules
# and JetPack releases; the first readable one wins.
_GPU_LOAD_GLOBS = (
    "/sys/devices/gpu.0/load",                     # Nano / TX1 / TX2 (JetPack 4.x)
    "/sys/devices/platform/gpu.0/load",            # Xavier / Orin (JetPack 5.x)
    "/sys/devices/platform/bus@0/*.gpu/load",      # Orin (JetPack 6.x)
    "/sys/devices/platform/*.gpu/load",
    "/sys/devices/*.gpu/load",
)

MAX_SERIES_SAMPLES = 20000

_gpu_load_path = None
_gpu_load_searched = False
_thermal_zones = None

psutil.cpu_percent(interval=None)  # prime: the first non-blocking call returns 0.0


def _find_gpu_load_path():
    global _gpu_load_path, _gpu_load_searched
    if not _gpu_load_searched:
        _gpu_load_searched = True
        for pattern in _GPU_LOAD_GLOBS:
            for path in sorted(glob.glob(pattern)):
                try:
                    with open(path) as f:
                        int(f.read().strip())
                    _gpu_load_path = path
                    return path
                except Exception:
                    continue
    return _gpu_load_path


def read_gpu_percent_sysfs():
    path = _find_gpu_load_path()
    if not path:
        return None
    try:
        with open(path) as f:
            return round(int(f.read().strip()) / 10.0, 1)
    except Exception:
        return None


SLOW_ZONE_MS = 1.0          # a zone read slower than this is "slow" (firmware / I2C)
SLOW_ZONE_EVERY = 20        # read slow zones only every Nth sample


def _read_zone(path):
    with open(path) as f:
        return int(f.read().strip()) / 1000.0


def _zones():
    """[(name, path, cost_ms)] — each zone's read cost measured once.

    Some thermal zones are read through firmware or I2C and take
    milliseconds (laptop Wi-Fi chips, a Jetson PMIC): read every sample, they
    made the sampler cost 7-15 ms of CPU per sample on real machines."""
    global _thermal_zones
    if _thermal_zones is None:
        zones = []
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            try:
                with open(f"{zone}/type") as f:
                    name = f.read().strip()
            except Exception:
                continue
            path = f"{zone}/temp"
            costs = []
            for _ in range(2):
                c0 = time.perf_counter()
                try:
                    _read_zone(path)
                except Exception:
                    pass
                costs.append((time.perf_counter() - c0) * 1000.0)
            zones.append((name, path, min(costs)))
        _thermal_zones = zones
    return _thermal_zones


def slow_zones():
    return [(n, round(c, 3)) for n, _, c in _zones() if c > SLOW_ZONE_MS]


_temp_cache = {}
_temp_calls = [0]


def read_temps(include_slow=None):
    """Temperatures of all plausible zones. Slow zones are re-read only every
    SLOW_ZONE_EVERY calls (their last value is reused in between)."""
    _temp_calls[0] += 1
    if include_slow is None:
        include_slow = _temp_calls[0] % SLOW_ZONE_EVERY == 1
    temps = {}
    for name, path, cost in _zones():
        if cost > SLOW_ZONE_MS and not include_slow:
            if name in _temp_cache:
                temps[name] = _temp_cache[name]
            continue
        try:
            val = _read_zone(path)
        except Exception:
            continue
        # Disabled/virtual zones report sentinel values (e.g. -40 C or 100+ C
        # on the Nano's PMIC zone); keep only physically plausible readings.
        if -30.0 < val < 130.0:
            temps[name] = val
            _temp_cache[name] = val
    return temps


def profile_sampler(n=10):
    """CPU ms per call of each sampler part (for `edgelens selftest`)."""
    pr = power_mod.PowerReader()
    parts = {
        "cpu_percent": lambda: psutil.cpu_percent(interval=None),
        "memory": psutil.virtual_memory,
        "swap": psutil.swap_memory,
        "thermal (fast zones)": lambda: read_temps(include_slow=False),
        "thermal (all zones)": lambda: read_temps(include_slow=True),
        "gpu load": read_gpu_percent_sysfs,
        "power sensors": pr.read,
    }
    out = {}
    for name, fn in parts.items():
        c0 = time.thread_time()
        for _ in range(n):
            try:
                fn()
            except Exception:
                pass
        out[name] = round((time.thread_time() - c0) / n * 1000.0, 3)
    return out


# --------------------------------------------------------------------------
# tegrastats

_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
_EMC = re.compile(r"EMC_FREQ (\d+)%")
# Power rails: "POM_5V_IN 2591/2591" (JetPack 4) or "VDD_GPU_SOC 2387mW/2387mW"
# (JetPack 5/6). Only POM_/VDD_/VIN_ names, so "RAM 2246/3964MB" never matches.
_RAIL = re.compile(r"\b((?:POM|VDD|VIN)_[A-Z0-9_]+) (\d+)(?:mW)?/(\d+)(?:mW)?")


def parse_tegrastats_line(line):
    """Parse one tegrastats line into {gpu_percent, emc_percent, rails_mw}."""
    out = {"gpu_percent": None, "emc_percent": None, "rails_mw": {}}
    m = _GR3D.search(line)
    if m:
        out["gpu_percent"] = float(m.group(1))
    m = _EMC.search(line)
    if m:
        out["emc_percent"] = float(m.group(1))
    for name, instant, _avg in _RAIL.findall(line):
        out["rails_mw"][name] = float(instant)
    return out


class TegrastatsStream:
    """One persistent tegrastats process, parsed on a background thread.

    latest() returns the most recent parsed line (or None). Safe to create
    on any host: start() returns False when tegrastats isn't installed.
    """

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self._latest = None
        self._lock = threading.Lock()

    @staticmethod
    def available():
        return shutil.which("tegrastats") is not None

    def start(self):
        if not self.available():
            return False
        cmd = ["tegrastats", "--interval", str(int(self.interval_ms))]
        if shutil.which("stdbuf"):
            # Through a pipe, tegrastats' stdout is block-buffered, so parsed
            # lines can arrive seconds late. Force line buffering.
            cmd = ["stdbuf", "-oL"] + cmd
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                universal_newlines=True, bufsize=1,
            )
        except Exception:
            self._proc = None
            return False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return True

    def _read(self):
        for line in self._proc.stdout:
            parsed = parse_tegrastats_line(line)
            parsed["timestamp"] = time.time()
            with self._lock:
                self._latest = parsed

    def latest(self):
        with self._lock:
            return dict(self._latest) if self._latest else None

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except Exception:
                self._proc.kill()
            self._proc = None


def read_gpu_percent_tegrastats():
    """Single-shot fallback (one tegrastats line). Only used by snapshot()
    when there is no sysfs load node and no running stream."""
    if not TegrastatsStream.available():
        return None
    try:
        proc = subprocess.Popen(["tegrastats", "--interval", "100"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                universal_newlines=True)
        line = proc.stdout.readline()
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()
        return parse_tegrastats_line(line)["gpu_percent"]
    except Exception:
        return None


# --------------------------------------------------------------------------
# sampling

_swap_cache = [None, 0]


def _swap():
    """Swap changes slowly and psutil reads several /proc files for it:
    refresh every 10th sample."""
    _swap_cache[1] += 1
    if _swap_cache[0] is None or _swap_cache[1] % 10 == 0:
        _swap_cache[0] = psutil.swap_memory()
    return _swap_cache[0]


_gpu_devfreq = [False]          # not looked up yet


def _gpu_freq_path():
    """devfreq cur_freq of the integrated GPU, or None (not a Jetson)."""
    if _gpu_devfreq[0] is False:
        _gpu_devfreq[0] = None
        for d in sorted(glob.glob("/sys/class/devfreq/*")):
            name = os.path.basename(d).lower()
            dev = os.path.realpath(d).lower()
            if any(k in name or k in dev for k in ("gpu", "gv11b", "ga10b", "gp10b", "gm20b")):
                _gpu_devfreq[0] = os.path.join(d, "cur_freq")
                break
    return _gpu_devfreq[0]


def read_gpu_freq_mhz():
    path = _gpu_freq_path()
    if not path:
        return None
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1e6
    except (OSError, ValueError):
        return None


def snapshot(stream=None, power_reader=None):
    """One telemetry sample. Safe on any host (Jetson or not), non-blocking."""
    t0 = time.perf_counter()
    c0 = time.thread_time()
    mem = psutil.virtual_memory()
    swap = _swap()
    teg = stream.latest() if stream is not None else None

    gpu = read_gpu_percent_sysfs()
    if gpu is None and teg is not None:
        gpu = teg.get("gpu_percent")
    if gpu is None and stream is None and power_reader is None:
        gpu = read_gpu_percent_tegrastats()   # legacy single-shot path (monitor)

    rails = None
    if power_reader is not None:
        rails = power_reader.read()
    if not rails and teg is not None and teg.get("rails_mw"):
        rails = {k: v / 1000.0 for k, v in teg["rails_mw"].items()}

    sample = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "mem_percent": mem.percent,
        "mem_used_gb": round(mem.used / 1e9, 2),
        "mem_total_gb": round(mem.total / 1e9, 2),
        "swap_used_gb": round(swap.used / 1e9, 2),
        "temps_c": read_temps(),
        "gpu_percent": gpu,
        "gpu_freq_mhz": read_gpu_freq_mhz(),
        "emc_percent": teg.get("emc_percent") if teg else None,
        "rails_w": rails or {},
        "power_w": power_mod.total_power_w(rails) if rails else None,
        "timestamp": time.time(),
        # The one clock (CLOCK_MONOTONIC) — what markers and stage timings use.
        # Series and energy integration use this, never wall time, which
        # jumps when chrony/NTP corrects it.
        "mono_ns": _now_ns(),
    }
    # Wall time includes waiting for the GIL while the workload thread runs,
    # so it overstates the sampler's real cost; CPU time is what it steals.
    sample["_cost_ms"] = (time.perf_counter() - t0) * 1000.0
    sample["_cpu_ms"] = (time.thread_time() - c0) * 1000.0
    return sample


class TelemetryRecorder:
    """Samples telemetry continuously on a background thread for the
    duration of a run and keeps the full timestamped series.

    Usage:
        rec = TelemetryRecorder(interval_s=0.1)
        rec.start()
        ... run the workload ...
        summary = rec.stop()   # mean/peak per metric, power/energy
        series = rec.series()  # timestamped samples, relative to start
    """

    def __init__(self, interval_s=0.1, use_tegrastats=None, use_power=True):
        self.interval_s = interval_s
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._stream = None
        self._use_tegrastats = use_tegrastats
        self._power = power_mod.PowerReader() if use_power else None
        self._power_denied = self._power.denied if self._power is not None else []
        self.start_time = None
        self.start_mono_ns = None
        # Background load: CPU time of everything EXCEPT this process (its
        # children included) and the processes registered with track_pid()
        # (the application under test). See background_load().
        self._own = {}                 # pid -> [first cpu s, last cpu s]
        self._sys0 = None
        self._bg = None

    def track_pid(self, pid):
        """Count this process (the application being measured) as part of
        the workload, not as background load."""
        if pid and pid not in self._own:
            self._own[pid] = [None, None]
            self._touch(pid, first=True)

    def _touch(self, pid, first=False):
        try:
            p = psutil.Process(pid)
            t = p.cpu_times()
            v = t.user + t.system + getattr(t, "children_user", 0.0) + \
                getattr(t, "children_system", 0.0)
        except (psutil.Error, OSError):
            return
        rec = self._own.setdefault(pid, [None, None])
        if first or rec[0] is None:
            rec[0] = v
        rec[1] = v

    @staticmethod
    def _sys_busy():
        t = psutil.cpu_times()
        idle = t.idle + getattr(t, "iowait", 0.0)
        total = sum(t)
        # guest time is already inside user/nice on Linux: don't count it twice
        total -= getattr(t, "guest", 0.0) + getattr(t, "guest_nice", 0.0)
        return total - idle

    def _bg_start(self):
        self._own = {pid: [None, None] for pid in self._own}
        for pid in list(self._own) + [os.getpid()]:
            self._touch(pid, first=True)
        self._sys0 = (self._sys_busy(), _now_ns())

    def _bg_stop(self):
        if self._sys0 is None:
            return None
        self._touch(os.getpid())
        for pid in list(self._own):
            self._touch(pid)
        busy0, t0 = self._sys0
        elapsed = (_now_ns() - t0) / 1e9
        if elapsed <= 0:
            return None
        sys_busy = self._sys_busy() - busy0
        own = sum((b - a) for a, b in self._own.values() if a is not None and b is not None)
        ncpu = psutil.cpu_count(logical=True) or 1
        other = max(0.0, sys_busy - own)
        return {
            "cpu_pct": round(other / (elapsed * ncpu) * 100.0, 2),     # % of the whole machine
            "cores": round(other / elapsed, 3),
            "workload_cores": round(own / elapsed, 3),
            "interval_s": round(elapsed, 2),
            "workload_pids": sorted(self._own),
            "method": ("system CPU time minus EdgeLens and the measured application "
                       "(psutil cpu_times; children included once they exit)"),
        }

    def background_load(self):
        return self._bg

    def start(self):
        self._samples = []
        self._stop_event.clear()
        # A sensor that is listed but can't be read (permissions, driver
        # quirks) must not silently disable power: verify one real read.
        if self._power is not None and self._power.available() and not self._power.read():
            self._power = None
        want_stream = self._use_tegrastats
        if want_stream is None:
            # Only needed when sysfs can't give GPU load or power directly.
            want_stream = (read_gpu_percent_sysfs() is None
                           or self._power is None or not self._power.available())
        if want_stream and TegrastatsStream.available():
            self._stream = TegrastatsStream(interval_ms=max(50, int(self.interval_s * 1000)))
            self._stream.start()
        psutil.cpu_percent(interval=None)   # reset the CPU% window to "now"
        self._bg_start()
        self.start_time = time.time()
        self.start_mono_ns = _now_ns()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                if len(self._samples) < MAX_SERIES_SAMPLES:
                    self._samples.append(snapshot(self._stream, self._power))
                if self._own and len(self._samples) % 10 == 0:
                    for pid in list(self._own):      # last value seen if it exits
                        self._touch(pid)
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # one closing sample so short runs still get a CPU% over the whole window
        try:
            self._samples.append(snapshot(self._stream, self._power))
        except Exception:
            pass
        if self._stream is not None:
            self._stream.stop()
        try:
            self._bg = self._bg_stop()
        except Exception:
            self._bg = None
        return self.summarize()

    def series(self):
        """Compact timestamped series. t_s is relative to the recording start
        on CLOCK_MONOTONIC; t_mono_ns is the absolute monotonic timestamp, so
        samples line up with marker spans from other processes."""
        t0 = self.start_mono_ns
        if t0 is None:
            t0 = self._samples[0].get("mono_ns", 0) if self._samples else 0
        out = []
        for s in self._samples:
            temps = s.get("temps_c") or {}
            mono = s.get("mono_ns")
            out.append({
                "t_s": round((mono - t0) / 1e9, 4) if mono is not None else None,
                "t_mono_ns": mono,
                "cpu_percent": s.get("cpu_percent"),
                "gpu_percent": s.get("gpu_percent"),
                "gpu_freq_mhz": s.get("gpu_freq_mhz"),
                "mem_percent": s.get("mem_percent"),
                "emc_percent": s.get("emc_percent"),
                "max_temp_c": max(temps.values()) if temps else None,
                "power_w": s.get("power_w"),
            })
            if s.get("rails_w"):
                out[-1]["rails_w"] = {k: round(v, 3) for k, v in s["rails_w"].items()}
        return out

    def summarize(self):
        if not self._samples:
            self._samples = [snapshot()]
            self._samples[0]["_single_fallback"] = True

        def column(key):
            return [s[key] for s in self._samples if s.get(key) is not None]

        def mean(values):
            return round(statistics.mean(values), 2) if values else None

        def peak(values):
            return round(max(values), 2) if values else None

        cpu_vals = column("cpu_percent")
        gpu_vals = column("gpu_percent")
        mem_vals = column("mem_percent")
        emc_vals = column("emc_percent")
        costs = column("_cost_ms")
        cpu_costs = column("_cpu_ms")

        temp_keys = set()
        for s in self._samples:
            temp_keys.update((s.get("temps_c") or {}).keys())
        temps_mean, temps_peak = {}, {}
        for k in sorted(temp_keys):
            vs = [s["temps_c"][k] for s in self._samples if k in (s.get("temps_c") or {})]
            temps_mean[k] = mean(vs)
            temps_peak[k] = peak(vs)

        n = len(self._samples)
        first, last = self._samples[0], self._samples[-1]
        if n > 1 and first.get("mono_ns") is not None and last.get("mono_ns") is not None:
            span = (last["mono_ns"] - first["mono_ns"]) / 1e9
        else:
            span = (last["timestamp"] - first["timestamp"]) if n > 1 else 0.0
        single = n == 1 and self._samples[0].get("_single_fallback")

        summary = {
            "sample_count": 0 if single else n,
            "clock": "CLOCK_MONOTONIC",
            "origin_mono_ns": self.start_mono_ns,
            # Core count lets diagnosis tell "one core saturated" from "all busy".
            "cpu_count": psutil.cpu_count(logical=True),
            # Wall-clock span from real timestamps, NOT n * interval_s.
            "duration_s": round(span, 2),
            "effective_interval_s": round(span / (n - 1), 3) if n > 1 else None,
            # Observer effect, made visible: CPU time the sampler thread spent
            # per sample, and the wall time a sample took (incl. GIL waits).
            "sampler_cpu_ms_mean": round(statistics.mean(cpu_costs), 3) if cpu_costs else None,
            "sampler_wall_ms_mean": round(statistics.mean(costs), 3) if costs else None,
            "cpu_percent_mean": mean(cpu_vals),
            "cpu_percent_peak": peak(cpu_vals),
            "gpu_percent_mean": mean(gpu_vals),
            "gpu_percent_peak": peak(gpu_vals),
            "mem_percent_mean": mean(mem_vals),
            "mem_percent_peak": peak(mem_vals),
            "emc_percent_mean": mean(emc_vals),
            "temps_c_mean": temps_mean,
            "temps_c_peak": temps_peak,
            "max_temp_c": max(temps_peak.values()) if temps_peak else 0,
            # the board's state when the run began (assumption contracts)
            "start_temp_c": (max((first.get("temps_c") or {}).values())
                             if (first.get("temps_c") or {}) else None),
            "slow_thermal_zones": slow_zones(),
            # flat/legacy fields (mean-based) consumed by older readers
            "cpu_percent": mean(cpu_vals),
            "gpu_percent": mean(gpu_vals),
            "mem_percent": mean(mem_vals),
            "temps_c": temps_mean,
        }
        reason = fix = None
        if self._power_denied:
            reason = "permission_denied"
            fix = power_mod.permission_fix_command(self._power_denied)
        summary["power"] = power_mod.summarize_power(self._samples, reason, fix)
        if self._bg is not None:
            summary["background"] = self._bg
        return summary


# --------------------------------------------------------------------------
# source probe (for `edgelens doctor`)

def _timed(fn):
    t0 = time.perf_counter()
    try:
        value, err = fn(), None
    except Exception as e:      # pragma: no cover - defensive
        value, err = None, f"{type(e).__name__}: {e}"
    return value, round((time.perf_counter() - t0) * 1000.0, 3), err


def probe_sources(tegrastats_timeout_s=3.0):
    """Which telemetry sources work on this host, their current values and
    how long one read takes. This is what `edgelens doctor` prints, and the
    first thing to look at when a metric shows up as n/a."""
    out = {}
    gpu_path = _find_gpu_load_path()
    value, ms, err = _timed(read_gpu_percent_sysfs)
    out["gpu_load_sysfs"] = {"path": gpu_path, "value_percent": value, "read_ms": ms,
                             "error": err}

    reader = power_mod.PowerReader()
    rails, ms, err = _timed(reader.read)
    total, method = power_mod.total_power(rails) if rails else (None, None)
    out["power_ina3221"] = {"rails_found": reader.rails(), "rails_w": rails or {},
                            "total_w": total, "method": method, "read_ms": ms,
                            "permission_denied": reader.permission_denied(),
                            "fix": reader.fix_command(),
                            "error": err if err else (None if rails or not reader.rails()
                                                      else "rails listed but unreadable")}

    temps, ms, err = _timed(read_temps)
    out["thermal_zones"] = {"count": len(temps), "max_c": max(temps.values()) if temps else None,
                            "read_ms": ms, "error": err}

    teg = {"available": TegrastatsStream.available(), "first_line_s": None,
           "gpu_percent": None, "rails_mw": {}}
    if teg["available"]:
        stream = TegrastatsStream(interval_ms=200)
        t0 = time.time()
        if stream.start():
            while time.time() - t0 < tegrastats_timeout_s:
                latest = stream.latest()
                if latest:
                    teg["first_line_s"] = round(time.time() - t0, 2)
                    teg["gpu_percent"] = latest.get("gpu_percent")
                    teg["rails_mw"] = latest.get("rails_mw") or {}
                    break
                time.sleep(0.05)
            stream.stop()
    out["tegrastats"] = teg
    return out

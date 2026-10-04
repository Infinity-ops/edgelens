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
import re
import shutil
import statistics
import subprocess
import threading
import time

import psutil

from . import power as power_mod

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


def _zones():
    global _thermal_zones
    if _thermal_zones is None:
        zones = []
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            try:
                with open(f"{zone}/type") as f:
                    zones.append((f.read().strip(), f"{zone}/temp"))
            except Exception:
                continue
        _thermal_zones = zones
    return _thermal_zones


def read_temps():
    temps = {}
    for name, path in _zones():
        try:
            with open(path) as f:
                val = int(f.read().strip()) / 1000.0
        except Exception:
            continue
        # Disabled/virtual zones report sentinel values (e.g. -40 C or 100+ C
        # on the Nano's PMIC zone); keep only physically plausible readings.
        if -30.0 < val < 130.0:
            temps[name] = val
    return temps


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

def snapshot(stream=None, power_reader=None):
    """One telemetry sample. Safe on any host (Jetson or not), non-blocking."""
    t0 = time.perf_counter()
    c0 = time.thread_time()
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
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
        "emc_percent": teg.get("emc_percent") if teg else None,
        "rails_w": rails or {},
        "power_w": power_mod.total_power_w(rails) if rails else None,
        "timestamp": time.time(),
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
        self.start_time = None

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
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                if len(self._samples) < MAX_SERIES_SAMPLES:
                    self._samples.append(snapshot(self._stream, self._power))
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
        return self.summarize()

    def series(self):
        """Compact timestamped series, t_s relative to the recording start."""
        t0 = self.start_time or (self._samples[0]["timestamp"] if self._samples else 0)
        out = []
        for s in self._samples:
            temps = s.get("temps_c") or {}
            out.append({
                "t_s": round(s["timestamp"] - t0, 4),
                "cpu_percent": s.get("cpu_percent"),
                "gpu_percent": s.get("gpu_percent"),
                "mem_percent": s.get("mem_percent"),
                "emc_percent": s.get("emc_percent"),
                "max_temp_c": max(temps.values()) if temps else None,
                "power_w": s.get("power_w"),
            })
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
        span = (self._samples[-1]["timestamp"] - self._samples[0]["timestamp"]) if n > 1 else 0.0
        single = n == 1 and self._samples[0].get("_single_fallback")

        summary = {
            "sample_count": 0 if single else n,
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
            # flat/legacy fields (mean-based) consumed by older readers
            "cpu_percent": mean(cpu_vals),
            "gpu_percent": mean(gpu_vals),
            "mem_percent": mean(mem_vals),
            "temps_c": temps_mean,
        }
        summary["power"] = power_mod.summarize_power(self._samples)
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

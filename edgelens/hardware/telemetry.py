"""
edgelens.hardware.telemetry
-----------------------------
Reads live system telemetry: CPU/RAM from psutil (cross-platform),
thermal zones from /sys/class/thermal (Linux/Jetson), and best-effort
GPU utilization by parsing a single tegrastats sample (Jetson only).

GPU/tegrastats parsing is written against NVIDIA's documented tegrastats
output format (e.g. "GR3D_FREQ 42%") but has not been validated against
real hardware here — treat gpu_percent as best-effort in v0.1 and
confirm the regex still matches your JetPack version's tegrastats output.
"""

import glob
import re
import statistics
import subprocess
import threading
import time

import psutil


def read_temps():
    temps = {}
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            with open(f"{zone}/type") as f:
                name = f.read().strip()
            with open(f"{zone}/temp") as f:
                raw = f.read().strip()
            val = int(raw) / 1000.0
            temps[name] = val
        except Exception:
            continue
    return temps


def read_gpu_percent_tegrastats():
    """Single-shot GPU utilization via tegrastats. Returns None off-Jetson
    or if tegrastats isn't on PATH."""
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = proc.stdout.readline()
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()
        match = re.search(r"GR3D_FREQ (\d+)%", line)
        if match:
            return float(match.group(1))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def snapshot():
    """One telemetry sample. Safe to call on any host (Jetson or not)."""
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    temps = read_temps()
    gpu = read_gpu_percent_tegrastats()
    return {
        "cpu_percent": cpu,
        "mem_percent": mem.percent,
        "mem_used_gb": round(mem.used / 1e9, 2),
        "mem_total_gb": round(mem.total / 1e9, 2),
        "swap_used_gb": round(swap.used / 1e9, 2),
        "temps_c": temps,
        "gpu_percent": gpu,
        "timestamp": time.time(),
    }


class TelemetryRecorder:
    """Samples telemetry continuously on a background thread for the
    duration of a benchmark run, instead of a single point-in-time
    snapshot.

    Why this matters: a single snapshot taken after the benchmark finishes
    can completely miss a transient thermal spike or a brief memory
    pressure event during the run — exactly the kind of thing the
    diagnosis engine (edgelens.diagnose.engine) needs to see. Reporting
    both MEAN and PEAK per metric lets the diagnosis engine reason about
    sustained load (mean) vs. transient events (peak) separately.

    Usage:
        rec = TelemetryRecorder(interval_s=0.2)
        rec.start()
        ... run the benchmark ...
        summary = rec.stop()   # dict with *_mean and *_peak fields
    """

    def __init__(self, interval_s=0.2):
        self.interval_s = interval_s
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._samples.append(snapshot())
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.summarize()

    def summarize(self):
        if not self._samples:
            # Recorder never got a tick in (very short benchmark) — fall
            # back to one live sample rather than returning nothing.
            single = snapshot()
            temps = single.get("temps_c") or {}
            return {
                "sample_count": 0,
                "cpu_count": psutil.cpu_count(logical=True),
                "duration_s": 0.0,
                "cpu_percent_mean": single["cpu_percent"],
                "cpu_percent_peak": single["cpu_percent"],
                "gpu_percent_mean": single["gpu_percent"],
                "gpu_percent_peak": single["gpu_percent"],
                "mem_percent_mean": single["mem_percent"],
                "mem_percent_peak": single["mem_percent"],
                "temps_c_mean": temps,
                "temps_c_peak": temps,
                "max_temp_c": max(temps.values()) if temps else 0,
                # flat/legacy fields consumed by the diagnosis engine
                "cpu_percent": single["cpu_percent"],
                "gpu_percent": single["gpu_percent"],
                "mem_percent": single["mem_percent"],
                "temps_c": temps,
            }

        def column(key):
            return [s[key] for s in self._samples if s.get(key) is not None]

        def mean(values):
            return round(statistics.mean(values), 2) if values else None

        def peak(values):
            return round(max(values), 2) if values else None

        cpu_vals = column("cpu_percent")
        gpu_vals = column("gpu_percent")
        mem_vals = column("mem_percent")

        temp_keys = set()
        for s in self._samples:
            temp_keys.update((s.get("temps_c") or {}).keys())
        temps_mean, temps_peak = {}, {}
        for k in temp_keys:
            vs = [s["temps_c"][k] for s in self._samples if k in (s.get("temps_c") or {})]
            temps_mean[k] = mean(vs)
            temps_peak[k] = peak(vs)

        return {
            "sample_count": len(self._samples),
            # Core count lets the diagnosis engine tell "one core saturated"
            # (single-threaded stage) from "all cores busy" — a system-wide
            # mean of ~25% on a 4-core Nano can be one core at 100%.
            "cpu_count": psutil.cpu_count(logical=True),
            # Wall-clock span from real timestamps, NOT sample_count *
            # interval_s: one sample (psutil 0.2s + a tegrastats read) takes
            # ~1s on a Jetson Nano, far longer than the requested interval.
            "duration_s": round(self._samples[-1]["timestamp"] - self._samples[0]["timestamp"], 2),
            "effective_interval_s": (
                round((self._samples[-1]["timestamp"] - self._samples[0]["timestamp"])
                      / (len(self._samples) - 1), 2)
                if len(self._samples) > 1 else None
            ),
            "cpu_percent_mean": mean(cpu_vals),
            "cpu_percent_peak": peak(cpu_vals),
            "gpu_percent_mean": mean(gpu_vals),
            "gpu_percent_peak": peak(gpu_vals),
            "mem_percent_mean": mean(mem_vals),
            "mem_percent_peak": peak(mem_vals),
            "temps_c_mean": temps_mean,
            "temps_c_peak": temps_peak,
            "max_temp_c": max(temps_peak.values()) if temps_peak else 0,
            # flat/legacy fields (mean-based) consumed by the diagnosis engine
            "cpu_percent": mean(cpu_vals),
            "gpu_percent": mean(gpu_vals),
            "mem_percent": mean(mem_vals),
            "temps_c": temps_mean,
        }

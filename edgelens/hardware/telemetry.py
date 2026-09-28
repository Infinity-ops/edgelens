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
import subprocess
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

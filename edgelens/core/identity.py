"""
edgelens.core.identity
------------------------
Three identities, so results can be compared honestly:

  environment_id  hardware + software + runtime configuration that changes
                  performance: board, L4T/JetPack, CUDA, TensorRT, Python,
                  key packages, power mode (nvpmodel) and clock locking
                  (jetson_clocks). Two runs with different environment_ids
                  are not an apples-to-apples comparison.
  experiment_id   WHAT was measured: pipeline, pack, stages, config,
                  requirements, and the model file's content hash.
  run_id          one execution: timestamp + random suffix. Unique per run.

Previously a single fingerprint_id hashed only the environment, so two
different experiments on the same board collided.
"""

import datetime
import glob
import hashlib
import json
import os
import subprocess
import uuid

from ..hardware import detector

# Environment fields that define the id. Anything not listed (e.g. OS
# hostname details, timestamps) can't make two identical setups differ.
_ENV_ID_KEYS = ("board", "l4t", "cuda", "tensorrt", "python", "packages",
                "power_mode", "cpu_clocks_locked", "gpu_clocks_locked", "cpu_count")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def detect_power_mode():
    """nvpmodel power mode, e.g. {"id": 0, "name": "MAXN"}; None off-Jetson."""
    name, mode_id = None, None
    try:
        out = subprocess.run(["nvpmodel", "-q"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True, timeout=5)
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("NV Power Mode:"):
                name = line.split(":", 1)[1].strip()
            elif line.isdigit():
                mode_id = int(line)
    except Exception:
        pass
    if mode_id is None:
        status = _read("/var/lib/nvpmodel/status")   # e.g. "pmode:0000 fmode:quiet"
        if status and "pmode:" in status:
            try:
                mode_id = int(status.split("pmode:")[1].split()[0])
            except ValueError:
                pass
    if name is None and mode_id is None:
        return None
    return {"id": mode_id, "name": name}


def _min_eq_max(min_path, max_path):
    lo, hi = _read(min_path), _read(max_path)
    if lo is None or hi is None:
        return None
    return lo == hi


def detect_clock_locking():
    """Heuristic for `jetson_clocks`: it pins min frequency to max. Reading
    `jetson_clocks --show` needs root; these sysfs files don't."""
    cpu = _min_eq_max("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
                      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    gpu = None
    for dev in sorted(glob.glob("/sys/devices/gpu.0/devfreq/*")
                      + glob.glob("/sys/devices/platform/*gpu*/devfreq/*")
                      + glob.glob("/sys/devices/platform/bus@0/*gpu*/devfreq/*")):
        gpu = _min_eq_max(os.path.join(dev, "min_freq"), os.path.join(dev, "max_freq"))
        if gpu is not None:
            break
    return cpu, gpu


def capture_environment(hw_sw=None):
    fp = hw_sw or detector.full_fingerprint()
    hw, sw = fp["hardware"], fp["software"]
    cpu_locked, gpu_locked = detect_clock_locking()
    try:
        from ..benchmark.onnx_pipeline import available_providers
        providers = available_providers()
    except Exception:
        providers = []
    return {
        "board": hw.get("model"),
        "is_jetson": hw.get("is_jetson"),
        "l4t": sw.get("jetpack_l4t"),
        "cuda": sw.get("cuda"),
        "tensorrt": sw.get("tensorrt"),
        "python": sw.get("python"),
        "os": sw.get("os"),
        "packages": sw.get("packages"),
        "ort_providers": providers,
        "power_mode": detect_power_mode(),
        "cpu_clocks_locked": cpu_locked,
        "gpu_clocks_locked": gpu_locked,
        "cpu_count": os.cpu_count(),
    }


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def environment_id(env):
    return _hash({k: env.get(k) for k in _ENV_ID_KEYS})


def file_sha256(path, limit_bytes=None):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


def experiment_id(result, model_sha=None):
    p = result.get("pipeline") or {}
    return _hash({
        "pipeline": p.get("name"),
        "pack": p.get("pack"),
        "stages": p.get("stages"),
        "config": p.get("config"),
        "requirements": result.get("requirements"),
        "source": result.get("pipeline_source"),
        "model_sha256": model_sha,
        "mode": result.get("mode"),
    })


def new_run_id():
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def code_version(cwd=None):
    """Short git commit of the caller's working directory, if it is a repo."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, timeout=3)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   universal_newlines=True, timeout=3).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        pass
    return None


def attach_identity(result, environment=None, model_path=None):
    """Add `environment` and `identity` blocks to a result, in place."""
    env = environment or capture_environment()
    model_sha = file_sha256(model_path) if model_path else None
    result["environment"] = env
    result["identity"] = {
        "environment_id": environment_id(env),
        "experiment_id": experiment_id(result, model_sha),
        "run_id": new_run_id(),
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_sha256": model_sha,
        "code_version": code_version(),
    }
    return result

"""
edgelens.hardware.power
-------------------------
Board power from the on-module INA3221 monitors, read directly from sysfs
(no tegrastats process, no root needed).

Two kernel interfaces exist across Jetson generations:

* JetPack 4.x (Nano / TX2 / Xavier):  ina3221x IIO driver
      /sys/bus/i2c/drivers/ina3221x/*/iio:device*/rail_name_N
      /sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power{N}_input      (mW)
* JetPack 5.x / 6.x (Orin):           ina3221 hwmon driver
      /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in{N}_label
      /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in{N}_input           (mV)
      /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/curr{N}_input         (mA)

What "total power" means is board-specific, so it is always reported with
its method:
  * "input_rail"   — a rail that measures the whole module input
                     (POM_5V_IN on the Nano, VDD_IN on Orin NX/Nano);
  * "sum_of_rails" — no input rail exists (e.g. AGX Orin): the sum of the
                     monitored module rails. This excludes the carrier
                     board, so it under-reads wall power.
These are on-module sensor readings, not a calibrated power meter.
"""

import glob
import os
import statistics

# Rails that measure the whole module input, in order of preference.
INPUT_RAILS = ("POM_5V_IN", "VDD_IN")
_IGNORED = ("NC",)

_IIO_GLOB = "/sys/bus/i2c/drivers/ina3221x/*/iio:device*"
_HWMON_GLOB = "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _discover(iio_glob=_IIO_GLOB, hwmon_glob=_HWMON_GLOB):
    """Return [(rail_name, kind, paths)] for every readable rail."""
    channels = []
    for dev in sorted(glob.glob(iio_glob)):
        for name_file in sorted(glob.glob(os.path.join(dev, "rail_name_*"))):
            idx = name_file.rsplit("_", 1)[-1]
            name = _read(name_file)
            power = os.path.join(dev, f"in_power{idx}_input")
            if name and name not in _IGNORED and _read(power) is not None:
                channels.append((name, "iio_mw", (power,)))
    for dev in sorted(glob.glob(hwmon_glob)):
        for label_file in sorted(glob.glob(os.path.join(dev, "in*_label"))):
            idx = os.path.basename(label_file)[2:].split("_")[0]
            name = _read(label_file)
            volt = os.path.join(dev, f"in{idx}_input")
            curr = os.path.join(dev, f"curr{idx}_input")
            if (name and name not in _IGNORED and not name.lower().startswith("sum")
                    and _read(volt) is not None and _read(curr) is not None):
                channels.append((name, "hwmon_mv_ma", (volt, curr)))
    return channels


class PowerReader:
    """Reads instantaneous per-rail power in watts. Cheap: a few small
    sysfs reads per call. available() is False on hosts without INA3221."""

    def __init__(self, iio_glob=_IIO_GLOB, hwmon_glob=_HWMON_GLOB):
        self._channels = _discover(iio_glob, hwmon_glob)

    def available(self):
        return bool(self._channels)

    def rails(self):
        return [name for name, _, _ in self._channels]

    def read(self):
        out = {}
        for name, kind, paths in self._channels:
            try:
                if kind == "iio_mw":
                    out[name] = int(_read(paths[0])) / 1000.0
                else:
                    mv, ma = int(_read(paths[0])), int(_read(paths[1]))
                    out[name] = (mv * ma) / 1e6
            except Exception:
                continue
        return out


def total_power(rails_w):
    """(watts, method) for one reading of per-rail watts."""
    if not rails_w:
        return None, None
    for name in INPUT_RAILS:
        if name in rails_w:
            return rails_w[name], "input_rail"
    return sum(rails_w.values()), "sum_of_rails"


def total_power_w(rails_w):
    return total_power(rails_w)[0]


def summarize_power(samples):
    """Mean/peak power and energy over a list of telemetry samples that
    carry "timestamp", "power_w" and "rails_w". Energy is the trapezoidal
    integral of power over the sampled interval."""
    pts = [(s["timestamp"], s["power_w"]) for s in samples if s.get("power_w") is not None]
    if not pts:
        return {"available": False}
    method = None
    rail_names = set()
    for s in samples:
        if s.get("rails_w"):
            method = total_power(s["rails_w"])[1]
            rail_names.update(s["rails_w"].keys())
    values = [p for _, p in pts]
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        energy += (t1 - t0) * (p0 + p1) / 2.0
    rails_mean = {}
    for name in sorted(rail_names):
        vs = [s["rails_w"][name] for s in samples if name in (s.get("rails_w") or {})]
        if vs:
            rails_mean[name] = round(statistics.mean(vs), 3)
    return {
        "available": True,
        "method": method,
        "sample_count": len(pts),
        "duration_s": round(pts[-1][0] - pts[0][0], 3),
        "power_w_mean": round(statistics.mean(values), 3),
        "power_w_peak": round(max(values), 3),
        "energy_j": round(energy, 4),
        "rails_w_mean": rails_mean,
    }

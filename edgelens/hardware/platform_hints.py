"""
edgelens.hardware.platform_hints
----------------------------------
What this system's image provides, so advice fits the system it is given on.

Found on an AGX Orin 32GB on a custom carrier with a Yocto (OE4Tegra)
rootfs: no apt, no nvpmodel, no jetson_clocks, and a system clock that read
1 January 1970 (no RTC time, no NTP). EdgeLens told that board to
`sudo apt install chrony`, called its unreadable power mode "not a Jetson?"
and stamped every result with 1970.
"""

import os
import shutil
import time

# Before this moment the clock was never set (2020-01-01T00:00:00Z).
CLOCK_SANE_AFTER = 1577836800


def has_apt():
    return shutil.which("apt-get") is not None or shutil.which("apt") is not None


def install_hint(package, what=None):
    """'sudo apt install X' on Debian/Ubuntu (JetPack), otherwise an honest pointer."""
    if has_apt():
        return f"sudo apt install {package}"
    return (f"install {what or package} with this image's package manager (no apt here; on a "
            f"Yocto image add it to the image recipe, e.g. IMAGE_INSTALL:append = \" {package}\")")


def nvpmodel_available():
    return shutil.which("nvpmodel") is not None or os.path.exists("/var/lib/nvpmodel/status")


def jetson_clocks_available():
    return shutil.which("jetson_clocks") is not None


POWER_MODE_NOT_SET_ADVICE = (
    "nvpmodel reports 'power mode is not set': the board runs with its default clock "
    "limits, not MAXN or any other defined mode, so results compare to no published "
    "mode. Set one with `sudo nvpmodel -m <id>` (MAXN is 0 on AGX Orin); on a shared "
    "board agree on it first")


def power_mode_unreadable_reason(is_jetson, status=None):
    if status == "not set":
        return POWER_MODE_NOT_SET_ADVICE
    if not is_jetson:
        return "power mode not readable (nvpmodel -q): not a Jetson"
    if not nvpmodel_available():
        return ("power mode not readable: this image has no nvpmodel (custom rootfs, e.g. "
                "Yocto/meta-tegra); record the mode the image sets, e.g. in the contract note")
    return "power mode not readable (nvpmodel -q gave no mode)"


def unpin_hint(available=None):
    if jetson_clocks_available() if available is None else available:
        return "undo with `sudo jetson_clocks --restore`"
    return "undo by writing back the min_freq values you changed (or reboot)"


def pin_clocks_fix(available=None):
    if jetson_clocks_available() if available is None else available:
        return "sudo jetson_clocks --store && sudo jetson_clocks"
    return ("jetson_clocks is not on this image: as root set every "
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq to its scaling_max_freq, "
            "and the GPU devfreq min_freq to its max_freq (/sys/class/devfreq/*gpu*/)")


def clock_unset(now=None):
    """The current wall-clock date as text when it was never set, else None."""
    t = time.time() if now is None else now
    if t >= CLOCK_SANE_AFTER:
        return None
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t))

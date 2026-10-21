"""Advice that fits the image it is given on (AGX Orin, Yocto/OE4Tegra).

That board has no apt, no nvpmodel, no jetson_clocks and a clock at 1970.
EdgeLens said `sudo apt install chrony`, "not a Jetson?" for the power
mode, and `sudo jetson_clocks` for a tool that is not there.
"""
from unittest.mock import patch

from edgelens import contract
from edgelens.hardware import platform_hints as ph
from edgelens.hardware import readiness


def _no_tools(name):
    return None


def test_install_hint_without_apt():
    with patch.object(ph.shutil, "which", _no_tools):
        h = ph.install_hint("chrony")
    assert "apt" in h and "sudo apt install" not in h and "IMAGE_INSTALL" in h
    with patch.object(ph.shutil, "which", lambda n: "/usr/bin/" + n):
        assert ph.install_hint("chrony") == "sudo apt install chrony"


def test_power_mode_reason_on_a_jetson_without_nvpmodel():
    with patch.object(ph.shutil, "which", _no_tools), \
         patch.object(ph.os.path, "exists", lambda p: False):
        r = ph.power_mode_unreadable_reason(True)
    assert "no nvpmodel" in r and "not a Jetson" not in r
    assert "not a Jetson" in ph.power_mode_unreadable_reason(False)


def test_pin_clocks_fix_without_jetson_clocks():
    with patch.object(ph.shutil, "which", _no_tools):
        assert "scaling_min_freq" in ph.pin_clocks_fix()


def test_clock_unset():
    assert ph.clock_unset(now=86400 + 3600).startswith("1970-01-02")
    assert ph.clock_unset(now=1_790_000_000) is None


def test_readiness_warns_about_an_unset_clock():
    checks = readiness.assess(True, None, False, False, [], {"total_w": 5.7},
                              clock_unset="1970-01-02 11:16 UTC")
    c = [c for c in checks if c["name"] == "System clock"]
    assert c and c[0]["status"] == readiness.WARN and "1970-01-02" in c[0]["value"]


def _env(**kw):
    e = {"board": "Taipur Orin", "module": "Jetson AGX Orin 32GB", "is_jetson": True}
    e.update(kw)
    return {"environment": e}


def test_board_assumption_checks_the_module_too():
    r = contract.check_assumption(_env(), "board", {"value": "AGX Orin 32GB"})
    assert r["status"] == contract.HOLDS and r["measured"] == "Taipur Orin (Jetson AGX Orin 32GB)"
    r = contract.check_assumption(_env(), "board", {"value": "AGX Orin Industrial"})
    assert r["status"] == contract.VIOLATED
    r = contract.check_assumption(_env(module=None, board="NVIDIA Jetson Nano Developer Kit"),
                                  "board", {"value": "Nano"})
    assert r["status"] == contract.HOLDS


def test_power_mode_assumption_message_on_custom_rootfs():
    with patch.object(ph.shutil, "which", _no_tools), \
         patch.object(ph.os.path, "exists", lambda p: False):
        r = contract.check_assumption(_env(power_mode=None), "power_mode", {"value": "MAXN"})
    assert r["status"] == contract.NOT_VERIFIED and "no nvpmodel" in r["reason"]


def test_readiness_pin_fix_without_jetson_clocks():
    checks = readiness.assess(True, None, False, False, [], {"total_w": 5.7},
                              jetson_clocks=False)
    c = [c for c in checks if c["name"] == "Clocks pinned"][0]
    assert "scaling_min_freq" in c["fix"] and "sudo jetson_clocks" not in c["fix"]


# ---- "NVPM WARN: power mode is not set!" (AGX Orin, Yocto image) ----

class _Out:
    def __init__(self, text):
        self.stdout = text


def test_power_mode_not_set_is_detected():
    from edgelens.core import identity
    with patch.object(identity.subprocess, "run",
                      return_value=_Out("NVPM WARN: power mode is not set!\n")):
        assert identity.detect_power_mode_status() == "not set"
    with patch.object(identity.subprocess, "run",
                      return_value=_Out("NV Power Mode: MAXN\n0\n")):
        assert identity.detect_power_mode_status() is None
    with patch.object(identity.subprocess, "run", side_effect=FileNotFoundError):
        assert identity.detect_power_mode_status() is None


def test_readiness_warns_when_power_mode_not_set():
    checks = readiness.assess(True, None, True, True, [], {"total_w": 5.7},
                              power_mode_status="not set")
    c = [c for c in checks if c["name"] == "Power mode"][0]
    assert c["status"] == readiness.WARN and "not set" in c["value"]
    assert "default clock limits" in c["advice"]


def test_contract_power_mode_not_set_is_violated():
    r = contract.check_assumption(_env(power_mode=None, power_mode_status="not set"),
                                  "power_mode", {"value": "MAXN"})
    assert r["status"] == contract.VIOLATED and r["measured"] == "not set"


def test_unpin_advice_without_jetson_clocks():
    checks = readiness.assess(True, None, False, False, [], {"total_w": 5.7},
                              jetson_clocks=False)
    c = [c for c in checks if c["name"] == "Clocks pinned"][0]
    assert "jetson_clocks" not in c["advice"] and "min_freq" in c["advice"]

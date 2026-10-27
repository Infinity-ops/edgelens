"""
edgelens.hardware.storage
---------------------------
Storage errors in the kernel log.

A failing SD card or disk does not only lose files: every library is
memory-mapped from it, so a page that can't be read back kills the process
with SIGBUS the moment the code touches it — it looks exactly like a broken
onnxruntime install. Jetson Nano / Xavier NX dev kits boot from SD cards, so
this is checked before anything is blamed on software.
"""

import re
import subprocess

# block-layer and filesystem errors; NOT "mmc1: error -110 whilst initialising
# SD card" (an empty slot probing at boot is normal)
_PATTERNS = re.compile(
    r"(I/O error|Buffer I/O error|blk_update_request: (I/O|critical)|"
    r"print_req_error: (I/O|critical)|EXT4-fs error|EXT4-fs warning.*error|"
    r"SQUASHFS error|mmcblk\d+: (error|timed out)|"
    r"mmc\d+: (Card stuck|Timeout waiting|cache flush error|card .* error))",
    re.IGNORECASE)


def _kernel_log():
    """(text, source) or (None, reason) when the kernel log can't be read."""
    for cmd, src in ((["dmesg"], "dmesg"),
                     (["journalctl", "-k", "-b", "--no-pager", "-q"], "journalctl -k")):
        try:
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout, src
    return None, "not readable without root (kernel.dmesg_restrict=1): sudo dmesg"


_DEV = re.compile(r"\b(mmcblk\d+|sd[a-z]+|nvme\d+n\d+)")


def _card_ids():
    """{block device: hardware id} — SD/eMMC CID (unique per card), else serial."""
    import glob
    import os
    out = {}
    for dev in glob.glob("/sys/block/*"):
        name = os.path.basename(dev)
        for f in ("device/cid", "device/serial", "device/wwid"):
            try:
                with open(os.path.join(dev, f)) as fh:
                    v = fh.read().strip()
            except OSError:
                continue
            if v:
                out[name] = f"{f.split('/')[-1]}:{v}"
                break
    return out


def _cache_path():
    import os
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "edgelens", "storage_errors.json")


def _load():
    import json
    try:
        with open(_cache_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(data):
    import json
    import os
    try:
        os.makedirs(os.path.dirname(_cache_path()), exist_ok=True)
        with open(_cache_path(), "w") as f:
            json.dump(data, f, indent=1)
    except OSError:
        pass


def forget(card_id=None):
    """Drop remembered errors (all, or one card) — after checking the card is sound."""
    data = _load()
    if card_id is None:
        data = {}
    else:
        data.pop(card_id, None)
    _save(data)


def kernel_storage_errors(max_lines=5):
    """Storage errors of THIS boot's kernel log, plus errors remembered for a card
    that is still in the board. A failing SD card does not heal by rebooting, but
    the kernel log starts empty (fourth Nano run: 144 errors before a reboot, 0
    after, same card). Errors are remembered per card ID, so a NEW card is clean.

    {"readable", "count", "lines", "source", "remembered": [{"device", "card",
    "count", "first_seen", "last_seen", "last_line"}]}"""
    import datetime
    text, src = _kernel_log()
    ids = _card_ids()
    data = _load()
    if text is None:
        out = {"readable": False, "count": None, "lines": [], "source": src}
    else:
        hits = [ln.strip() for ln in text.splitlines() if _PATTERNS.search(ln)]
        out = {"readable": True, "count": len(hits), "lines": hits[-max_lines:], "source": src}
        now = datetime.datetime.now().isoformat(timespec="minutes")
        per_card = {}                      # card -> (device, error lines of this boot)
        for ln in hits:
            m = _DEV.search(ln)
            card = ids.get(m.group(1)) if m else None
            if card:
                per_card.setdefault(card, (m.group(1), []))[1].append(ln)
        for card, (dev, lines) in per_card.items():
            rec = data.setdefault(card, {"device": dev, "count": 0, "first_seen": now})
            # the log of one boot is re-read on every check: keep the largest count
            rec.update(device=dev, count=max(rec.get("count", 0), len(lines)),
                       last_seen=now, last_line=lines[-1][:200])
        if per_card:
            _save(data)
    present = set(ids.values())
    out["remembered"] = [dict(v, card=k) for k, v in data.items() if k in present]
    return out


STORAGE_ADVICE = ("The kernel reports storage read/write errors. A failing SD card or "
                  "disk makes memory-mapped libraries crash with SIGBUS and can corrupt "
                  "results: copy your data off and move to a new card before trusting "
                  "any measurement from this one.")

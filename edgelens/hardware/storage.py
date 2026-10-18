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


def kernel_storage_errors(max_lines=5):
    """{"readable", "count", "lines", "source"} for this boot's kernel log."""
    text, src = _kernel_log()
    if text is None:
        return {"readable": False, "count": None, "lines": [], "source": src}
    hits = [ln.strip() for ln in text.splitlines() if _PATTERNS.search(ln)]
    return {"readable": True, "count": len(hits), "lines": hits[-max_lines:], "source": src}


STORAGE_ADVICE = ("The kernel reports storage read/write errors. A failing SD card or "
                  "disk makes memory-mapped libraries crash with SIGBUS and can corrupt "
                  "results: copy your data off and move to a new card before trusting "
                  "any measurement from this one.")

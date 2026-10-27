"""A pipeline with a deliberate memory leak: the known answer for checking
that `edgelens soak` detects one (used by scripts/validate_on_jetson.sh).

    edgelens soak --duration 3m --window 15 --pipeline examples/soak_leak_demo.py \
        --period-ms 50 --save leak.json
    -> MEMORY_LEAK, about 1 MB per second of growth

LEAK_KB per iteration (default 50 KB) is kept forever in a list.
"""
import os
import time

import edgelens as el

LEAK_KB = int(os.environ.get("LEAK_KB", "50"))
_kept = []


def build_pipeline():
    p = el.Pipeline("leak-demo")

    def model(x=None):
        _kept.append(bytearray(LEAK_KB * 1024))     # never freed
        end = time.perf_counter() + 0.002
        while time.perf_counter() < end:
            pass
    p.add_stage("model", model)
    return p

"""A Python loop EdgeLens doesn't run itself (a ROS 2 node, a service),
instrumented with edgelens.mark.

    edgelens record --deadline-ms 50 --save py.json -- python examples/markers/python_app.py
"""
import sys
import time

from edgelens import mark

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def busy(ms):
    end = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < end:
        pass


for i in range(N):
    with mark.iteration():
        with mark.span("build_window"):
            busy(0.3)
        # GPU inference: mark.span("lstm", sync=torch.cuda.synchronize)
        with mark.span("lstm"):
            busy(6.0 if i % 50 == 0 else 2.5)
    time.sleep(0.015)

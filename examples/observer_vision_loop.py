"""
Observer mode: keep YOUR loop, add a few lines, get the full EdgeLens result.

    python examples/observer_vision_loop.py
    edgelens diagnose observer_run.json
    edgelens report observer_run.json
"""
import time

import numpy as np

import edgelens as el


def fake_camera():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def main():
    with el.trace("webcam-detector", pack="vision", target_fps=30,
                  save="observer_run.json") as t:
        for _ in range(300):
            with t.iteration():
                with t.stage("capture"):
                    frame = fake_camera()
                with t.stage("preprocess"):
                    x = frame[::2, ::2].astype(np.float32) / 255.0
                with t.stage("inference"):
                    time.sleep(0.004)          # your model here
                    scores = x.mean(axis=(0, 1))
                with t.stage("postprocess"):
                    _ = int(np.argmax(scores))
    r = t.result
    print(f"p99 {r['latency']['p99']} ms, deadline misses {r['deadline']['misses']}"
          f"/{r['iterations']} at {r['requirements']['deadline_ms']} ms frame budget")


if __name__ == "__main__":
    main()

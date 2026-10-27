"""
A toy robot for `edgelens replay / detect / envelope / perturb / parity`.

    python examples/stream/make_toy_robot.py toy_data/

writes toy_data/{train,test}/run_NN.csv: 20 Hz rows of three sensors
(vibration `vib`, a steering signal `steer`, a channel `aux` that never moves)
and a motor health truth `health` (100 = healthy) that drops in 10 % steps
after a random onset. Vibration grows as the motor degrades — that is what
the stand-in model (toy_model.py) reads.
"""

import os
import sys

import numpy as np

RATE = 20


def scenario(path, seed, seconds=900):
    rng = np.random.default_rng(seed)
    n = seconds * RATE
    t = np.arange(n) / RATE
    onset = rng.uniform(250, 450)
    step_s = rng.uniform(20, 40)
    health = np.where(t < onset, 100.0,
                      np.maximum(10.0, 100.0 - 10.0 * ((t - onset) // step_s + 1)))
    vib = rng.standard_normal(n) * (1.0 + 3.0 * (100.0 - health) / 100.0)
    steer = np.sin(t / 7.0) + 0.1 * rng.standard_normal(n)
    with open(path, "w") as f:
        f.write("time,vib,steer,aux,health\n")
        for k in range(n):
            f.write(f"{t[k]:.3f},{vib[k]:.5f},{steer[k]:.5f},0,{health[k]:g}\n")


def main(out):
    for part, seeds in (("train", range(1, 7)), ("test", range(101, 104))):
        os.makedirs(os.path.join(out, part), exist_ok=True)
        for s in seeds:
            scenario(os.path.join(out, part, f"run_{s:03d}.csv"), s)
    print(f"wrote {out}/train (6 runs) and {out}/test (3 runs)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy_data")

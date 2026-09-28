# EdgeLens

**An open-source performance doctor for NVIDIA Jetson AI workloads.**

Most Jetson monitoring tools tell you *what* your board is doing (CPU 47%,
GPU 92%, temp 61C). EdgeLens tells you **why your AI pipeline is slow** —
by breaking latency down stage-by-stage (capture → preprocess → H2D copy →
inference → D2H copy → postprocess) and running a rule-based diagnosis on
top of it.

```
$ edgelens benchmark
   Stage        Avg latency   Share
   capture      3.14 ms       8.6%
   preprocess   13.99 ms      38.2%   <-- the actual problem
   h2d copy     3.08 ms       8.4%
   inference    11.98 ms      32.7%
   d2h copy     1.41 ms       3.9%
   postprocess  2.98 ms       8.2%

$ edgelens diagnose
   CPU_BOUND_PREPROCESS (confidence 78%)
   Preprocessing consumes 38.2% of total pipeline latency while CPU
   utilization is 93.0%.
   -> Move resize/color-conversion to the GPU (CUDA/VPI) or use
      hardware-accelerated decode instead of CPU-side OpenCV.
```

## ⚠️ Honest status of this release

- The **CLI, benchmark harness, diagnosis engine, and HTML report** are
  fully implemented and tested (see `tests/`, and run `--demo` mode
  yourself on any machine).
- The **Jetson hardware detection and telemetry code** (`hardware/detector.py`,
  `hardware/telemetry.py`) is written against NVIDIA's documented interfaces
  (`/proc/device-tree/model`, `/etc/nv_tegra_release`, `/usr/local/cuda/version.json`,
  `tegrastats` output format) but **has not yet been run on physical Jetson
  hardware**. If you run this on a real board, please open an issue with
  your `edgelens doctor` output (working or broken) — that feedback is what
  turns this from "should work" into "verified."
- Stage timing in v0.1 uses wall-clock (`time.perf_counter`). CUDA-event-level
  timing (more accurate for GPU work) is a documented v0.2 target — see
  `ROADMAP.md`.

## Install

```bash
git clone <your-repo-url> edgelens
cd edgelens
pip install -e .
```

## Try it right now (no Jetson required)

```bash
edgelens doctor                                   # environment fingerprint
edgelens benchmark --demo --scenario preprocess    # synthetic pipeline run
edgelens diagnose                                  # bottleneck verdict
edgelens report                                    # self-contained HTML report
```

Demo scenarios: `balanced`, `preprocess`, `memory`, `gpu`, `thermal` — each
produces realistic synthetic data for that bottleneck type, so you can see
exactly how EdgeLens would diagnose each failure mode.

On an actual Jetson, `edgelens benchmark` will use real telemetry
automatically (no `--demo` needed) — though see the status note above.

## Commands

| Command | Purpose |
|---|---|
| `edgelens doctor` | Hardware + software fingerprint (board, JetPack/L4T, CUDA, TensorRT, key packages) |
| `edgelens monitor` | Live terminal dashboard (CPU/GPU/RAM/temp), thin wrapper over psutil + tegrastats |
| `edgelens benchmark` | Runs a benchmark, measures per-stage latency/FPS/utilization |
| `edgelens diagnose` | Rule-based bottleneck verdict: CPU-bound / GPU-bound / thermal / memory-bound / transfer-bound |
| `edgelens report` | Self-contained HTML report (fingerprint + breakdown + diagnosis), attachable to a GitHub issue |

## Wiring in your real pipeline

By default, on real hardware without a wired-in model, `edgelens benchmark`
times a placeholder pipeline (clearly labeled `pipeline_source: "placeholder"`
in its output) so the CLI never crashes — but it's not measuring your model.

To measure your actual pipeline, pass `stage_fns` to `run_benchmark()` from
Python instead of using the bare CLI:

```python
from edgelens.benchmark.runner import run_benchmark

def capture():      ...   # grab a frame
def preprocess():   ...   # resize/normalize
def h2d_copy():      ...   # copy to GPU
def inference():    ...   # run your TensorRT engine
def d2h_copy():      ...   # copy result back
def postprocess():  ...   # NMS / decode

result = run_benchmark(iterations=200, stage_fns={
    "capture": capture, "preprocess": preprocess, "h2d_copy": h2d_copy,
    "inference": inference, "d2h_copy": d2h_copy, "postprocess": postprocess,
})
```

A CLI flag for pointing at a script that defines these functions is planned
for v0.2 (see `ROADMAP.md`).

## Why not just use jetson-stats?

`jetson-stats` is a mature, actively maintained Jetson telemetry/control
library — EdgeLens is not trying to replace it, and deliberately does **not**
import it (it's AGPL-3.0 licensed; EdgeLens is MIT and reads the same
underlying system files directly to avoid license entanglement). Think of
`jetson-stats` as the telemetry layer and EdgeLens as the layer above it that
answers "given this telemetry, why is *my application* slow, and what should
I change?"

## License

MIT — see `LICENSE`.

## Roadmap

See `ROADMAP.md` for the v0.2+ plan (CUDA-event timing, precision/power-mode
sweeps, regression detection, camera/GStreamer integration) and the longer-term
plan for other accelerator families (Hailo, Rockchip, Qualcomm, Intel).

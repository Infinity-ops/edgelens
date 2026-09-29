# EdgeLens

**An open-source performance doctor for NVIDIA Jetson AI workloads.**

Most Jetson monitoring tools tell you *what* your board is doing (CPU 47%,
GPU 92%, temp 61C). EdgeLens tells you **why your AI pipeline is slow** —
by breaking latency down stage-by-stage (capture → preprocess → H2D copy →
inference → D2H copy → postprocess) and running an evidence-based diagnosis
on top of it.

```
$ edgelens benchmark --model small_cnn.onnx --iterations 150
   Stage        Avg latency   Share
   capture      0.83 ms       25.6%
   preprocess   0.16 ms       4.8%
   h2d copy     0.00 ms       0.0%
   inference    2.26 ms       69.2%   <-- inference-dominated, as expected
   d2h copy     0.00 ms       0.0%
   postprocess  0.01 ms       0.4%
   Total: 3.27 ms   FPS: 306.28

$ edgelens diagnose
   BALANCED (evidence strength 50%)
   No single stage or resource dominates the latency budget.
   Evidence: cpu_percent_mean=71.5, gpu_percent_mean=0 (no GPU on this host)
   -> Consider a precision/power-mode sweep to find further headroom.
```
*(Real captured output from a small untrained CNN on a CPU-only host —
see "Try it right now" below to reproduce. On Jetson with a heavier model
you'd typically see a CPU_BOUND_PREPROCESS or GPU_BOUND verdict instead;
this example is included because it's genuine, not because it's the most
dramatic case.)*

## ⚠️ Honest status of this release

- `edgelens benchmark --model your_model.onnx` runs a **real ONNX Runtime
  inference session** — `CPUExecutionProvider` on a laptop,
  `CUDAExecutionProvider`/`TensorrtExecutionProvider` on Jetson if the
  matching `onnxruntime-gpu` build is installed. This is a real
  measurement, not a placeholder — see `edgelens/benchmark/onnx_pipeline.py`.
- Telemetry (CPU/GPU/RAM/temperature) is sampled continuously on a
  background thread for the whole benchmark run and reported as both
  **mean and peak** per metric — a single end-of-run snapshot can miss a
  transient thermal spike entirely, so this is deliberately not that.
- Every diagnosis reports an `evidence_strength` (a heuristic score) **and**
  the raw measured numbers behind it (`evidence: {...}`), so you can check
  the reasoning yourself instead of trusting a label.
- **Telemetry-dependent findings (THERMAL, MEMORY_BOUND, CPU_BOUND_PREPROCESS,
  GPU_BOUND) are automatically demoted when the telemetry sample count is
  low.** Why this exists: `psutil.cpu_percent(interval=0.2)` blocks for
  200ms per call — a fast benchmark (a small model, or few iterations) can
  finish its entire timed loop faster than that, so the background sampler
  only gets one instantaneous reading. A single ambient temperature spike
  at that instant is not evidence the *benchmark* caused it. Below 3
  samples, `evidence_strength` is halved and the finding's `detail` gets an
  explicit `[LOW CONFIDENCE]` note explaining why — found and fixed via
  real hardware testing, not simulated. Run more `--iterations` or a
  heavier model for a telemetry verdict you can actually trust.
- The **Jetson hardware detection code** (`hardware/detector.py`'s
  `/proc/device-tree/model` and `/etc/nv_tegra_release` parsing,
  `hardware/telemetry.py`'s `tegrastats` regex) is written against NVIDIA's
  documented formats but **has not yet been run on physical Jetson
  hardware**. If you run this on a real board, please open an issue with
  your `edgelens doctor` output (working or broken) — that's what turns
  this from "should work" into "verified." The `pyproject.toml`
  `Development Status` classifier is deliberately **Pre-Alpha** until that
  validation happens.
- Stage timing uses wall-clock (`time.perf_counter`). CUDA-event-level
  timing (more accurate for GPU work, isolates GPU time from CPU
  scheduling noise) is a documented v0.3 target — see `ROADMAP.md`.
- `--demo` mode (synthetic data, for previewing the tool without a model
  or Jetson) is loudly and unmissably labeled everywhere it appears — in
  the terminal, in the saved JSON (`"mode": "demo"`), and in the HTML
  report (page title, a red banner top and bottom, and a background
  watermark). It should never be mistaken for a real measurement.

## Install

```bash
git clone <your-repo-url> edgelens
cd edgelens
pip install -e ".[dev]"       # core + pytest, for running the test suite
```

Optional extras (`pip install -e ".[extra1,extra2]"`):

| Extra | Adds | When you need it |
|---|---|---|
| `dev` | `pytest` | Running `tests/` |
| `onnx` | `onnxruntime` | Using `--model` with a real `.onnx` file |
| `fixtures` | `onnx` (the model-building library, not the runtime) | Only if regenerating `tests/fixtures/tiny_model.onnx` via `make_tiny_model.py` |

`onnx` (the fixtures extra) is deliberately kept separate from `dev` —
newer `onnx` releases pull in `protobuf>=6`, which can break sibling
packages elsewhere in your environment that are pinned to `protobuf` 5.x
(this happened in real testing against a project depending on
`ankaios-sdk`/`grpcio-tools`). Most contributors never need it.

For Jetson GPU acceleration, install the JetPack-specific wheel instead
of plain `onnxruntime`:

```bash
pip install onnxruntime-gpu      # Jetson JetPack wheel, for CUDA/TensorRT
```

(This wheel comes from NVIDIA's own JetPack index, not standard PyPI, so
`edgelens` itself does not pin a version for you.)

## Try it right now

No model of your own handy yet? A small (untrained, CPU-friendly) real
CNN is bundled for exactly this — it's what produced the example output
above:

```bash
edgelens doctor                                              # environment fingerprint
edgelens benchmark --model tests/fixtures/small_cnn.onnx --iterations 150
edgelens diagnose                                             # bottleneck verdict, with evidence
edgelens report                                                # self-contained HTML report
```

With your own model:

```bash
edgelens benchmark --model your_model.onnx
edgelens compare before.json after.json            # regression check between two runs
```

No model handy? Preview the tool with synthetic data instead:

```bash
edgelens benchmark --demo --scenario preprocess
```

Demo scenarios: `balanced`, `preprocess`, `memory`, `gpu`, `thermal` — each
produces realistic synthetic data for that bottleneck type, so you can see
exactly how EdgeLens would diagnose each failure mode. Demo output is
loudly watermarked everywhere — see the status note above.

## Commands

| Command | Purpose |
|---|---|
| `edgelens doctor` | Hardware + software fingerprint (board, JetPack/L4T, CUDA, TensorRT, key packages) |
| `edgelens monitor` | Live terminal dashboard (CPU/GPU/RAM/temp), thin wrapper over psutil + tegrastats |
| `edgelens benchmark --model X.onnx` | Runs a real ONNX Runtime benchmark, measures per-stage latency/FPS/utilization |
| `edgelens diagnose` | Evidence-based bottleneck verdict: CPU-bound / GPU-bound / thermal / memory-bound / transfer-bound |
| `edgelens report` | Self-contained HTML report (fingerprint + breakdown + diagnosis), attachable to a GitHub issue |
| `edgelens compare before.json after.json` | Before/after diff — FPS, latency, per-stage deltas, and a PASS/REGRESSION verdict |

## Wiring in your real pipeline

`--model your_model.onnx` gives you a real measurement with minimal
preprocessing (`OnnxStagePipeline` in `edgelens/benchmark/onnx_pipeline.py`
does a synthetic random frame, a min-max normalize, and a raw output sum —
deliberately minimal so it never silently misrepresents YOUR actual
pre/postprocessing).

To measure your actual camera/preprocessing/model, pass your own
`stage_fns` to `run_benchmark()` from Python instead:

```python
from edgelens.benchmark.runner import run_benchmark

def capture():      ...   # grab a real frame
def preprocess():   ...   # your real resize/normalize
def h2d_copy():      ...   # copy to GPU
def inference():    ...   # run your real TensorRT engine
def d2h_copy():      ...   # copy result back
def postprocess():  ...   # your real NMS / decode

result = run_benchmark(iterations=200, stage_fns={
    "capture": capture, "preprocess": preprocess, "h2d_copy": h2d_copy,
    "inference": inference, "d2h_copy": d2h_copy, "postprocess": postprocess,
})
```

A CLI flag for pointing at a script that defines these functions
(`edgelens benchmark --pipeline my_pipeline.py`) is planned for v0.2 —
see `ROADMAP.md`.

## Why not just use jetson-stats?

`jetson-stats` is a mature, actively maintained Jetson telemetry/control
library — EdgeLens is not trying to replace it, and deliberately does **not**
import it (it's AGPL-3.0 licensed; EdgeLens is MIT and reads the same
underlying system files directly to avoid license entanglement). Think of
`jetson-stats` as the telemetry layer and EdgeLens as the layer above it that
answers "given this telemetry, why is *my application* slow, and can I
prove a fix worked?" (see `edgelens compare`).

Likewise, EdgeLens is not trying to replace Nsight Systems/Compute — those
give cycle-level CUDA profiling that a Python tool cannot reproduce.
EdgeLens sits above both: interpretation, benchmarking, and regression
detection, not raw profiling.

## License

MIT — see `LICENSE`.

## Roadmap

See `ROADMAP.md` for the v0.2+ plan (physical Jetson validation, CUDA-event
timing, precision/power-mode sweeps, CI regression, camera/GStreamer
integration) and the longer-term plan for other accelerator families
(Hailo, Rockchip, Qualcomm, Intel).

# EdgeLens Roadmap

## Guiding principle

Depth on one ecosystem beats breadth on ten. v0.1–v0.9 are NVIDIA Jetson
only, going as deep as CUDA/TensorRT-level instrumentation allows. Other
accelerator families come later, once the core engine is proven — see
"Beyond Jetson" at the bottom.

## v0.1 — shipped in this release

- [x] `doctor` — hardware/software fingerprint
- [x] `monitor` — live CPU/GPU/RAM/temp dashboard
- [x] `benchmark` — stage-level latency harness (wall-clock timing)
- [x] `diagnose` — rule-based bottleneck engine (CPU/GPU/thermal/memory/transfer-bound)
- [x] `report` — self-contained HTML report + JSON fingerprint
- [x] `--demo` mode for previewing output without Jetson hardware
- [x] MIT license, no AGPL dependency

**Not yet validated on physical Jetson hardware** — see README status note.

## v0.2 — CUDA-aware timing (the technical moat)

- CUDA event-based timing (`cudaEvent_t` via PyCUDA or `torch.cuda.Event`)
  for the H2D copy / inference / D2H copy stages, replacing wall-clock
  timing with GPU-side measurement that isn't polluted by CPU scheduling
  noise. This is what stops EdgeLens from being "another jtop."
- `edgelens benchmark --pipeline my_pipeline.py` — point at a script
  defining stage functions instead of wiring `stage_fns` from Python.
- Validate `hardware/telemetry.py` tegrastats parsing against JetPack 5.x
  and 6.x real output (regex may need adjusting per L4T version).

## v0.3 — precision & power-mode sweeps

- `edgelens optimize model.onnx` — run the same benchmark across
  FP32/FP16/INT8 and across power modes (5W/10W/15W/MAXN), report
  FPS, latency, and **FPS/W** for each, so "should I switch precision"
  has a measured answer instead of a guess.

## v0.4 — reproducibility as a first-class feature

- Fingerprint format (already scaffolded in `benchmark/fingerprint.py`)
  becomes the standard way to attach a bug report or compare two
  developers' results.
- `edgelens compare fingerprint_a.json fingerprint_b.json` — diff two runs.

## v0.5 — regression detection (the feature that gets companies to adopt it)

- `edgelens ci --baseline baseline.json --limits limits.yaml` — pass/fail
  exit code for CI pipelines, catching "why did FPS silently drop after
  last week's commit."

## v0.6+ — integration depth (real value, but integration work — not before the core is solid)

- Camera/GStreamer/V4L2/CSI pipeline instrumentation
- DeepStream pipeline hooks
- Docker/container deployment readiness checks (`edgelens deploy-check`)
- Isaac ROS / ROS 2 node-level profiling

## Beyond Jetson

Once the core engine (stage attribution + diagnosis + fingerprinting) is
proven and stable on Jetson, the same `core` abstractions (defined
hardware-neutrally in terms of "stage," "telemetry," "fingerprint") extend
to other accelerator families via new `hardware/` and `benchmark/` adapters:

- **EdgeLens for Hailo** — HailoRT-based telemetry/timing adapter
- **EdgeLens for Rockchip** — RKNN-based adapter
- **EdgeLens for Qualcomm** — SNPE/QNN-based adapter
- **EdgeLens for Intel** — OpenVINO-based adapter

This is explicitly **not** v0.1–v0.9 work. Generalizing too early loses the
deep, vendor-specific hooks (CUDA events, DLA, NVENC/NVDEC) that make the
Jetson version actually useful, and waters down the value before the core
is proven on any one platform.

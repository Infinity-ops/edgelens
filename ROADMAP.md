# EdgeLens Roadmap

## Guiding principle

Depth on one ecosystem beats breadth on ten. v0.1–v0.6 are NVIDIA Jetson
only, going as deep as CUDA/TensorRT-level instrumentation allows. Other
accelerator families come later, once the core engine is proven — see
"Beyond Jetson" at the bottom.

## What EdgeLens is (and isn't)

jtop gives you board telemetry. Nsight gives you cycle-level GPU/CUDA
profiling. trtexec gives you one engine's raw numbers. None of them tell
you _why your application_ is slow in plain language, or let you prove a
fix worked. That gap — not raw timing precision — is what EdgeLens is
for. CUDA-event timing, DLA telemetry, TensorRT introspection (below) are
all _inputs_ to that interpretation layer; they are not the moat by
themselves. EdgeLens orchestrates and interprets what the existing NVIDIA
tooling already measures — it does not re-implement it.

| Tool                        | Main question it answers                                                    |
| --------------------------- | --------------------------------------------------------------------------- |
| jetson-stats (jtop)         | What is my board doing right now?                                           |
| Nsight Systems              | What is my whole system doing?                                              |
| Nsight Compute              | Why is this CUDA kernel behaving this way?                                  |
| trtexec / TensorRT profiler | How fast is my engine, layer by layer?                                      |
| **EdgeLens**                | **Why is my application performing this way, and what should I test next?** |

## v0.1.0 — the actual first release (gate: validated on real Jetson hardware)

This absorbs what was previously planned as a separate "v0.2 hardware
validation" release. A Pre-Alpha release claiming Jetson support with
zero Jetson testing behind it is a worse first impression than taking
longer and shipping something genuinely validated — so v0.1.0 is not
tagged or published until every item below is checked off.

**Already built and tested (on a non-Jetson host — CPU ONNX Runtime
path, demo path, full CLI, validated against both a tiny linear model
and a real multi-layer CNN):**

- [x] `doctor` — hardware/software fingerprint
- [x] `monitor` — live CPU/GPU/RAM/temp dashboard
- [x] `benchmark --model X.onnx` — real ONNX Runtime inference
      (CPUExecutionProvider laptop / CUDA-TensorRT Jetson), stage-level
      latency via IOBinding for real H2D/D2H separation on GPU providers
- [x] Continuous background telemetry sampling (mean + peak per metric)
- [x] `diagnose` — rule-based bottleneck engine, evidence-first output
      (`evidence_strength` + raw evidence numbers), with automatic
      demotion of telemetry-dependent findings when the sample count is
      too low to trust
- [x] `compare` — before/after diff with PASS/REGRESSION verdict
- [x] `report` — self-contained HTML report + JSON fingerprint
- [x] `--demo` mode, unmissably labeled everywhere it appears
- [x] MIT license (SPDX), no AGPL dependency
- [x] Known bug fixed: `capture()` no longer regenerates a random frame
      every iteration (was costing ~1ms on realistic input sizes, enough
      to look like a false bottleneck) — found via real-CNN testing

**Still required before v0.1.0 is tagged — the actual gate:**

- [ ] Run `doctor` on a real Jetson (Nano/Orin/Xavier — whatever's
      available); fix whatever the `/etc/nv_tegra_release` /
      `/proc/device-tree/model` parsing gets wrong
- [ ] Run `benchmark --model X.onnx` with `onnxruntime-gpu` (the
      JetPack-specific wheel, not the PyPI one) on the same board;
      confirm the CUDA/TensorRT provider path and IOBinding H2D/D2H
      separation actually work
- [ ] Validate the `tegrastats` GPU% regex against real output for
      whichever JetPack version is available; fix the regex against the
      actual format if it doesn't match
- [x] `edgelens benchmark --pipeline my_pipeline.py` — point at a script
      defining a `build_stage_fns()` function, instead of only
      supporting `stage_fns` wired from Python. Clear, user-facing error
      messages for every malformed-script case (missing entrypoint,
      wrong return type, missing/extra/non-callable stages) — see
      `tests/test_pipeline_loader.py` and `tests/fixtures/example_pipeline.py`.
- [ ] Set `Development Status :: 3 - Alpha` once everything above is
      checked off and confirmed working

## v0.2 — GPU execution analyzer

- CUDA event-based timing (`torch.cuda.Event` or PyCUDA) for the
  inference stage, replacing wall-clock timing with GPU-side measurement
  that isn't polluted by CPU scheduling noise
- DLA (Deep Learning Accelerator) utilization telemetry — Jetson Orin's
  dedicated inference cores, separate from the GPU
- CPU-side vs GPU-side time split (launch overhead, GPU idle time,
  synchronization waits) — the beginning of the "GPU execution analyzer"
  framing, built on top of the event timing above, not a separate effort

## v0.3 — precision/power sweeps + TensorRT introspection

- `edgelens optimize model.onnx` — run the same benchmark across
  FP32/FP16/INT8 and across power modes (5W/10W/15W/MAXN), report FPS,
  latency, and **FPS/W** for each
- TensorRT engine introspection (layer list, per-layer precision, which
  ops fall back off TensorRT) — placed here, not earlier, because it
  needs the v0.1 real-hardware foundation and the v0.2 GPU-side timing
  to produce trustworthy numbers rather than guesses. This is where an
  `edgelens analyze model.onnx`-style command — ONNX/TensorRT
  compatibility, operator fallback detection, the "Model Doctor" framing
  — would eventually live, once it can be built on validated ground
  instead of ahead of it.

## v0.4 — reproducibility as a first-class feature

- Split the fingerprint into `environment_id` (hardware+software) and
  `experiment_id`/`run_id` (model+config+timestamp) — right now
  `fingerprint_id` only hashes environment, so two different experiments
  on the same machine collide

## v0.5 — regression detection in CI

- `edgelens ci --baseline baseline.json --limits limits.yaml` — pass/fail
  exit code for CI pipelines (the `compare` command already exits 1 on
  regression — this wraps it with configurable thresholds and a
  GitHub-Actions-friendly summary format)

## v0.6+ — integration depth

- Camera/GStreamer/V4L2/CSI pipeline instrumentation
- DeepStream pipeline hooks
- Docker/container deployment readiness checks
- Isaac ROS / ROS 2 node-level profiling
- An automated multi-configuration "optimization lab" sweeping precision
  × resolution × power mode × batch size against a stated FPS/power
  target — real value, but only once v0.1–v0.5 are individually proven;
  bundling it earlier risks shipping an automation layer on top of
  numbers nobody has validated yet

## Beyond Jetson

Once the core engine (stage attribution + diagnosis + fingerprinting) is
proven and stable on Jetson, the same abstractions extend to other
accelerator families via new `hardware/` and `benchmark/` adapters:

- **EdgeLens for Hailo** — HailoRT-based telemetry/timing adapter
- **EdgeLens for Rockchip** — RKNN-based adapter
- **EdgeLens for Qualcomm** — SNPE/QNN-based adapter
- **EdgeLens for Intel** — OpenVINO-based adapter
- Desktop/RTX GPU support, for the same reason as above: a larger
  addressable audience, once Jetson depth is proven rather than instead
  of it

This is explicitly **not** v0.1–v0.6 work.

## Deliberately out of scope for now

A mentor review raised a longer-term open-core/commercial direction
(hosted benchmark comparison across hardware, a performance knowledge
base, fleet monitoring for robot deployments). Noted here as a real
possibility worth remembering, not as a roadmap commitment — it depends
on an audience and dataset that only exist after the open-source core
is genuinely trusted, which is still gated on v0.1.

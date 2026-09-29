# EdgeLens Roadmap

## Guiding principle

Depth on one ecosystem beats breadth on ten. v0.1–v0.9 are NVIDIA Jetson
only, going as deep as CUDA/TensorRT-level instrumentation allows. Other
accelerator families come later, once the core engine is proven — see
"Beyond Jetson" at the bottom.

## v0.1.0 — this release

- [x] `doctor` — hardware/software fingerprint
- [x] `monitor` — live CPU/GPU/RAM/temp dashboard
- [x] `benchmark --model X.onnx` — **real ONNX Runtime inference**
      (CPUExecutionProvider laptop / CUDA-TensorRT Jetson), stage-level
      latency via IOBinding for real H2D/D2H separation on GPU providers
- [x] Continuous background telemetry sampling (mean + peak per metric),
      not a single end-of-run snapshot
- [x] `diagnose` — rule-based bottleneck engine, evidence-first output
      (`evidence_strength` + raw evidence numbers, not an unqualified
      "confidence" score), with automatic demotion of telemetry-dependent
      findings when the sample count is too low to trust (found via real
      hardware testing: a fast benchmark can finish faster than one
      `psutil` sampling interval, yielding a single noisy reading)
- [x] `compare` — before/after diff with PASS/REGRESSION verdict
- [x] `report` — self-contained HTML report + JSON fingerprint
- [x] `--demo` mode, unmissably labeled (terminal banner top+bottom, HTML
      page title/banner/watermark) — cannot be mistaken for a measurement
- [x] MIT license (SPDX), no AGPL dependency
- [x] `Development Status :: 2 - Pre-Alpha` classifier (honest — see below)

**Not yet validated on physical Jetson hardware.** Everything above has
been tested on a non-Jetson host (CPU ONNX Runtime path, demo path, full
CLI, and against a real multi-layer CNN in addition to the tiny test
fixture — which is how the capture-stage RNG-cost bug below was found).
The Jetson-specific code paths — `is_jetson()`/`detect_jetpack()`/
`detect_cuda()` in `hardware/detector.py`, the `tegrastats` regex in
`hardware/telemetry.py`, and the CUDA/TensorRT execution provider path in
`benchmark/onnx_pipeline.py` — are written against NVIDIA's documented
interfaces but have never executed on real silicon. **This is the #1
priority for v0.2**, and the classifier stays Pre-Alpha until it's done.

**Known limitation, fixed:** the default `OnnxStagePipeline.capture()`
used to generate a fresh random frame every iteration via
`np.random.rand()`. For a realistic CNN input size this costs ~1ms per
call — enough to rival or exceed a small model's actual inference time,
making "capture" look like a real bottleneck when it was actually
measuring NumPy's RNG cost. Found via testing against a real CNN (not
the microsecond-scale tiny fixture). Fixed: the frame is now generated
once at construction and referenced, not regenerated, per iteration.

## v0.2 — physical Jetson validation (next release, and the actual gate)

- [ ] Run `doctor` on a real Jetson (Nano/Orin/Xavier — whatever's
      available); fix whatever the `/etc/nv_tegra_release` /
      `/proc/device-tree/model` parsing gets wrong
- [ ] Run `benchmark --model X.onnx` with `onnxruntime-gpu` on the same
      board; confirm the CUDA/TensorRT provider path and IOBinding H2D/D2H
      separation actually work
- [ ] Validate the `tegrastats` GPU% regex against real output for
      whichever JetPack version is available
- [ ] Bump `Development Status` to Alpha once the above is done and this
      roadmap entry is checked off
- [ ] `edgelens benchmark --pipeline my_pipeline.py` — point at a script
      defining stage functions instead of wiring `stage_fns` from Python

## v0.3 — CUDA-aware timing (the technical moat)

- CUDA event-based timing (`torch.cuda.Event` or PyCUDA) for the
  inference stage, replacing wall-clock timing with GPU-side measurement
  that isn't polluted by CPU scheduling noise. This is what stops
  EdgeLens from being "another jtop."

## v0.4 — precision & power-mode sweeps

- `edgelens optimize model.onnx` — run the same benchmark across
  FP32/FP16/INT8 and across power modes (5W/10W/15W/MAXN), report
  FPS, latency, and **FPS/W** for each.

## v0.5 — reproducibility as a first-class feature

- Split the fingerprint into `environment_id` (hardware+software) and
  `experiment_id`/`run_id` (model+config+timestamp) — right now
  `fingerprint_id` only hashes environment, so two different experiments
  on the same machine collide.

## v0.6 — regression detection in CI

- `edgelens ci --baseline baseline.json --limits limits.yaml` — pass/fail
  exit code for CI pipelines (the `compare` command already exits 1 on
  regression — this wraps it with configurable thresholds and a
  GitHub-Actions-friendly summary format).

## v0.7+ — integration depth

- Camera/GStreamer/V4L2/CSI pipeline instrumentation
- DeepStream pipeline hooks
- Docker/container deployment readiness checks (`edgelens deploy-check`)
- Isaac ROS / ROS 2 node-level profiling

## Beyond Jetson

Once the core engine (stage attribution + diagnosis + fingerprinting) is
proven and stable on Jetson, the same `core` abstractions extend to other
accelerator families via new `hardware/` and `benchmark/` adapters:

- **EdgeLens for Hailo** — HailoRT-based telemetry/timing adapter
- **EdgeLens for Rockchip** — RKNN-based adapter
- **EdgeLens for Qualcomm** — SNPE/QNN-based adapter
- **EdgeLens for Intel** — OpenVINO-based adapter

This is explicitly **not** v0.1–v0.9 work.

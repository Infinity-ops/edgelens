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

## v0.1.0 — the first release (gate: validated on a real Jetson Nano)

**Done and tested (101 tests, Python 3.8 and 3.12):**

- [x] `doctor`, `monitor`, `benchmark`, `diagnose`, `report`, `compare`
- [x] Real ONNX Runtime benchmarking (CPU / CUDA / TensorRT providers, IOBinding
      H2D/D2H split); **all model inputs fed with declared dtypes**,
      `--input-shape`, no guessing of non-image dynamic dims
- [x] **Generic Pipeline engine** — any number of named stages with roles;
      harness mode (`Pipeline.run`, `--pipeline`) and **observer mode**
      (`edgelens.trace`); one-argument stages receive the previous output
- [x] **Scenario packs** — `vision` (the classic six stages), `timeseries`
      (window/hop, real-time factor, `WindowSource` replay), `custom`;
      `register_pack()` for third-party packs
- [x] **Raw trace** (per stage per iteration) and **timestamped telemetry
      series** in every result
- [x] **Tail metrics** — min/mean/std/p50/p90/p95/p99/p99.9/max/jitter with
      sample-count honesty rule
- [x] **`--deadline-ms`** (misses, ratio, worst overrun, bursts),
      `--period-ms` backlog estimate, `--pace` measured response time
- [x] **Power and energy** via INA3221 (JetPack 4 iio + JetPack 5/6 hwmon);
      energy/iteration, iterations/J, dynamic energy over idle baseline
- [x] **environment_id / experiment_id / run_id**; power mode and clock
      locking in the environment; **environment-aware `compare`**
      (`--strict-env`)
- [x] **Categorical evidence strength** (weak/moderate/strong) + data quality;
      role-based rules; DEADLINE_MET / DEADLINE_MISSED; pack findings
- [x] **`schema_version: 1`** in every JSON document; pre-v0.1 files still read
- [x] Telemetry sampler rewritten: non-blocking CPU%, sysfs GPU load, one
      persistent tegrastats process (was one process per sample), sampler
      cost reported
- [x] HTML report escapes all data-derived text
- [x] Nano-validated diagnosis rules: INFERENCE_ON_CPU, single-core
      CPU_BOUND_PREPROCESS

**Release gate — run `scripts/validate_on_jetson.sh` on the Nano:**

- [x] sysfs GPU load node read correctly (Nano, JetPack 4.6)
- [x] INA3221 power on the Nano (root-only files detected, fix printed; 4.2 W / 51.3 mJ per inference)
- [x] `nvpmodel` parsed (MAXN / 5W); clock pinning flips after `sudo jetson_clocks`
- [x] `compare --strict-env` exits 2 across power modes
- [x] CUDA and TensorRT provider runs; multi-input model; timeseries example;
      paced run; observer mode
- [x] README status updated, `Development Status :: 3 - Alpha`
- [x] CI: lint, tests on Python 3.8–3.13 (with and without onnxruntime),
      wheel build + smoke test; release workflow (TestPyPI / PyPI)
- [ ] Set the release date in CHANGELOG.md, tag v0.1.0, publish

## What would show v0.1 is worth continuing

v0.1 is an experiment: it tests whether "requirement → evidence → decision"
is a problem engineers actually have. Stars are not the signal. These are:

- [ ] 10 people using it on a real pipeline (not the bundled fixtures)
- [ ] 3 hardware reports from boards other than the Nano
- [ ] 2 outside contributors
- [ ] 1 researcher using EdgeLens results in an experiment or paper
- [ ] 3 external case studies, one of each:
  1. EdgeLens identified the dominant stage in a real latency problem;
  2. a change was made and `compare` showed the measured before → after;
  3. a stated requirement (e.g. p99 < 10 ms, < 0.1 % misses) got a PASS or
     FAIL with evidence.

The *Case study* issue form collects these.

## Release themes

| Release | Theme | One line |
| --- | --- | --- |
| v0.1 | **Measure** | Measure and diagnose real edge AI pipelines |
| v0.2 | **Validate** | Prove a workload meets its performance contract, on Jetson AGX Orin (Industrial) |
| v0.3 | **Understand the runtime** | Continuous workloads: queues, drops, concurrency, multiple models |
| v0.4 | **Optimize** | Find the best configuration under your constraints |

## v0.2 — validate (performance contracts) + Orin depth

- **First milestone: physical validation on Jetson AGX Orin Industrial**
  (`scripts/validate_on_jetson.sh`), before any new Orin feature

- Contract YAML (latency percentiles, deadline miss ratio, power, energy,
  quality) and `edgelens validate contract.yaml` with per-requirement
  PASS/FAIL, the limiting stage with its evidence, and the recommended next
  experiments; violation events for runtime monitors
- Stable, documented Python API
- Non-root power access: optional systemd/udev helper so the INA3221 files
  stay readable across reboots
- Orin platform profile: DLA activity, EMC frequency, per-rail power,
  throttle detection, real thermal trip points instead of a fixed 80 C
- `quality_metrics=` hook (quality × latency × energy, e.g. FP32 vs INT8)
- Placement profile output (task × device × power mode → latency, energy,
  memory, cold start) for schedulers and resource allocators

## v0.3 — streams and more packs

- Live stream mode: queue depth, backpressure, dropped / out-of-order samples
- Critical-path latency for concurrent stages (from the trace)
- `llm` pack (time to first token, tokens/s, KV-cache memory, energy/token)
- `multi-model` pack (co-location interference, diverse redundant paths,
  output disagreement)
- `doctor` fix hints (mismatched ORT wheel, power mode, clocks)

## v0.4 — experiments and edge/cloud

- `edgelens experiment run x.yaml`: precision × power mode × batch ×
  placement sweeps, constraint filtering, Pareto front
- `offload` pack + network probe (RTT, jitter, bandwidth, break-even)
- Opt-in fault scenarios orchestrating `stress-ng` / `tc netem`

## v0.5+

- CUDA-event GPU-side timing; TensorRT layer attribution
- CI regression mode with thresholds file
- ROS 2 / GStreamer / DeepStream adapters

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

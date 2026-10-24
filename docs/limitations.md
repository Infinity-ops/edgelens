# Known limitations (v0.2)

What EdgeLens does not do, or does with a caveat you should know before
trusting a number. Where a limitation affects a result, the result itself
says so (a note, `low_confidence`, NOT_MEASURED or INCONCLUSIVE); this page
collects them in one place. What was checked on which board:
[jetson.md](jetson.md).

## Platform

- **Linux only** (x86-64 and aarch64), Python 3.8–3.13. macOS and Windows
  are not supported: telemetry, power, process sampling and markers use
  `/proc`, sysfs and UNIX sockets.
- **Board telemetry and power need an NVIDIA Jetson.** On other machines
  EdgeLens measures latency, CPU and memory; GPU load, temperatures and power
  are reported as unavailable (`n/a`, `null`), never as 0. Containers often
  hide thermal zones and power sensors too.
- **Hardware verification is partial.** See [jetson.md](jetson.md): the v0.1
  scope on a Jetson Nano and the onnxruntime CPU path on an AGX Orin 32GB.
  CUDA / TensorRT / DLA on Orin, Xavier and Orin Nano / NX have not been
  run by the maintainers: reports from those boards are very welcome.

## Timing

- Stage times are host-side CLOCK_MONOTONIC intervals; stages run one after
  another in one thread. An asynchronous GPU call is charged to the stage
  where the host waits for it (e.g. a copy or synchronisation), not to the
  stage that launched it. No CUDA-event timing, no concurrent stages yet.
- The telemetry sampler is a thread in the measured process. Its CPU cost
  per sample is in every result (`sampler_cpu_ms_mean`), and `selftest`
  measures how much it disturbs a reference workload. When the workload
  keeps every core busy (onnxruntime's default of one spinning thread per
  core does), the sampler competes with it; use `--threads` below the core
  count (or a longer `telemetry_interval_s=` in the Python API) if tail
  latency matters.
- Pure-Python pipelines that hold the GIL can starve the sampler; the
  result then warns "Telemetry starved".
- Percentiles need enough samples: p99 needs at least 100 iterations, p99.9
  at least 1000; below that they are reported as missing, not estimated.
- Paced runs (`--pace`, `periodic`) count from the scheduled release time, so
  OS wake-up lateness is part of the response time; `periodic` reports it
  separately.
- Throughput in back-to-back runs is 1 / mean latency; in paced, markers,
  attach and ROS 2 runs it is the observed rate (`throughput_basis`).

## Power and energy

- Power comes from the on-module INA3221 monitors, sampled every 100 ms by
  default: a calibrated power meter it is not. On the AGX Orin there is no
  input rail; the **sum of module rails** under-reads the carrier board
  (`method: sum_of_rails`).
- On JetPack 4.x the sensor files are root-only; `edgelens doctor` prints the
  one-line fix.
- Energy per inference is total energy over the measured window divided by
  iterations. When an inference is shorter than the sampling interval, it is
  derived from average power, not observed per inference; `periodic` says
  when the extra energy per inference is below the sensor's resolution.
  Fewer than 3 power samples: energy is NOT_MEASURED.

## Long runs (`soak`)

- Leak detection uses the process's private memory (RssAnon) where Linux
  provides it; memory-mapped files (a model, a data file being read) are
  reported separately and are not a leak. GPU memory is tracked through
  Jetson NvMap only.
- A leak only shows once the process has used up memory it already holds:
  start a soak in a fresh process and run it long enough; growth below
  8 MB or 5% is never called a leak.
- p99 drift needs windows of at least 100 iterations each.

## Models and data

- `--model` runs ONNX models through onnxruntime only (CPU, CUDA, TensorRT
  execution providers). Other runtimes: `--pipeline` with your own code, or
  markers in your application; `sweep --external` compares exported outputs.
- `--inputs` are fed exactly as given (no normalisation) and `--labels` are
  scored against the model's **first** output: regression, 0/1 detection,
  or classification (top-1 / top-5). Detection boxes, segmentation masks,
  mAP or IoU are not computed: use `--pipeline` with your own
  post-processing.
- Image folders are decoded into memory once (up to 2 GB of pixels).
- TensorRT engines are built on first use (minutes); `sweep` reports the
  build time separately and caches engines.

## Contracts and validation

- A contract checks only the assumptions it lists. An assumption EdgeLens
  cannot verify on a board (for example a power mode an image doesn't
  report) makes the verdict INCONCLUSIVE, never PASS.
- `validate` never passes a requirement the run did not measure; a
  measurement-integrity issue (lost markers, unsynchronised GPU spans,
  log-sourced timing, clock doubt) turns a latency PASS into NOT_MEASURED.

## Experimental parts

`stream`, `replay`, `detect`, `envelope`, `perturb`, `parity` and
`ros2 latency` are experimental in 0.2 ([stability.md](stability.md)):
their file formats are frozen, their reports may still change. Not yet run
on Jetson hardware by the maintainers.

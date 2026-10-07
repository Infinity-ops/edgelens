# Jetson validation log

## Status of this release (v0.1.0)

**Validated on a real Jetson Nano** (JetPack 4.6 / L4T R32.7.6, Python 3.8
venv, ONNX Runtime CPU, CUDA and TensorRT providers):

| Area           | Verified on the Nano                                                                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Engine         | `--model` (single and multi-input), `--pipeline` (vision and timeseries), observer mode, paced mode                                                       |
| Statistics     | per-stage and end-to-end latency, p50–p99.9, max, jitter, percentile sample-count rule                                                                    |
| Requirements   | deadline met / missed / thin-margin detection, miss bursts, real-time factor                                                                              |
| Environment    | power mode (nvpmodel) and clock pinning (jetson_clocks) detected; `compare --strict-env` exits 2 across MAXN vs 5W                                        |
| Telemetry      | CPU, RAM, thermal zones, GPU load (sysfs)                                                                                                                  |
| Power & energy | INA3221 `POM_5V_IN` read from sysfs: 4.2 W mean, 51.3 mJ per inference (TensorRT, small CNN). Needs root or a one-time `chmod` on JetPack 4.6, see below |
| Outputs        | JSON schema v1 with trace, telemetry series and identities; HTML report; compare exit codes                                                               |

**Implemented, not yet verified on real hardware:**

- **Jetson Orin** support in general (hwmon power layout, JetPack 5/6 GPU
  load paths). Unit-tested against the documented layouts only.
  `edgelens doctor` shows a _Telemetry sources_ table; if a source is
  missing on your board, please open an issue with that table.

**Known limits (by design in v0.1):**

- Stage timing is host wall-clock (`time.perf_counter_ns`), stages run
  sequentially in one thread. GPU-side CUDA-event timing and concurrent
  stages come later.
- The telemetry sampler is a Python thread in the measured process. Its
  CPU cost per sample is reported in every result (`sampler_cpu_ms_mean`)
  so the observer effect is visible, not hidden.
- Power figures are on-module sensor readings with their method
  (`input_rail` / `sum_of_rails`), not a calibrated power meter.
- **Jetson Nano / JetPack 4.x: the power sensor files are root-only.**
  `edgelens doctor` detects this and prints the fix. Either make them
  readable until the next reboot:
  `sudo chmod o+r /sys/bus/i2c/drivers/ina3221x/*/iio:device*/rail_name_* /sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input`
  or run one benchmark as root: `sudo $(which edgelens) benchmark ...`
  (its output files are then owned by root).
- No built-in camera, video, audio or CAN readers: your own capture code
  becomes the first stage (or use observer mode in your existing loop).
  Recorded signals can be replayed with `WindowSource`.

## Real results on a Jetson Nano

Measured with this release (`tests/fixtures/small_cnn.onnx`, 300 iterations
unless noted):

| Run                                                          | Mean    | p99      | What EdgeLens reported                                                      |
| -------------------------------------------------------------- | ------- | -------- | ----------------------------------------------------------------------------- |
| CPU provider, MAXN                                            | 27.8 ms | 30.3 ms  | `INFERENCE_ON_CPU`: 92% of the time is CPU inference                        |
| CPU provider, 5W mode                                          | 86.8 ms | 184.3 ms | `compare --strict-env`: environment mismatch (power mode), exit 2           |
| TensorRT, clocks not pinned (100 it.)                           | 17.8 ms | 148.1 ms | tail is 11x the median; clocks not pinned                                   |
| TensorRT, after `jetson_clocks` (100 it.)                      | 6.8 ms  | 9.4 ms   | 2.6x faster mean, 16x lower p99                                             |
| TensorRT, with board power (clocks not pinned)                 | 14.2 ms | 113.4 ms | 4.2 W mean · 51.3 mJ per inference · 19.5 inferences per joule              |
| Timeseries pipeline (10 kHz, 1024/512 window), 2,000 windows   | 5.4 ms  | 8.4 ms   | real-time factor 0.11; 0 of 2,000 deadline misses at 51.2 ms; p99.9 10.5 ms |
| Same, released every 2 ms (`--pace`)                            | —       | —        | cannot keep up: queue grows to ~1 s; measured, not simulated                |

The clock-pinning row is the kind of thing EdgeLens exists for: same board,
same model, same code, and a 16x difference in tail latency that FPS alone
would never show.

> **Note:** the "clocks not pinned" rows above were measured at different
> times and board states (iteration count and thermal/clock-boost state
> differ between runs), so don't read them as a controlled A/B pair across
> every column — only the `jetson_clocks` before/after row on its own is a
> controlled comparison (same iteration count, same session). A fully
> controlled re-run of all rows, back-to-back from a known clock state, is
> tracked for the next validation pass.

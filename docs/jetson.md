# Jetson validation log

## Status of v0.1.0 (Jetson Nano)

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

## v0.2 validation status

Rule: a v0.2 feature is called *verified on hardware* only where its row
below says **measured**, from a run on a real board whose output was kept.
Everything else is implemented and passes CI (Python 3.8–3.13, Linux x86-64),
nothing more.

Boards:

- **Jetson Nano** (JetPack 4.6 / L4T R32.7.6): the v0.1 scope is verified
  (section above). v0.2 development runs on this Nano found and fixed bugs
  (see the changelog); no complete v0.2 gate run has been archived yet.
- **Jetson AGX Orin 32GB** (`tegra234`, as `edgelens doctor` reports it),
  custom carrier, Yocto / OE4Tegra image, **shared with another team**:
  clocks left as configured (not pinned), power mode not set by nvpmodel,
  onnxruntime **CPU execution provider only** (no CUDA / TensorRT build on
  this image), runs as root. Power is the **sum of module rails** (no input
  rail on the AGX Orin), which under-reads the carrier board.

| Gate | What is measured | Jetson Nano (JP 4.6) | Jetson AGX Orin 32GB |
| --- | --- | --- | --- |
| T2 one clock | stage timing, markers, telemetry on CLOCK_MONOTONIC | _pending_ | _pending_ |
| T3 selftest: harness error | 1 ms / 10 ms reference, median error | _pending_ | _pending_ |
| T3 selftest: span cost | Python / C, µs per span (kernel sendto) | _pending_ | _pending_ |
| T3 selftest: sampler | % of one core; telemetry shift % | _pending_ | _pending_ |
| T4 A/A | identical setups → NO DETECTABLE CHANGE | _pending_ | _pending_ |
| T4 A/B | `jetson_clocks` → IMPROVED, change and 95% CI | _pending_ | not possible on a shared board (changes its clocks) |
| T5 schema | 0.1.0 file loads in every command | _pending_ | _pending_ |
| S1 markers | C++ and Python: received / lost at a realistic rate | _pending_ | _pending_ |
| S2 attach | process table; deliberate leak detected | _pending_ | _pending_ |
| S3 ROS 2 | chain latency vs markers in the node | _pending_ | _pending_ (no ROS 2 on this image) |
| S4 bag replay | robot bag (IMU / odometry / commands) → 1 Hz pipeline, SLA check | _pending_ | _pending_ |
| I3 sweep | p50 / energy / max error / decision agreement per backend (CPU, CUDA, TRT, TRT-FP16, DLA) | _pending_ | _pending_ (CPU only on this image) |
| I1 periodic | 1 Hz vs back-to-back p50; energy per period; pinned vs unpinned trade-off | _pending_ | **measured** (1 Hz vs back to back, energy per period); pinned vs unpinned not possible on a shared board |
| I2 soak | 30 min at 1 Hz: drift, throttling, memory growth; deliberate leak found | _pending_ | **partial**: 30 min run measured; memory verdict and deliberate leak still open (below) |
| E1 contracts | PASS on a quiet board; each assumption deliberately violated → INCONCLUSIVE | _pending_ | **partial**: PASS path measured; violations not yet run |
| E2 timeline | two applications + board power on one Perfetto timeline, opened in ui.perfetto.dev | _pending_ | _pending_ |
| E3 CI | example repository: self-hosted runner blocks a pull request that breaks the contract | _pending_ | _pending_ |
| L4 replay | windowed pipeline, incremental vs recompute features: per-stage ms per step; `--speed 1` deadline misses | _pending_ | _pending_ |
| L4 detect / envelope / perturb / parity | identical decisions on the board as on the laptop (same time to detect, same outputs to 1e-4) | _pending_ | _pending_ |

On a shared board, `bash scripts/validate_on_jetson.sh` **without**
`EDGELENS_VALIDATE_SUDO=1` never changes clocks or the power mode: the
violated-assumption gates edit the contract, not the board. Only the
`jetson_clocks` A/B and the pinned/unpinned trade-off need the board to
yourself.

### What was measured on the AGX Orin

One small LSTM (ONNX, about 2 ms per inference on the CPU), onnxruntime CPU
execution provider, EdgeLens 0.2.0.dev4/dev5:

| Run | Result |
| --- | --- |
| Back to back, onnxruntime default threads (one per core) | p50 2.09 ms, p99 2.65 ms · CPU 98% of all 8 cores · 11.2 W · 23.7 mJ per inference |
| Same, 4 intra-op threads | p50 2.15 ms, p99 2.45 ms · CPU 49% · 8.8 W · **19.3 mJ** per inference |
| Same, 2 / 1 threads | p50 3.12 / 5.14 ms · CPU 27% / 13% · 23.3 / 35.1 mJ per inference |
| EdgeLens vs a bare onnxruntime loop, same model | p50 within 0.7–2.3%. p99 34–41% higher in EdgeLens: it ran 2437 iterations with its telemetry sampler thread while onnxruntime's default threads kept every core busy; the bare loop ran 600 without a sampler. Which of the two explains the tail is not yet separated |
| `periodic`, 1 Hz vs back to back (default threads) | p50 6.15 ms vs 2.21 ms (**2.8x**); OS wake-up p50 0.35 ms; 5.15 J per period incl. idle (5.23 W idle); energy above idle per inference below the sensor's resolution |
| Paced 1 Hz, default vs 1 thread | p50 7.6 ms vs 32.9 ms at the same mean power (5.3 / 5.2 W): one thread is slower **and** not cheaper at 1 Hz |
| `contract run`, 5 min at 1 Hz | PASS: every assumption held (board, clocks not pinned, input period, start temperature 43.6 C, background load 0.13%); p99 20.3 ms, 0 of 301 deadline misses, 5.2 W mean, 44.3 C peak |
| `soak`, 30 min at 1 Hz, 4 threads | drift +39% p50 (not significant: the windows alternate between ~7 ms and ~31 ms levels); no throttling; peak 44.3 C |
| `soak` memory | 0.2.0.dev4 reported LEAK SUSPECTED (+221 MB/h). The growth (+107 MB) matches the input windows read in order from a 146 MB memory-mapped file (1836 × 60 kB = 110 MB): file pages, not a leak. 0.2.0.dev5 separates private memory from mapped files; **a rerun to confirm this is pending** |

Open on this board: the latency levels' cause (dev5 records the CPU clock
per level), the soak rerun, the deliberate leak, and the remaining gates.

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

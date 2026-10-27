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
  system clock never set (no RTC time, no NTP), onnxruntime **CPU execution
  provider only** (no CUDA / TensorRT build on this image), runs as root.
  Power is the **sum of module rails** (no input rail on the AGX Orin),
  which under-reads the carrier board.

Named configurations (a claim holds for the configuration it was measured
on, nothing wider):

| ID | Board | Software | Runtimes run | EdgeLens | Date | Evidence publishable |
| --- | --- | --- | --- | --- | --- | --- |
| **NANO-1** | Jetson Nano | JetPack 4.6 / L4T R32.7.6, Python 3.8 venv | onnxruntime-gpu: CPU, CUDA, TensorRT | 0.1.0 | before 2026-10-07 | yes (repository fixtures) |
| **ORIN-2** | Jetson AGX Orin 32GB, custom carrier | Yocto / OE4Tegra, kernel 6.12, Python 3.11 venv, onnxruntime 1.31 (CPU build; CUDA 12.6 and TensorRT 10.3 on the image, not used), shared board | onnxruntime: CPU only | 0.2.0rc1 | October 2026 (board clock unset) | yes (repository workloads only) |
| **CI-X86** | GitHub-hosted runner | Ubuntu 22.04, Python 3.8–3.13 | onnxruntime: CPU | every commit | every commit | yes (Actions logs) |

A row is **independently reproducible** when its evidence is publishable and
the exact commands are in [validation-v0.2.md](validation-v0.2.md):
NANO-1, ORIN-2 and CI-X86. ORIN-2 is the gate script, run once on the
shared board without `EDGELENS_VALIDATE_SUDO`: the results below are from
that run.

| Gate | What is measured | Jetson Nano (JP 4.6) | Jetson AGX Orin 32GB (ORIN-2) |
| --- | --- | --- | --- |
| T2 one clock | stage timing, markers, telemetry on CLOCK_MONOTONIC | _pending_ | **measured**: stage timing, `perf_counter` and telemetry on CLOCK_MONOTONIC (1 ns resolution) |
| T3 selftest: harness error | 1 ms / 10 ms reference, median error | _pending_ | **measured**: +1.0 µs at 1 ms and at 10 ms (limits 50 / 200 µs); scheduling jitter 29 µs (worst 1%) |
| T3 selftest: span cost | Python / C, µs per span (kernel sendto) | _pending_ | **measured** (Python): 6.9 µs per span, 3.6 µs beyond the kernel's `sendto` (3.3 µs); 20,000 spans at 4,000/s, 0 lost. C: not run (no compiler on this image) |
| T3 selftest: sampler | % of one core; telemetry shift % | _pending_ | **measured**: 4.7% of one core at 10 samples/s (WARN, limit 2%); shifts a 2 ms reference by 0.05% |
| T4 A/A | identical setups → NO DETECTABLE CHANGE | _pending_ | **measured**: 5 + 5 runs, p50 +1.2% [-0.9, +3.3], smallest detectable ±2.9%: NO DETECTABLE CHANGE. Energy and mean power read REGRESSED (+3.8% / +1.8%) on the same identical runs: see [limitations](limitations.md#power-and-energy). Known change (+1 ms in a 2 ms stage): REGRESSED, +49.9% p50 |
| T4 A/B | `jetson_clocks` → IMPROVED, change and 95% CI | _pending_ | not possible on a shared board (changes its clocks) |
| T5 schema | 0.1.0 file loads in every command | _pending_ | **measured**: a 0.1.0 result loads in `diagnose`, `validate`, `compare` |
| S1 markers | C++ and Python: received / lost at a realistic rate | _pending_ | **measured** (Python): 200 iterations, 600 spans, 0 lost; `import-log` 300 lines. C++: not run (no compiler on this image) |
| S2 attach | process table; deliberate leak detected | _pending_ | **measured**: `watch` per-process CPU / memory / context switches with board power; a deliberate 10 MB/s leak read as +573 MB/min |
| S3 ROS 2 | chain latency vs markers in the node | _pending_ | _pending_ (no ROS 2 on this image) |
| S4 bag replay | robot bag (IMU / odometry / commands) → 1 Hz pipeline, SLA check | _pending_ | _pending_ (rosbags not installed) |
| I3 sweep | p50 / energy / max error / decision agreement per backend (CPU, CUDA, TRT, TRT-FP16, DLA) | _pending_ | **measured** (CPU only): CPU row timed and checked; CUDA / TensorRT / FP16 / DLA reported *unavailable* (not in this onnxruntime build), never as numbers |
| I1 periodic | 1 Hz vs back-to-back p50; energy per period; pinned vs unpinned trade-off | _pending_ | **measured**: 1 Hz, 10 Hz, 100 Hz vs back to back; energy per period. Extra energy per inference: found unreliable, fixed in 0.2.0 (below). Pinned vs unpinned: not possible on a shared board |
| I2 soak | 30 min at 1 Hz: drift, throttling, memory growth; deliberate leak found | _pending_ | **measured**: deliberate leak found (3 min, private memory +3,493 MB/h, LEAK SUSPECTED; latency drift < 1%, not significant; no throttling). 30 min bag soak: _pending_ (needs S4) |
| E1 contracts | PASS on a quiet board; each assumption deliberately violated → INCONCLUSIVE | _pending_ | **measured**: PASS on the quiet board (5 min at 1 Hz, p99 8.1 ms, 0 of 301 misses, background load 0.09%); board, start temperature, clock state, input pacing, background load (3 CPU hogs) and provider each deliberately violated → INCONCLUSIVE (the provider before the run). Power mode: not readable on this image |
| E2 timeline | two applications + board power on one Perfetto timeline, opened in ui.perfetto.dev | _pending_ | **partial**: one Perfetto JSON from two recordings: spans, iterations, CPU / GPU load, temperature and power tracks; opening it in ui.perfetto.dev by hand: _pending_ |
| E3 CI | example repository: self-hosted runner blocks a pull request that breaks the contract | _pending_ | **partial**: `ci/perf_gate.sh` on the board: PASS, job summary, evidence and timeline written; a self-hosted runner blocking a pull request: _pending_ |
| L4 replay | windowed pipeline, incremental vs recompute features: per-stage ms per step; `--speed 1` deadline misses | _pending_ | _pending_ |
| L4 detect / envelope / perturb / parity | identical decisions on the board as on the laptop (same time to detect, same outputs to 1e-4) | _pending_ | _pending_ |

On a shared board, `bash scripts/validate_on_jetson.sh` **without**
`EDGELENS_VALIDATE_SUDO=1` never changes clocks or the power mode: the
violated-assumption gates edit the contract, not the board. Only the
`jetson_clocks` A/B and the pinned/unpinned trade-off need the board to
yourself.

### What was measured on the AGX Orin (ORIN-2)

The repository's own workloads (`tests/fixtures/small_cnn.onnx`,
`tests/fixtures/lstm_anomaly.onnx`, `examples/`), onnxruntime CPU execution
provider, EdgeLens 0.2.0rc1, clocks not pinned and no power mode set (the
board's defaults). Commands: the README's "On a Jetson" block, the
[quick start](../examples/quickstart/README.md) and
`bash scripts/validate_on_jetson.sh` (no sudo). The gate script ran alone,
except for its last five minutes: a second copy was started by mistake while
the E3 gate ran. E3's own background-load check recorded it (7.4% of the CPU
from other processes, under the contract's 20% limit) and its verdict stands
on that measurement; the other gates' results are from the quiet board.

| Run | Result |
| --- | --- |
| README block: `small_cnn.onnx`, 300 iterations, 33 ms deadline | p50 2.78 ms, p99 4.53 ms, max 5.92 ms; 0 of 300 misses; 9.16 W, 26.3 mJ per inference (sum of rails); inference 81% of the time, preprocessing 17% |
| Quick start (`quickstart.py`, 1000 windows) | p50 0.549 ms, p99 0.691 ms, p99.9 0.866 ms; `validate` PASS, `check_result.py` OK. The run takes 0.6 s: energy not measured (fewer than 3 power samples) and "Telemetry starved" (the pure-Python stages hold the GIL), both as designed |
| `examples/timeseries_vibration.py`, 2000 windows | mean 1.01 ms, p99 1.65 ms, p99.9 3.99 ms; 0 of 2000 misses at 51.2 ms; 5.09 W, 4.75 mJ per window |
| `periodic`, `lstm_anomaly.onnx`, 30 inferences per rate | back to back p50 0.27 ms; every 10 ms 1.62 ms (5.9x), every 100 ms 3.70 ms (13.6x), every 1000 ms 3.85 ms (14.1x); OS wake-up p50 0.22–0.89 ms; 4.28 J per period at 1 Hz with 3.91 W idle (GPU railgate on) |
| `periodic` extra energy per inference at 1 Hz | three runs on the quiet board printed 57, 114 and 376 mJ for the same 0.3 ms inference; 376 mJ in a 4 ms inference would be ~100 W on a board drawing 4–9 W. The idle baseline had moved, not the model: 0.2.0 measures idle before and after, and reports no figure when the row draws more than the same work back to back (see the changelog) |
| `contract run`, 5 min at 1 Hz (`examples/ci-jetson`) | PASS: board, clocks as declared, input period, start temperature 42.9 C, background load 0.09%; p99 8.09 ms, 0 of 301 misses, peak 42.9 C |

### Development runs on the same AGX Orin

Before ORIN-2, development builds ran on this board with a model and data
that are not public; those results are not published and no claim rests on
them. They found the bugs listed under "found on an AGX Orin" in the
[changelog](../CHANGELOG.md). One behaviour was checked there and nowhere
else yet: on a 30 min soak reading a memory-mapped input file, 0.2.0rc1
reported the mapped-file growth as file pages and private memory as flat,
where 0.2.0.dev4 had called the same growth a leak.

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

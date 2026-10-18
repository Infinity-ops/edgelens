# Changelog

## Unreleased — 0.2.0.dev0 (Layers 0–3 of the v0.2 roadmap)

### Added — Layer 3: evidence and workflow
- **Assumption contracts (E1):** `validate --contract c.yaml` checks
  requirements AND the conditions they are claimed under — board, power mode,
  pinned clocks, input period, start temperature, thermal envelope,
  background CPU load, provider, model hash — plus `min_duration` /
  `min_iterations` and requirement IDs. A violated or unverifiable assumption
  gives INCONCLUSIVE, never PASS; a FAIL outside the assumptions is reported
  but not counted as evidence. Unknown keys are errors with a suggestion.
  `--evidence e.md` writes a Markdown evidence record (one row per
  requirement ID). `edgelens contract init | check | run`: `run` checks the
  assumptions it can before measuring, then measures exactly what the contract
  needs (paced period, deadline, iterations, duration; soak or periodic when
  the guarantees need it) and validates.
- **Background load** is now measured in every run: CPU time of every process
  except EdgeLens and the measured application (`telemetry.background`);
  telemetry also records the start temperature, the GPU clock and every
  power rail per sample.
- **`edgelens export --perfetto` (E2):** spans of every process and thread,
  end-to-end iterations, deadline misses, board counters (CPU/GPU load, GPU
  clock, EMC, memory, temperature, power and each rail) and per-process
  counters on one timeline for ui.perfetto.dev; several files share the
  monotonic clock when recorded on the same boot. Checked against Perfetto's
  trace processor.
- **Performance CI (E3):** `examples/ci-jetson/` — a complete example
  repository: GitHub Actions workflow for a self-hosted Jetson runner
  (concurrency per board, clocks pinned and restored, evidence as job summary
  and artifact), a DEM2 contract, the gate script (exit code = verdict) and a
  DEM2-style pipeline; docs/contracts.md, docs/timeline.md, docs/ci.md.
- Dependency: PyYAML (contract files).

### Added — Layer 2: insights nobody else gives you
- **`edgelens sweep` (I3):** one model on CPU / CUDA / TensorRT FP32, FP16,
  INT8 / DLA (+ external runtimes such as EDDL via exported inputs and saved
  outputs): latency, energy, session start-up (TensorRT engine cache) and
  output equivalence against a reference backend — max |error|, values within
  tolerance, decision agreement (argmax / threshold). Verdicts EQUIVALENT /
  NUMERIC_DRIFT / DECISIONS_DIFFER; finding OUTPUT_DRIFT; validate
  `--max-output-error`, `--min-decision-agreement`.
- **`edgelens periodic` (I1):** latency and energy at the real inference rate
  (period sweep) against back-to-back, with service time and OS wake-up delay
  kept apart, energy per period (idle included) and extra energy per
  inference over a measured idle baseline; finding DUTY_CYCLE_PENALTY; GPU
  railgate state; `compare` of two periodic runs prints the trade-off both
  ways; validate `--max-energy-per-period-mj`.
- **`edgelens soak` (I2):** time-boxed windows streamed to JSONL (crash-safe,
  `--report` rebuilds), for EdgeLens-driven pipelines and observed
  applications; drift (significance-tested trend), thermal throttling (cooling
  devices, clock drops), process RSS and Jetson NvMap growth, available-memory
  trend and time to OOM, swap; findings LATENCY_DRIFT, THERMAL_THROTTLING,
  MEMORY_LEAK, GPU_MEMORY_GROWTH, OUT_OF_MEMORY_RISK, SWAP_IN_USE; validate
  `--max-latency-drift-pct`, `--max-mem-growth-mb-per-h`,
  `--max-throttled-windows`, `--max-temp-c`.
- `tests/fixtures/lstm_anomaly.onnx` (DEM2-shaped stand-in),
  `examples/soak_leak_demo.py`; docs/insights.md.

### Fixed — from the second Jetson Nano run
- The GPU-provider check reported "SIGBUS: broken install" with a single,
  sound onnxruntime-gpu 1.11 (JetPack 4.6): it allocated a bare CUDA tensor.
  It now creates and runs a CUDA session on a tiny embedded model and names
  the step that failed; "two packages" is blamed only when two are installed.
- A SIGBUS is first checked against the kernel log: storage errors (a failing
  SD card) are named as the cause. `doctor` has a Storage row; the validation
  script warns that results from such a card are not evidence.
- The check's cache can't outlive the problem: only a success is cached,
  keyed by interpreter, onnxruntime install, CUDA driver file and boot id;
  failures are re-tested every run. `doctor` says when a result came from the
  cache; `doctor --no-cache` and `cuda_probe(force=True)` test again.
- `doctor`'s GPU row shows the check's actual finding instead of always
  "CUDA/cuDNN libraries missing".
- The validation script's preflight no longer prints its output twice.

### Fixed — from the first Jetson Nano run
- A broken onnxruntime (two packages installed over each other) crashed every
  command with "Bus error": the CUDA provider check now runs in a child
  process with a timeout, cached per install; `doctor` names the conflict and
  the fix; the validation script stops at a preflight check, gives every step
  a timeout and stops on the first crash with a Python traceback.
- Markers on slow CPUs: the collector receives raw and parses later; senders
  flush overflow in 4 KB chunks with a 200 µs back-off (a full queue cost
  234 µs per C span on the Nano); faster Python `mark.span`.
- selftest: genuinely realistic marker rate (4000 spans/s), costs beyond the
  kernel's sendto(), collector capacity, scheduling jitter (desktop hint),
  sampler breakdown, implausible chrony bounds flagged.
- Telemetry: slow thermal zones (firmware / I2C reads of several ms) are read
  every 20th sample; swap every 10th.

Implemented and tested in CI (Python 3.8 and 3.13); not yet validated on
Jetson hardware — see docs/jetson.md for the rows still pending.

### Added — can you trust the numbers?
- **One clock (T2):** every timestamp is CLOCK_MONOTONIC: harness and
  observer stages, markers, telemetry samples (`t_mono_ns`), energy
  integration. New `clock` block in results with the wall↔monotonic offset
  and its uncertainty; chrony error bounds for cross-machine comparisons.
- **`edgelens selftest` (T3):** measures EdgeLens itself on the board:
  clock, harness error against a known reference workload, Python and C
  marker cost next to the kernel's `sendto()` cost, marker delivery and exact
  loss accounting, datagram queue limit, sampler CPU, telemetry disturbance,
  chrony.
- **Statistical compare (T4):** `benchmark --repeat N [--cooldown S]` writes
  NAME.r01.json…; `compare 'a.r*.json' 'b.r*.json'` gives IMPROVED /
  REGRESSED / NO DETECTABLE CHANGE from Welch intervals over runs, with the
  smallest change the runs could have detected; `--metric`, `--min-effect`.
  Single-run compares show a within-run block-bootstrap interval of p99.
  Results now carry `latency_series_ms`.
- **Schema v1 frozen (T5):** docs/schema.md; a file written by edgelens
  0.1.0 is a test fixture that must keep working everywhere; newer schemas
  are refused clearly.
- **Measurement integrity checks:** lost marker events, unsynchronised GPU
  spans, log-sourced timing, missing latency, cross-machine clock doubt.
  `validate` turns a latency PASS into NOT_MEASURED when an issue blocks it
  (a FAIL stays a FAIL); `diagnose` lists them.

### Added — systems EdgeLens doesn't run
- **Markers (S1):** header-only `edgelens_mark.h` for C/C++ (`EL_SPAN`,
  `EL_ITERATION`, `EL_SPAN_GPU`, `EL_SPAN_CUDA`, `-DEDGELENS_DISABLE`) and
  `edgelens.mark` for Python; never-blocking UNIX datagrams with overflow
  batching and exact loss accounting. `edgelens record [-- CMD]`,
  `edgelens include` (for `-I`), `edgelens import-log --regex`.
- **Attach mode (S2):** `edgelens watch --pid / --name / -- CMD`: per-process
  CPU, memory and growth rate, threads, context switches, with board
  telemetry on one timeline; collects markers too.
- **ROS 2 latency (S3, experimental):** `edgelens ros2 latency --from --to`:
  chain latency by matching header stamps (no clock sync needed), data age
  with explicit clock uncertainty, per-topic rate, jitter and gaps.
- **ROS bag replay (S4):** `examples/rosbag_to_windows.py` (as-of merge of
  /imu, /odom, /cmd_vel_rc100 + any topic, data-age columns, gap report;
  extra `edgelens[rosbag]`) and `examples/dem2_bag_replay.py` (1 Hz
  stand-in LSTM pipeline with a freshness check).
- `examples/markers/cpp_app.cpp`, `examples/markers/python_app.py`;
  `scripts/validate_on_jetson.sh` extended with a gate per feature and a
  SUMMARY.txt; docs/markers.md, docs/validation-v0.2.md.

### Measurement correctness (from an independent pre-release review)
- Throughput in markers / attach / ROS 2 / paced runs is the observed rate
  over the run, not 1 / mean latency (`throughput_basis` says which).
- `record -- CMD` uses a private socket; `watch --pid/--name` keeps only the
  target's own spans. Energy per iteration in observed runs covers only the
  active window (first to last span), not start-up.
- Markers are fork-safe in C and Python; duplicate sequence numbers, senders
  killed before their final message, unstored spans, multi-thread runs
  without `@iteration`, and ROS messages missing from a chain all block a
  latency PASS instead of passing silently.

### Changed
- Timeseries pack: `hop` may be larger than `window` (predict less often
  than the sensor delivers, e.g. 0.5 s of history once per second).
- Role inference recognises lstm/gru/rnn/cnn/eddl/trt/onnx/anomaly names as
  inference.

## 0.1.0 — 2026-10-07

### Added
- Generic `edgelens.Pipeline` engine: any named stages with roles; harness
  mode (`Pipeline.run`, `benchmark --pipeline` with `build_pipeline()`) and
  observer mode (`edgelens.trace`).
- Scenario packs: `vision`, `timeseries` (with `WindowSource`), `custom`;
  `register_pack()`.
- Result document `schema_version: 1`: end-to-end `latency` stats
  (p90/p99.9/max/jitter/tail spread), `stage_stats_ms`, `deadline`,
  `backlog`, `pack_metrics`, `telemetry_series`, `energy`, `trace`,
  `environment`, `identity`.
- `benchmark` flags: `--deadline-ms`, `--period-ms`, `--pace`,
  `--idle-baseline`, `--input-shape`.
- INA3221 power reader (JetPack 4 iio, JetPack 5/6 hwmon); power in
  `monitor`; energy per iteration.
- `environment_id` / `experiment_id` / `run_id`; nvpmodel power mode and
  clock-lock detection.
- `compare`: any stage names (added/removed shown), deadline-miss and energy
  deltas, environment mismatch warning, `--strict-env` (exit code 2).
- Diagnosis: DEADLINE_MISSED, DEADLINE_MET, INFERENCE_ON_CPU, single-core
  CPU_BOUND_PREPROCESS, timeseries findings (CANNOT_KEEP_UP,
  TAIL_OVERRUNS_HOP, HIGH_OVERLAP_COST), vision BELOW_TARGET_FPS.
- `examples/timeseries_vibration.py`, `examples/observer_vision_loop.py`,
  `scripts/validate_on_jetson.sh`.

### Changed
- `evidence_strength` is now categorical (`weak`/`moderate`/`strong`); the
  numeric ordering value moved to `rank_score`.
- Diagnosis rules use stage roles, not the six fixed stage names.
- `--pipeline` scripts may return any stage names (no longer exactly six).
- Telemetry: non-blocking CPU%, sysfs GPU load, one persistent tegrastats
  process; default sampling interval 0.1 s.
- `report` uses the environment captured at benchmark time.
- `fingerprint_id` is now an alias of `environment_id`.

### Added after Jetson Nano validation
- `doctor` shows **Benchmark readiness**: power mode, clock pinning, GPU
  inference providers and power-sensor access, each with a copyable fix.
  Built from the three issues hit on a real Nano (unpinned clocks, PyPI
  onnxruntime replacing the GPU build, root-only power sensor).
- `doctor` prints a *Telemetry sources* table (GPU load, INA3221 power,
  thermal zones, tegrastats) with values and read cost.
- Diagnosis `CLOCKS_NOT_PINNED`: long tail with unpinned clocks on Jetson
  (from a real Nano run: p99 148 ms unpinned vs 9.4 ms pinned).
- `service_latency` in results; `compare` flags runs with different
  requirements (deadline/period/pacing/pack).

### Fixed after Jetson Nano validation
- Paced mode: real-time factor used response time (incl. queueing) and the
  hop period instead of service time and the actual period (Nano run showed
  RTF 10.4 instead of ~2.7). An explicit `--period-ms` now also sets the
  default deadline.
- `compare` named a 0.014 -> 0.050 ms copy (+257%) as the largest
  regression; stages are now ranked by milliseconds added, ignoring noise.
- Clock fields are labelled "pinned (min = max)": the Nano's 5W nvpmodel
  mode pins CPU clocks too, so "jetson_clocks" was a wrong attribution.
- Sampler cost is reported as CPU time (`sampler_cpu_ms_mean`); wall time
  (incl. GIL waits) is kept as `sampler_wall_ms_mean`.
- tegrastats is run line-buffered (`stdbuf -oL`) so parsed lines arrive on
  time through a pipe; a power sensor that is listed but unreadable falls
  back to tegrastats instead of silently disabling power.
- Non-Jetson notice no longer claims everything runs in demo mode.
- Power on the Jetson Nano (JetPack 4.6): the INA3221 files are root-only
  (mode 0600). This looked like "no INA3221 in sysfs"; EdgeLens now reports
  `permission_denied` with a one-line `chmod` fix (covering the rail-name
  files too) in `doctor` and after `benchmark`. Verified on a Nano: 4.2 W,
  51.3 mJ per inference.
- Writing an output file that is not writable (e.g. created by an earlier
  `sudo edgelens` run) gives a one-line error with the `chown` fix instead
  of a traceback.

### Project
- GitHub Actions: lint, tests on Python 3.8–3.13 (with and without
  onnxruntime), wheel build + clean-venv smoke test; release workflow with
  PyPI trusted publishing (manual run -> TestPyPI, GitHub Release -> PyPI).
- `CITATION.cff`, `CONTRIBUTING.md`, bug-report / hardware-report issue
  forms, PR template. README: badges, `pip install edgelens` first,
  architecture diagram, measured headline result. Classifier: Alpha.

### Added: `edgelens validate`
- Checks a saved run against requirements given as flags (latency
  percentiles, worst case, deadline miss ratio, throughput, power, energy
  per iteration) and answers PASS / FAIL / INCONCLUSIVE per requirement with
  the measured value. On FAIL it names the limiting stage and attaches the
  run's diagnosis. A requirement the run cannot measure (too few iterations
  for the percentile, no deadline, no power sensor, a `--demo` result) is
  never a PASS. Exit codes 0 / 1 / 2; verdict saved as `*.validation.json`.
  Contract files come in v0.2.

### Fixed in the pre-release review
- The sdist was missing `tests/__init__.py` and `tests/fixtures/` (the test
  suite failed when run from the sdist); added `MANIFEST.in`.
- The release workflow ran no tests before publishing; it now runs the suite
  and checks that `pyproject.toml` and `edgelens.__version__` agree.
- README links were relative and broke on the PyPI project page.
- Board name kept the device-tree NUL byte (`"...Developer Kit\u0000"`).
- `INFERENCE_ON_CPU` told users on hosts without any GPU provider to install
  the Jetson onnxruntime-gpu wheel; the advice now depends on what the host has.
- Single-core `CPU_BOUND_PREPROCESS` text claimed "other cores sit idle"; it
  now reports the measured cores-busy figure.
- HTML report and console output crashed (`UnicodeEncodeError`) under a
  non-UTF-8 locale.
- A finite input source that ran out escaped as a bare `StopIteration`; it
  now says how many items were needed.
- `diagnose`/`report`/`compare` gave a traceback for a malformed or non-result
  JSON file; `--iterations 0` was accepted.
- GPU providers that onnxruntime lists but cannot load (onnxruntime-gpu without
  matching CUDA/cuDNN — found on a real laptop) were treated as available:
  `benchmark --model` crashed and `doctor` reported GPU inference as ok. EdgeLens
  now checks that CUDA really loads, picks the best provider that does, records
  skipped providers with the reason, fails loudly for an explicit `--provider`
  that would silently fall back to CPU, and `doctor` warns.

### Fixed
- Multi-input ONNX models crashed with a raw `ValueError`.
- Non-image dynamic dimensions were silently set to 224.
- A new tegrastats process was spawned for every telemetry sample.
- HTML report inserted stage names / paths / JSON unescaped.
- `GPU_BOUND` was suppressed whenever any other finding existed.
- `requirements.txt` was missing `numpy`.

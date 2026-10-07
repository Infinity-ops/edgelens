# Changelog

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

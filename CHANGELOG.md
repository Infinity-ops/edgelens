# Changelog

## 0.1.0 — unreleased (pending Jetson Nano validation, see ROADMAP.md)

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

### Fixed
- Multi-input ONNX models crashed with a raw `ValueError`.
- Non-image dynamic dimensions were silently set to 224.
- A new tegrastats process was spawned for every telemetry sample.
- HTML report inserted stage names / paths / JSON unescaped.
- `GPU_BOUND` was suppressed whenever any other finding existed.
- `requirements.txt` was missing `numpy`.

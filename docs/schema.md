# Result document: schema v1 (frozen)

Every EdgeLens run — `benchmark`, `record`, `watch`, `import-log`,
`ros2 latency`, observer mode — writes one JSON document with
`"schema_version": 1`. CI pipelines, reports and project deliverables build
on it, so its stability is a promise:

1. **Fields are only ever added within v1.** A field's name, unit and
   meaning never change. Readers must ignore fields they don't know.
2. Renaming, removing or redefining a field **bumps `schema_version`**, and
   EdgeLens keeps reading every older version.
3. A document from a *newer* schema is refused with a clear message
   ("upgrade edgelens"), never misread.
4. `tests/fixtures/result_v0_1_0.json` — written by the published
   edgelens 0.1.0 — must keep working in `diagnose`, `validate`, `compare`
   and `report`. That test is the promise in code.

## Always present (since 0.1.0)

`schema_version`, `edgelens_version`, `mode`, `pipeline_source`, `pipeline`
(`name`, `pack`, `stages[{name, role}]`, `config`), `requirements`
(`deadline_ms`, `period_ms`, `paced`), `iterations`, `latency` (end-to-end:
`n`, `min`, `mean`, `std`, `p50`, `p90`, `p95`, `p99`, `p99_9`, `max`,
`jitter`, `tail_spread`, `insufficient_samples`), `stage_stats_ms` (the same
per stage), `stage_avg_ms`, `total_latency_ms`, `fps`, `telemetry`,
`energy`, `trace`.

Percentiles are `null` when the run has too few samples to support them
(p99 needs ≥ 100, p99.9 ≥ 1000) — a percentile is never faked from the max.

## Optional, by the version that added them

| Field | Since | What |
| --- | --- | --- |
| `environment`, `identity` | 0.1.0 | board, JetPack, power mode, clocks; environment / experiment / run ids |
| `service_latency`, `deadline`, `backlog`, `pack_metrics`, `telemetry_series` | 0.1.0 | |
| `clock` | 0.2.0 | `CLOCK_MONOTONIC`, resolution, `perf_counter_is_monotonic`, wall↔monotonic offset and its uncertainty, `trace_origin_mono_ns`, `telemetry_origin_mono_ns` |
| `telemetry_series[].t_mono_ns` | 0.2.0 | absolute monotonic timestamp of each sample |
| `latency_series_ms` | 0.2.0 | per-iteration end-to-end latency (for confidence intervals, drift) |
| `markers` | 0.2.0 | `received`, `lost`, `lost_ratio`, `malformed`, `senders[]`, `gpu_timing`, `source` (`socket` / `log`) |
| `process`, `process_series` | 0.2.0 | attach mode: per-process CPU, RSS and growth, threads, context switches |
| `observation` | 0.2.0 | target, how the run ended, exit code |
| `ros2` | 0.2.0 | topics, chain matching, data age and its clock uncertainty, `latency_basis` |
| `protocol` | 0.2.0 | how the numbers were taken: `warmup_iterations`, `warmup_policy`, `iterations_requested` / `iterations_timed`, `time_box_s`, `paced`, `period_ms`, `gc_mode`, `telemetry_interval_s`, `repeats` (+ `repeat_index`, `cooldown_s` with `--repeat`) |
| `quality` | 0.2.0 | `--model --inputs --labels`: RMSE / MAE / accuracy (top-1, top-5), per level or class, `samples_predicted`, `full_pass` |
| `process_note` | 0.2.0 | `record -- CMD` when the command exited before process sampling started |
| `periodic` | 0.2.0 | `edgelens periodic` only: `back_to_back` (p50 / p99, power and its `power_sample_count`), `idle_power_w` (mean of `idle_power_before_w` and `idle_power_after_w`, with `idle_drift_w`), `periods[]` and `primary` (service p50 / p99, OS wake-up, `energy_per_period_mj`, `dynamic_energy_per_inference_mj` or a bound `dynamic_energy_below_mj` + `dynamic_energy_below_reason`, or `dynamic_energy_not_attributable`), `duty_cycle_penalty`, `gpu_railgate`, `notes` |

`mode` values: `hardware`, `demo`, `markers`, `attach`, `imported-log`,
`ros2`.

## Time base

Since 0.2.0 every timestamp is `CLOCK_MONOTONIC`. `trace.events[].start_ns`
is relative to `clock.trace_origin_mono_ns`; `telemetry_series[].t_s` is
relative to `clock.telemetry_origin_mono_ns`. Their difference aligns the
trace with telemetry exactly. Energy is integrated on the monotonic clock.

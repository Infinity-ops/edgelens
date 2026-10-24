# Assumption contracts: requirements as code (v0.2 Layer 3)

> **Status: v0.2 (Alpha).** Tested in CI. On real hardware
> ([jetson.md](jetson.md)): a `contract run` PASS on an AGX Orin 32GB;
> the deliberately violated assumptions are not yet run there.

A latency number is only meaningful together with the conditions it was
measured under. "p99 = 41 ms" on a board with pinned clocks, a cold start
and nothing else running says nothing about the same board in its default
power mode with the desktop open. An assumption contract writes both down:

```yaml
contract: vibration-anomaly-detector
version: "1.0"
requirements_doc: SRS-2.1
min_duration: 30m            # less evidence -> INCONCLUSIVE
min_iterations: 1000

assumes:                     # the conditions of the claim
  board: AGX Orin
  power_mode: MAXN
  clocks_pinned: true
  input_period_ms: {value: 1000, tolerance_pct: 5}
  max_start_temp_c: 55
  max_background_cpu_pct: 10

guarantees:                  # what the system must do under them
  p99_ms:          {limit: 3000, id: SRS-R07}
  max_miss_ratio:  {limit: 0, deadline_ms: 3000, id: SRS-R08}
  max_temp_c:      {limit: 85, id: SRS-R14}
```

```bash
edgelens contract init detector.yaml                  # commented template
edgelens contract check detector.yaml                 # typos, types; how long a run it needs
edgelens validate run.json --contract detector.yaml --evidence evidence.md
edgelens contract run detector.yaml --pipeline pipeline.py   # measure exactly what it needs, then validate
```

## The verdict (assume–guarantee)

| Assumptions | Evidence (duration, iterations) | Guarantees | Verdict | Exit |
| --- | --- | --- | --- | --- |
| all HOLD | met | all PASS | **PASS** | 0 |
| all HOLD | met | one FAIL | **FAIL** (with the limiting stage and why) | 1 |
| all HOLD | met | one NOT MEASURED | **INCONCLUSIVE** | 2 |
| one VIOLATED or NOT VERIFIED | — | anything | **INCONCLUSIVE** — never PASS | 2 |
| — | not met | anything | **INCONCLUSIVE** | 2 |

A guarantee that FAILs while an assumption is violated is reported, but the
verdict stays INCONCLUSIVE: the run measured a different situation than the
one the contract is about, so it is neither evidence for nor against it.

An assumption EdgeLens cannot verify from the run (a power mode off a Jetson,
no temperature sensor, a result from an older EdgeLens without background-load
data) is NOT VERIFIED, which also blocks a PASS: an unchecked condition is not
a held one.

## Assumptions

| Key | Holds when | Measured from |
| --- | --- | --- |
| `board` | the text is part of the board name (`AGX Orin`, `Nano`) | device tree |
| `power_mode` | nvpmodel mode name (`MAXN`) or id (`0`) equal | `nvpmodel -q` |
| `clocks_pinned` | CPU and GPU min frequency = max frequency (`true`), or both free (`false`) | cpufreq / devfreq sysfs |
| `input_period_ms` `{value, tolerance_pct}` | the run released one input per period (paced benchmark, `periodic`, `soak --period-ms`) or, for `record` / `watch`, the observed iteration rate matches | run configuration; observed rate |
| `max_start_temp_c` | the hottest thermal zone at the start of the run is at most this | first telemetry sample |
| `max_temp_c` | the run stayed inside this thermal envelope (peak) | telemetry |
| `max_background_cpu_pct` | other processes used at most this share of the whole CPU during the run | system CPU time minus EdgeLens and the measured application |
| `provider` | ONNX Runtime ran on this execution provider (no silent fallback) | session |
| `model_sha256` | the measured model file has this hash (prefix allowed) | file hash |

Notes:

- **Input period, recorded applications:** the application's own iteration
  rate is compared with the period. If the application is slower than the
  period (it falls behind), the rate cannot tell the input rate from the
  backlog, so the configured period (`record --period-ms`) is used and the row
  says so; the falling behind itself shows up in the guarantees.
- **`max_temp_c` in `assumes` vs. in `guarantees`:** as an assumption it is the
  thermal envelope the claim is made for (e.g. a climate chamber); as a
  guarantee it is something the system must achieve. Most contracts want the
  start temperature as the assumption and the peak as the guarantee.
- **Background load** counts every process except EdgeLens itself and the
  measured application (`record -- CMD`, `watch --pid`, `soak --pid`). A
  `contract run` checks it for 2 s **before** measuring, together with board,
  power mode, clocks, start temperature and provider, and stops (exit 2)
  instead of spending 30 minutes on a run that can only be INCONCLUSIVE
  (`--run-anyway` measures regardless).

## Guarantees

The same keys as the `edgelens validate` flags: `p50_ms`, `p95_ms`, `p99_ms`,
`p99_9_ms`, `max_latency_ms`, `max_miss_ratio`, `min_fps`, `max_power_w`,
`max_energy_mj`, `max_output_error`, `min_decision_agreement`, `max_temp_c`,
`max_latency_drift_pct`, `max_mem_growth_mb_per_h`, `max_throttled_windows`,
`max_energy_per_period_mj`, `max_rmse`, `min_accuracy` (with `--inputs` and
`--labels`). Each takes a number, or
`{limit: N, id: REQ-ID, note: text}`. `max_miss_ratio` also takes
`deadline_ms`: misses counted against another deadline are NOT MEASURED.

## Writing contracts that can't be weakened by accident

- Unknown keys are **errors**, with a suggestion (`clock_pinned` → "Did you
  mean 'clocks_pinned'?"). A misspelt assumption silently ignored would turn a
  conditional claim into an unconditional one.
- Every verdict records the contract file's SHA-256, its name and version: the
  evidence says exactly which contract was checked.
- `contract run` measures with the contract's period, deadline, iterations and
  duration, so the run cannot accidentally be shorter or looser than the
  contract requires. Soak guarantees (drift, memory, throttling) make it a
  soak of `min_duration`; `max_energy_per_period_mj` makes it a periodic run;
  output guarantees come from `edgelens sweep` (validate the backend file).

## Evidence record

`--evidence evidence.md` writes Markdown — paste-ready for a deliverable or a
CI job summary: the verdict, the contract (file, hash, requirements document),
the run's identity (board, power mode, run id, environment id), one row per
assumption and per requirement ID with expected, measured and status, the
measurement-integrity issues and, on FAIL, the limiting stage and the
diagnosis. `run.validation.json` holds the same as data.

For a project with a requirements document: one contract per requirement
group, `contract run` on the target board, and the `evidence.md` files are
the PASS/FAIL evidence, with the conditions they hold under stated next to
every number.

# Insights nobody else gives you (v0.2 Layer 2)

> **Status: v0.2 (Alpha).** Tested in CI. On real hardware
> ([jetson.md](jetson.md)): `periodic` measured on an AGX Orin 32GB (CPU
> provider); a 30-minute `soak` partly (its memory verdict is open); `sweep`
> across GPU backends not yet.

## Which backend? `edgelens sweep`

Speed is meaningless if the answer changed. One model, several backends,
the same inputs:

```bash
edgelens sweep --model lstm.onnx --backends cpu,cuda,tensorrt,tensorrt-fp16,dla
```

| Column | Meaning |
| --- | --- |
| Start-up | session creation, including a TensorRT engine build (cached in `~/.cache/edgelens/trt_engines`) |
| p50 / p99 | latency over `--iterations` |
| Energy/inf | energy per inference (power sensor needed) |
| (dla) | "unavailable" on boards without a DLA engine (Nano, TX2, PCs): TensorRT would otherwise run it on the GPU without saying so |
| Max \|error\| | largest difference to the reference backend's outputs (default: cpu); NaN / inf where the reference is finite = `inf` (counted as `non_finite_values`) — validate fails it |
| Decisions agree | argmax (multi-value outputs), `> threshold` (single scores in [0, 1], default 0.5, or whenever `--threshold` is given) or exact equality (integer / boolean outputs) equal to the reference |
| Verdict | **EQUIVALENT** (all values within `--atol/--rtol`, all decisions equal), **NUMERIC_DRIFT** (values differ, decisions agree), **DECISIONS_DIFFER** (decisions flipped) |

Use **real inputs** (`--inputs data.npz`, one array per model input, first
dimension = sample): agreement on random data says little about real
decisions, and the report says so. With `--labels y.npy` (one per sample)
each backend is also scored against the ground truth (RMSE, accuracy,
top-5: see [usage.md](usage.md); "vs labels" column): agreeing with the CPU is not the same as being right.
`--threads N` sets onnxruntime's intra-op threads for every backend.

**External runtimes (EDDL, TFLite, C++):** `--export-inputs inputs.npz` writes the
evaluation inputs; run them through your runtime, save its outputs as `.npz`
(one array per model output, same order), and add
`--external eddl=eddl_out.npz,timing=eddl_record.json` — the timing file is an
`edgelens record` result of the same runtime with markers.

Each backend's full result is saved next to the summary
(`NAME.<backend>.json`), so you can `validate` it:

```bash
edgelens validate sweep.tensorrt-fp16.json --p99-ms 5 --min-decision-agreement 1.0 --max-output-error 0.01
```

## The real inference rate: `edgelens periodic`

Benchmarks run back to back. A 1 Hz model wakes a sleeping board every
second: clocks have ramped down, caches are cold, the GPU may be power-gated.

```bash
edgelens periodic --model lstm.onnx --period-ms 10 --period-ms 100 --period-ms 1000
```

- **p50 / p99** = the pipeline's own work per inference (service time).
- **OS wake-up** = how late the loop started after its release time — a
  separate cost, not the model's.
- **Energy / period** = mean power x period, idle time included (what the battery pays);
  **extra / inference** is above the measured idle baseline (what the model
  costs). With inferences much shorter than the 100 ms power sampling, energy
  per inference is an average over the periods: >= 30 periods. When the extra
  power is below what the sensor resolves (max(20 mW, 1% of idle)), it is
  reported as a bound, "< N mJ", never as 0.
- The **longest** period is the reference result (`primary`) that `compare`
  and `validate` use: the real rate.
- **DUTY_CYCLE_PENALTY** when periodic p50 >= 1.5x back-to-back and >= 0.5 ms more.

Two periodic runs (e.g. without and with `sudo jetson_clocks`) in
`edgelens compare` print the trade-off both ways:

> At one inference every 1000 ms, pinning the clocks moves p99 latency from
> X to Y ms and costs Z mJ per period.

## Long runs: `edgelens soak`

```bash
edgelens soak --duration 30m --pipeline pipeline.py --period-ms 1000 --save soak.json
edgelens soak --duration 30m -- ./my_app --config app.yaml       # markers + process
```

Every window (default 60 s) is appended to `soak.jsonl` and flushed: a crash
or power cut loses at most one window (`edgelens soak --report soak.jsonl`
rebuilds the analysis). Findings:

| Finding | When |
| --- | --- |
| LATENCY_DRIFT | p50/p99 trend significant (t-test on the slope) AND >= 10% start to end |
| THERMAL_THROTTLING | a cooling device that limits frequency active (fans, Tegra voltage floors such as CORE-floor / DFLL-floor and voltage-temperature scaling such as vdd-gpu-vts are not throttling and are ignored), or the CPU/GPU clock cap (scaling_max_freq / devfreq max_freq) below 95% of the first window's — reasons listed per window. A clock that is merely low because the board is idle is not throttling |
| MEMORY_LEAK | the process's **private** memory (RssAnon) grows significantly (t-test on the slope) by at least max(5 MB/h, 1% of RSS per hour). Memory-mapped files (RssFile: a model or a data file being paged in) are reported as "Memory-mapped files" and are not a leak; without RssAnon (non-Linux) RSS is used |
| LATENCY_LEVELS | window p50s fall into two levels at least 2x apart (each seen in 2+ windows): the table names both levels, how many windows were slow, how often it switched, and the CPU clock per level — usually DVFS or another process, not the model |
| GPU_MEMORY_GROWTH | Jetson NvMap (the GPU's share of unified memory) grows — invisible in RSS |
| OUT_OF_MEMORY_RISK | available memory falls; projected to run out within 24 h |
| SWAP_IN_USE | swap grew during the run |

Requirements for soak results: `validate --max-latency-drift-pct 10
--max-mem-growth-mb-per-h 20 --max-throttled-windows 0 --max-temp-c 85`.

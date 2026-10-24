# Quick start: a reproducible EdgeLens run on any Linux machine

The workload is [quickstart.py](quickstart.py): a 2-channel vibration
sensor at 4 kHz, a new 256-sample window every 32 ms, an FFT, 32 band
features and a small linear model. NumPy only, seeded: every machine does
exactly the same work. The data is generated in the script, so there is
nothing to download and no licence to check.

```bash
python3 -m venv edgelens-try && . edgelens-try/bin/activate
pip install --upgrade pip && pip install edgelens
edgelens benchmark --pipeline quickstart.py --iterations 1000 --save run.json
edgelens diagnose run.json
edgelens validate run.json --p99-ms 32 --max-miss-ratio 0     # exit code 0 = PASS
python check_result.py run.json                                # the machine-independent parts
```

## What you should see

Latencies depend on your machine; the structure, the deadline and the
verdicts do not. Unedited output from a cloud VM (x86-64, 2 vCPUs, Python
3.12, EdgeLens 0.2.0, no Jetson):

```
   EdgeLens Benchmark (hardware mode, timeseries pack, custom pipeline   
                            (quickstart.py))                             
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Stage    ┃ Role       ┃ Mean     ┃ p99   ┃ Share                      ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ acquire  │ input      │ 0.007 ms │ 0.067 │ # 1.6%                     │
│ fft      │ preprocess │ 0.061 ms │ 0.253 │ ### 13.6%                  │
│ features │ preprocess │ 0.362 ms │ 0.971 │ #################### 80.6% │
│ model    │ inference  │ 0.018 ms │ 0.084 │ # 4.0%                     │
│ decision │ decision   │ 0.001 ms │ 0.001 │ # 0.2%                     │
└──────────┴────────────┴──────────┴───────┴────────────────────────────┘
End-to-end: mean 0.462 ms   Throughput: 2164.5 it/s   (1000 iterations)
min 0.338 · p50 0.416 · p95 0.648 · p99 1.170 · p99.9 2.855 · max 4.455 · jitter
0.197 ms
╭────────────────────────────────── Deadline ──────────────────────────────────╮
│ MET  deadline 32.0 ms · misses 0/1000 (0.000%) · worst 4.455 ms (overrun 0.0 │
│ ms) · longest burst 0                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
GC (normal): 0/0/0 collections (gen0/1/2) · longest pause 0.0 ms
Backlog at 32.0 ms period (simulated: periodic arrivals + measured service 
times): utilization 0.0144, stable=True, max backlog 0 item(s)
Real-time factor: mean 0.0144 · p99 0.0366 · max 0.1392 (service time / 32.0 ms 
period, headroom 98.56%)
Telemetry over 6 samples (0.45s, sampler 0.915 ms CPU/sample): CPU mean 42.77% /
peak 55.0% · GPU n/a · peak temp n/a
environment_id 6d0c26fa7f36 · experiment_id 53370ef858d2 · run_id 
20261024T030942-fd5256
Saved → run.json
```

- On this machine `features` takes about 80% of the time: a Python loop over
  32 bands, the stage to optimise first.
- The deadline is the hop period, 32 ms, set by the `timeseries` pack from
  the sample rate and hop; nothing in the command sets it.
- `validate` prints PASS and exits 0; `check_result.py` prints `OK: ...`.
- Without Jetson sensors, GPU, temperature and power are `n/a`: EdgeLens
  never reports a value it could not read.

## The machine-readable result

`run.json` is the full result ([docs/schema.md](../../docs/schema.md),
`schema_version: 1`): per-stage statistics, every iteration's latency, the
deadline analysis, a trace event per stage and iteration, telemetry, and the
environment (`environment_id` changes when the hardware or software does,
so `edgelens compare` can warn that two runs are not comparable).
`run.validation.json` holds the verdict of `validate`.
[check_result.py](check_result.py) lists what must hold on every machine.

## Next

- Change `features` (for example compute the 32 band means with one
  `np.add.reduceat` call), run again with `--save after.json`, and
  `edgelens compare run.json after.json`.
- Your own pipeline: replace the stages; [docs/usage.md](../../docs/usage.md)
  shows the other two ways in (a model alone, or your own loop).

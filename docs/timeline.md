# One timeline in Perfetto (v0.2 Layer 3)

> **Status: v0.2 development.** Validated against Perfetto's trace processor
> (the engine behind ui.perfetto.dev) in CI; the Jetson rows in
> [jetson.md](jetson.md) are pending.

```bash
edgelens export run.json --perfetto timeline.json
edgelens export fortis.json lstm.json --perfetto dem2.json     # several files, one clock
```

Open https://ui.perfetto.dev → *Open trace file*. The file stays in your
browser; nothing is uploaded. (chrome://tracing reads it too.)

## What is on it

| Track | From |
| --- | --- |
| one process per application, one row per thread, a span per stage | `benchmark`, `periodic`, `contract run` (stages); `record`, `watch`, soak with markers (C/C++/Python spans, real pids and tids) |
| **end-to-end iterations** per process, latency in the span's arguments | harness iterations; `@iteration` marker spans |
| **deadline miss** markers | iterations longer than the run's deadline |
| **Board telemetry**: CPU load %, GPU load %, GPU clock MHz, EMC load %, memory used %, temperature (hottest zone), board power W, **one track per power rail** | the run's telemetry series |
| Process: CPU % of one core, RSS MB, threads | `watch` / `record` of an application |

Span arguments carry the iteration number, the duration in ms and, for
marker spans, whether the GPU work was synchronized (`gpu_synced`).

## Several processes, several files

Every EdgeLens timestamp is CLOCK_MONOTONIC. Files recorded on the same boot
therefore line up exactly: record FORTIS (C++, markers) and the LSTM service
(Python, markers) with one `edgelens record` and you see the CAN decode, the
inference and the board power on one screen:

```bash
edgelens record --duration 60 --period-ms 1000 --save dem2.json &   # one collector
./fortis --config dem2.yaml &                                        # both send markers
python3 lstm_service.py
edgelens export dem2.json --perfetto dem2.perfetto.json
```

Files from different boots or machines cannot share a monotonic clock: each is
then aligned at its own start and the export says so (`--align start` forces
this, `--align mono` forces real time).

## Limits

- A result keeps the first 200 000 spans (the statistics cover all of them);
  the export notes a truncated trace.
- `soak` results hold per-window summaries, not a span series: export the
  shorter runs, or a `record` taken during the soak.
- The format is the Chrome JSON trace format. It is plain JSON, so it is easy
  to post-process, but larger than Perfetto's protobuf format.

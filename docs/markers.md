# Measuring code EdgeLens doesn't run: markers, attach mode, logs, ROS 2

> **Status: v0.2 (Alpha).** Tested in CI (C, C++ and Python senders);
> the Jetson rows in [jetson.md](jetson.md) are pending.

Most real edge AI systems are not one Python loop: C++ nodes, ROS 2,
several processes, a runtime manager in between. EdgeLens has four ways in,
from "no code changes" to "exact per-inference timing":

| You can... | Use | You get |
| --- | --- | --- |
| change nothing | `edgelens watch --pid / --name / -- cmd` | per-process CPU, memory (and its growth), threads, context switches + board power and temperature, on one timeline. **No latency.** |
| change nothing, app logs its timing | `edgelens import-log app.log --regex ...` | latency from the app's own log lines (weaker evidence: no loss accounting) |
| add a few lines | markers: `edgelens_mark.h` (C/C++), `edgelens.mark` (Python) | exact spans on CLOCK_MONOTONIC, end-to-end iterations, loss accounting, GPU-sync check |
| use ROS 2 header stamps | `edgelens ros2 latency --from /a --to /b` (experimental) | chain latency, data age, rate, jitter and gaps per topic |

All of them write the standard result document, so `diagnose`, `validate`,
`compare` and `report` work the same as for `edgelens benchmark`.

## Markers in C / C++

```cpp
#include "edgelens_mark.h"          // header-only; -I"$(edgelens include)"

while (running) {
    EL_ITERATION();                  // one whole cycle: end-to-end latency
    { EL_SPAN("build_window"); build(); }
    { EL_SPAN("lstm");          infer(); }
}
```

```bash
g++ -O2 -std=c++11 -I"$(edgelens include)" app.cpp -o app
edgelens record --deadline-ms 3000 --save run.json -- ./app --config x.yaml
edgelens validate run.json --p99-ms 3000 --max-miss-ratio 0
```

C (no RAII): `el_span_t s = el_begin("lstm"); infer(); el_end(s);`

The application may also be started separately: run `edgelens record`
without a command (it listens on `$EDGELENS_SOCKET`, default
`/tmp/edgelens.sock`, and accepts every sender) and start the application
with the same `EDGELENS_SOCKET`. Root services can send to
a collector run by a normal user (the socket is world-writable).

**Root applications:** `sudo` resets the environment, so start them with
`edgelens record -- sudo -E ./app` (or `--preserve-env=EDGELENS_SOCKET`).

**Compile-time off switch:** `-DEDGELENS_DISABLE` turns every call into
nothing, so markers can stay in production code.

### GPU work: synchronise before the end stamp

CUDA and TensorRT calls return before the GPU has finished. A span that ends
right after `enqueueV3()` measures the *enqueue*, not the inference, and
looks impossibly fast. End GPU spans after synchronising the stream:

```cpp
#define EDGELENS_WITH_CUDA
#include "edgelens_mark.h"
{ EL_SPAN_CUDA("lstm", stream); context->enqueueV3(stream); }   // syncs the stream at the end

// without the CUDA header: any sync function
{ EL_SPAN_GPU("lstm", my_sync, ctx); infer_async(); }
```

Python: `with mark.span("lstm", sync=torch.cuda.synchronize): ...`

Spans ended after a sync carry the `g` flag. When inference spans without
it are recorded while the board's GPU is busy, `record`, `diagnose` and
`validate` say so (`GPU_TIMING_NOT_SYNCED`).

## Markers in Python

```python
from edgelens import mark

while running:
    with mark.iteration():
        with mark.span("build_window"):
            x = window()
        with mark.span("lstm"):
            y = model(x)
```

## What markers guarantee

- **One clock.** Spans use `clock_gettime(CLOCK_MONOTONIC)` — the clock
  EdgeLens uses for stage timing and telemetry — so a span, a power sample
  and a temperature reading line up exactly. Wall time is never used for
  durations (it jumps when chrony corrects it).
- **The application never waits for the collector.** One UNIX datagram per
  span, sent with `MSG_DONTWAIT`; a short internal lock (spin, then yield)
  guards the overflow batch. No collector running = the call returns at once
  (and backs off for a second before trying again).
- **Bursts get through short queues.** Linux limits a UNIX datagram queue by
  message *count* (`net.unix.max_dgram_qlen`, as low as 10 on some systems;
  `edgelens selftest` shows it). When the queue is full, spans join a pending
  batch (32 KB, ~400 spans) that is sent as one datagram on the next call.
- **Losses are counted exactly — when the application exits normally.**
  Every span gets a sequence number, sent or not; the collector counts gaps,
  and the final message sent at exit makes losses at the very end visible.
  With any loss, `validate` turns a latency PASS into NOT_MEASURED (a lost
  span might have been the slow one); a FAIL stays a FAIL.
  **One exception:** a process killed by a signal it doesn't handle (Ctrl-C
  on a C++ program without a handler) skips the exit flush, so spans still
  pending inside it can't be seen. EdgeLens reports such senders
  (`SENDER_NOT_FINISHED_CLEANLY`) and blocks a PASS when the queue had been
  full during the run. Fix: handle SIGINT/SIGTERM with a flag and return from
  `main` — see `examples/markers/cpp_app.cpp`. Never call `el_flush()` from a
  signal handler.
- **fork() is safe.** A child process becomes a new sender (own id and
  sequence), in C (`pthread_atfork`) and Python (`os.register_at_fork`).
- **Only the target's spans.** `edgelens record -- CMD` gives the command a
  private socket, so other marker-enabled programs on the board can't mix in.
  `watch --pid/--name` uses the shared socket and keeps only spans from the
  target and its children (others are reported as excluded).
- **End-to-end needs @iteration across threads.** Without `@iteration`
  spans, each thread/process is cut into iterations by name repetition; if
  spans come from more than one thread, end-to-end latency is ambiguous and
  `validate` won't PASS it (`ITERATIONS_AMBIGUOUS`). Stage timings are fine.
- **Throughput is the observed rate** (iterations per second of the run),
  not 1 / latency, because the application — not EdgeLens — paces the loop.
  Energy per iteration is integrated over the active window (first to last
  span), so model loading before the loop doesn't count.
- **Cost is measured, not assumed:** `edgelens selftest` reports the cost of
  one span on your board, next to the kernel's own `sendto()` cost.

## Wire protocol (any language)

One datagram per span, plain text, to the UNIX socket:

```
EL1 X <src> <seq> <pid> <tid> <start_ns> <end_ns> <flags> <name>
EL1 F <src> <seq> <pid>                  # optional: "my last seq was <seq>"
```

`src` identifies the sender; `seq` increments for every span the sender
*tried* to send; times are CLOCK_MONOTONIC ns; `flags` is `-` or `g`; the
name `@iteration` marks one whole iteration. A datagram may hold several
lines separated by `\n`. From a shell, for a quick test:

```bash
printf 'EL1 X sh 0 %d 0 %d %d - lstm' $$ 1000 2000000 | socat - UNIX-SENDTO:/tmp/edgelens.sock
```

## Attach mode: `edgelens watch`

```bash
edgelens watch --name my_app --duration 1800 --save my_app.json   # a running process
edgelens watch --pid 4242 --duration 60
edgelens watch -- ./my_app --config x.yaml                          # start it
```

Per process (and its children, `--no-children` to exclude): CPU as % of one
core, resident memory and its **growth rate** (MB/min, least-squares), thread
count, voluntary and involuntary (preempted) context switches per second.
Board telemetry (CPU, GPU, memory, temperature, power) runs on the same
monotonic clock. If the application also sends markers, `watch` collects
them too.

What attach mode cannot do: per-inference latency (add markers), and
per-process GPU usage — Jetson's integrated GPU has no per-process
accounting, so GPU % is board-level. Process CPU is counted in 10 ms kernel
ticks, so each sample is good to about `1 / interval` percentage points
(`--process-interval`, default 0.5 s → ±2 points).

## Timing from logs: `edgelens import-log`

```bash
edgelens import-log app.log --regex 'inference took (?P<ms>[0-9.]+) ms' --stage lstm
edgelens import-log app.log --regex '(?P<stage>\w+) done in (?P<us>\d+) us'
edgelens import-log app.log --regex 'span start=(?P<start_ns>\d+) end=(?P<end_ns>\d+)'
```

Duration groups: `ms`, `us`, `ns`, `s`, or `start_ns` + `end_ns`. Evidence
is marked weaker (`TIMING_FROM_LOG`): the application measured itself.

## ROS 2: `edgelens ros2 latency` (experimental)

```bash
source /opt/ros/humble/setup.bash
edgelens ros2 latency --from /sensor/imu --to /model/prediction --duration 60 --deadline-ms 3000
```

- **Chain latency** — output arrival minus the arrival of the input with the
  same `header.stamp`. Both arrivals are on this machine's clock, so no
  clock sync is needed. Requires the output node to copy the input stamp
  (the usual ROS 2 convention).
- **Data age** — arrival minus `header.stamp`: compares the sensor
  machine's clock with this one. Exact with `--same-machine`; across
  machines its uncertainty is the sum of both chrony error bounds
  (`--source-bound-ms` from `edgelens selftest` on the sensor machine).
  Without them it is reported but INCONCLUSIVE, never a confident number.
  Messages that arrive "before they were stamped" are flagged
  (`NEGATIVE_DATA_AGE`).
- Per topic: rate, interval jitter, gaps and estimated missing messages.
  Unmatched outputs or topic gaps block a chain-latency PASS
  (`ROS_MESSAGES_MISSING`): a missing prediction may have been the slow one.

The rclpy adapter is not yet validated against a live ROS 2 system; the
measurement logic is unit-tested.

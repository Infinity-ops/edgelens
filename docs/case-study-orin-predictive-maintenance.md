# Case study: a predictive-maintenance LSTM on a Jetson AGX Orin

EdgeLens v0.2 was validated in a real deployment: an LSTM that predicts the
health of an industrial mobile robot from IMU, odometry and command data,
running at one inference per second on a Jetson AGX Orin (Yocto /
meta-tegra rootfs) in the EU project EdgeAI-Trust. The model and its data
belong to a project partner and are not published here; this page describes
the method and what EdgeLens found, so you can repeat it with your own model.

## The setup

| | |
| --- | --- |
| Board | Jetson AGX Orin, custom Yocto image (BusyBox userland, no apt / pip / venv by default), shared with another team |
| Model | ONNX LSTM, input `[1, 100, 150]` (100 time steps x 150 features), one regression output |
| Data | the partner's prepared windows (raw float32) and the target value per window |
| Rate | one window per second in deployment |
| Constraint | the board's clocks and power mode belong to the other team: nothing may change them |

## What was run

```bash
# full pass: latency per stage + accuracy against the partner's targets
edgelens benchmark --model lstm.onnx --inputs windows.bin --labels targets.bin \
    --iterations 2437 --deadline-ms 1000 --save full.json
# the same with fewer onnxruntime threads, then what changed
edgelens benchmark --model lstm.onnx --inputs windows.bin --labels targets.bin \
    --threads 4 --iterations 2437 --deadline-ms 1000 --save t4.json
edgelens compare full.json t4.json
# at the deployment rate, and for 30 minutes
edgelens periodic --model lstm.onnx --inputs windows.bin --inputs-order spread --period-ms 1000
edgelens soak --duration 30m --model lstm.onnx --inputs windows.bin --period-ms 1000
# the claim, with the conditions it holds under
edgelens contract run lstm.yaml --model lstm.onnx --inputs windows.bin --labels targets.bin
```

## What it found (and what changed in EdgeLens because of it)

- **Accuracy reproduced on the board.** The partner's reference RMSE was
  reproduced from the ONNX model on the Orin, and the EdgeLens median latency
  stayed within a few percent of a bare onnxruntime loop: the harness does
  not distort what it measures. Before this, the replay needed hand-written
  scripts; `--inputs` / `--labels` replace them.
- **Threads, not the model, set the CPU bill.** onnxruntime's default (one
  spinning thread per core) kept every core busy for a model that needs a
  few milliseconds. Fewer intra-op threads kept the latency and freed the
  CPU for the rest of the system. `--threads` is now recorded in the result,
  and `compare` reports it as a setup change.
- **1 Hz is not back-to-back.** At one inference per second the board clocks
  down between inferences and each one pays a wake-up cost that a
  back-to-back benchmark never sees; `periodic` measures it.
- **"Memory growth" was the data file.** RSS rose during the soak because the
  memory-mapped data file was being paged in. The leak check now uses the
  process's private memory (RssAnon) and reports mapped files separately.
- **Two latency levels.** The soak's windows alternated between two distinct
  latency levels instead of drifting. The soak report now names both levels,
  the number of slow windows and the CPU clock in each.
- **A rootfs without the usual tools.** No `timeout`, no `head -N`, an unset
  nvpmodel power mode: EdgeLens' checks and hints now handle a minimal
  BusyBox / Yocto userland. A contract that assumes a power mode the board
  can't report stays INCONCLUSIVE, like every assumption that can't be
  verified, never PASS.

## Inside or outside the deployed system

On the deployed board, next to the production runtime, EdgeLens only
**observes** (read-only): `watch` / `record` / `soak --pid`, ROS 2 topic
latency, `/proc` and sysfs telemetry, contracts checked on those
observations. Everything that **runs the model** (replay with `--inputs`,
accuracy, thread and backend comparison, fault campaigns, `contract run`)
runs offline: a lab board or the same board when the production stack is
stopped.

Funded by the EU project EdgeAI-Trust (Hochschule Hamm-Lippstadt).

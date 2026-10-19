# Windowed time-series pipelines: from inference latency to decision latency (v0.2 Layer 4)

> **Status: v0.2 development.** Checked against a real 30-column, 20 Hz robot
> pipeline (LSTM, 100 x 150 window) and the training code's own pandas
> preprocessing: model inputs identical to 4e-7, the same RMSE and the same
> time to detect. Jetson rows in [jetson.md](jetson.md) are pending.

A sensor model does not see a sample. It sees a **window** of preprocessed
history, and its answer is smoothed and thresholded before anyone acts on it.
So the model can answer in 2 ms while the alarm comes minutes after the event.
Measuring the model's inference time says nothing about that. These commands
measure the whole decision:

| Command | Question it answers |
| --- | --- |
| `edgelens stream check spec.yaml data/*.csv` | What does this pipeline imply (history, warm-up, smoothing delay)? Is the data what the pipeline assumes (rate, gaps, dead columns)? |
| `edgelens replay spec.yaml data/*.csv` | What does each step cost, per stage, on this board, and what does the system decide? |
| `edgelens detect runs/*.json` | How long from the event to the alarm, how many false alarms per healthy hour, how many missed? What if the threshold or smoothing changed? |
| `edgelens envelope fit / check` | Is the model answering about data it has seen in training? |
| `edgelens perturb spec.yaml data.csv --faults f.yaml` | At which sensor-fault severity does the decision break? |
| `edgelens parity spec.yaml data.csv --reference x.bin` | Does the deployed feature path compute what training computed? |

Everything is generic: column names, rates, window sizes, the scaler, the
model and the alarm rule come from one **stream spec** file. No pandas on the
device: the stream package needs NumPy (and onnxruntime for ONNX models).

## Try it (any machine, 2 minutes)

```bash
cp -r examples/stream toy && cd toy
python make_toy_robot.py toy_data                  # 6 training + 3 test runs, 20 Hz
edgelens stream check toy.stream.yaml toy_data/test/*.csv --no-model
edgelens stream fit-scaler toy.stream.yaml toy_data/train/*.csv --out scaler.json
edgelens envelope fit toy.stream.yaml toy_data/train/*.csv --out envelope.json
echo "envelope: {file: envelope.json}" >> toy.stream.yaml
edgelens replay toy.stream.yaml toy_data/test/*.csv --save-dir runs
edgelens detect runs/*.json --thresholds 0.9,0.7,0.5 --save detect.json --markdown detect.md
edgelens diagnose runs/run_101.json                          # DECISION_LATENCY, CADENCE_MISMATCH
edgelens perturb toy.stream.yaml toy_data/test/*.csv --faults faults.yaml --markdown rob.md
edgelens validate detect.json --max-time-to-detect-s 120 --max-false-alarms-per-h 0.5
```

## The stream spec

```yaml
name: motor-health
data:
  time_column: timestamp          # seconds, or 'YYYY-MM-DD HH:MM:SS.fraction'
  rate_hz: 20                     # rows per second in the TRAINING data
  columns: [imu.ax, imu.ay, ...]  # model inputs, in training order
  target: {columns: [m1, m2, m3, m4], reduce: min, scale: 0.01}   # ground truth
scaler: {file: scaler.json}       # per-column centre and scale
features:
  block: 15                       # rows per model step (non-overlapping blocks)
  stats: [min, std, max, mean, last]
  sequence: 100                   # steps per model input -> [1, 100, 150]
  padding: zeros                  # what training did before the window was full
model: {onnx: model.onnx, provider: CPUExecutionProvider}   # or python: file.py:fn
output: {slew_per_step: 0.01, mean_steps: 50}
decision: {threshold: 0.5, alarm_when: below, healthy_level: 1.0}
deployment: {period_ms: 1000}     # how often the deployed system calls the model
envelope: {file: envelope.json}
```

How the features are defined (they must be what training did, or every
number downstream is about another pipeline; `edgelens parity` checks it):

* Rows are scaled per column: `(x - center) / scale`.
* Block *k* covers rows *kW .. kW+W-1*. Each statistic is taken over the
  block's rows: `std` uses ddof=1 (pandas), and `last` is the block's last
  row. This is pandas `rolling(W)` sampled every W rows, which is how most
  training code computes it. Training code that concatenates the block's own
  row with its rolling statistics needs `last`; the model's input width tells
  you (`stream check` says so if it doesn't match).
* `layout: stat-major` (default) = all mins, then all stds, … ; `column-major`
  = all stats of column 1, then column 2, …
* `padding: zeros` = the first window is zero-padded, as `gen_sequence`-style
  training code does. Zero in scaled space is the training centre, so the
  model reports a "typical" state until the window holds real data. The
  guard marks those predictions `warming_up`.

The scaler: `edgelens stream fit-scaler` computes scikit-learn's
`RobustScaler` (or `--kind standard|minmax`) on the training recordings. If
you have the training code's fitted scaler, export it instead of refitting it:

```python
json.dump({"center": s.center_.tolist(), "scale": s.scale_.tolist(),
           "columns": var_columns}, open("scaler.json", "w"))
```

(For StandardScaler use `mean_` as the centre.) With `columns` in the file,
a scaler applied to the wrong column order is an error, not a silent
accuracy bug.

## replay: the pipeline, one block at a time

```bash
edgelens replay spec.yaml test/*.csv --save-dir runs           # as fast as possible
edgelens replay spec.yaml test.csv --speed 1 --save rt.json    # real time
edgelens replay spec.yaml test.csv --speed 20 --save x20.json  # 20x real time, paced
edgelens replay spec.yaml test.csv --features recompute        # a naive port of training code
```

Stages are `ingest → features → inference → postprocess → decision`. The
result is an ordinary EdgeLens result, so `validate`, `contract`, `compare`
and `export --perfetto` work on it. It also stores every decision with its
ground truth (`decision.series`). The deadline defaults to the deployed
period (else the trained step).

`--features incremental` computes only the new block and keeps the last
`sequence` block vectors in a ring. `recompute` recomputes every block of
the window from the raw history at each step, which is what a direct port of
batch training code does. On the 100 x 150 pipeline this is 0.07 ms against
2.2 ms per step: same numbers, 30x the cost.

Printed after the table, **where decision time goes** lists the delays that
are built into the pipeline, so you can see them before measuring anything:

| | |
| --- | --- |
| history the model sees | `sequence x step` |
| warm-up | the padded windows at the start |
| block wait | the newest row waits up to one block |
| output mean | about `(mean_steps - 1) / 2` steps of delay |
| slew floor | a slew limit of 0.01 per 0.75 s step needs at least 37 s to move from 0.98 to 0.5, however fast the model reacts |
| post-processing cost | measured: output time to detect minus the raw output's |

## detect: decisions in time

```bash
edgelens detect runs/*.json --save detect.json --markdown detect.md
edgelens detect runs/*.json --thresholds 0.9,0.8,0.7,0.6,0.5      # decision-latency curve
edgelens detect runs/*.json --mean-steps 10 --slew-per-step 0     # what-if, no re-run
```

| Metric | Definition |
| --- | --- |
| onset | first step where the truth leaves its healthy level |
| event | first step where the truth crosses the threshold |
| time to detect | first alarm at or after onset, minus the event. It is **negative** when the alarm came before the threshold was crossed: an early warning on a degrading system. |
| missed | an event with no alarm after the onset |
| false alarm | an alarm **episode** that starts while the truth is healthy, counted per healthy hour |
| early alarm | alarm time while degraded but not yet past the threshold. This is not a false alarm, and it is reported separately. |

Each metric is given for the raw model output and after post-processing, so
the price of smoothing is a number. `--save` writes a result that `validate`
and contracts read. The guarantee is the worst time to detect over the
recordings.

New requirement keys, for flags and contracts:
`max_time_to_detect_s`, `max_false_alarms_per_h`, `max_missed_events`,
`max_rmse`. New contract assumptions: `min_in_envelope_pct` and
`stream_spec_sha256`. When only decision requirements fail, the
explanation is a decision finding, not a stage bottleneck.

## envelope: answering about data it has seen

```bash
edgelens envelope fit spec.yaml train/*.csv --out envelope.json   # quantile 0.1 %, margin 10 %
edgelens envelope check spec.yaml test/*.csv                      # who leaves it, how often
```

With `envelope: {file: ...}` in the spec, every prediction gets one guard
state:

| State | Meaning |
| --- | --- |
| `valid` | every input is inside the envelope |
| `degraded` | some inputs are outside (at most `invalid_pct`, 10 % by default) |
| `invalid` | more than that are outside, or there is a NaN |
| `gap` | a time gap in the block |
| `warming_up` | the window still holds padding |

An input that was constant in training (a dead channel, the z position of a
ground robot) is "outside" as soon as it moves at all. `check` groups the
offenders by column. Position and map features are typical: they describe
*where* the robot is, not its health.

## perturb: where the decision breaks

```bash
edgelens perturb spec.yaml --init faults.yaml
edgelens perturb spec.yaml test/*.csv --faults faults.yaml --save rob.json --markdown rob.md
```

| Kind | Severity means |
| --- | --- |
| `noise` | Gaussian sigma, in units of the column's training scale (IQR) |
| `bias` | offset, in training scales |
| `gain` | multiply by (1 + severity) |
| `stuck` | the channel freezes at its value at `start_s` |
| `dropout` | probability that a row is lost and the previous value held |
| `spike` | amplitude of spikes on 1 % of rows |
| `gap` | seconds of rows removed at `start_s` |
| `rate` | rows per second relative to training (0.5 = half) |

Each fault is swept over its severities on every recording (seeded, so the
run is reproducible), scored with the detection metrics and the envelope.
The output gives the **breaking point**: the mildest severity at which a
fault is missed, a false alarm appears, or detection comes more than 10
model steps later. The fault file is plain data, so it can be shared with
partners who run the same campaign in another tool.

A pattern worth looking for: the guard drops to 0 % in-envelope at a bias
severity where the alarms are still right, and the false alarms appear one
step later. In that case the guard warns before the decision fails.

## parity: train/serve skew

```bash
edgelens parity spec.yaml scenario.csv                                # streaming vs batch
edgelens parity spec.yaml scenario.csv --reference x_scenario.bin     # training's own tensor
edgelens parity spec.yaml scenario.csv --reference-outputs y_pred.bin # e.g. a C++ runtime
```

The reference tensor is the model input the training code produced for the
same recording: `.npy`, or float32 `.bin` in C order `[steps, sequence,
width]`. The `.bin` form is what an `export_scenario_to_binary`-style script
writes for a C++ runtime. Exit code 0 means match, 1 means mismatch. For a
mismatch, the report gives the signature of the usual bugs:

* **only `std` differs, by `sqrt((W-1)/W)`:** ddof=0 (NumPy) on one side,
  ddof=1 (pandas) on the other.
* **every statistic of one column differs:** the scaler, or the column
  order.
* **the windows match when shifted by one step:** block alignment or
  padding.
* **a different number of steps:** dropped rows, or a partial last block.

## Diagnosis findings

| Finding | When |
| --- | --- |
| DECISION_LATENCY | a step takes milliseconds, the alarm comes seconds or minutes after the event; splits the time into the model's own response and the post-processing |
| CADENCE_MISMATCH | the deployed period differs from the trained step by more than 10 % (the window then covers another span of time, and per-step constants change their meaning in seconds), or the data arrive at another rate. Also for recorded applications whose pipeline config states `trained_step_ms`. |
| WARMUP_PADDING | the first windows are padding |
| OUT_OF_ENVELOPE | fewer than 99 % of predictions inside the training envelope |
| DATA_GAPS | the recording stalls; a block spans the gap |
| DEAD_INPUTS | inputs that never varied in training |

## Limits

- Replay feeds recorded rows. It measures the pipeline's compute and
  decisions on the board, not the sensor bus or ROS transport. For those,
  use `edgelens record` / `edgelens ros2` on the running system.
- The "when the event happened" depends on the ground truth's resolution:
  10 % steps of a health label place the event at the label change, not at
  the physical fault.
- Time to detect over a few recordings is a few numbers. The worst case is
  reported, and a contract on it needs more than a handful of scenarios
  before it means much.

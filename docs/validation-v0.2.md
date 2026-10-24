# Validating v0.2 on a Jetson (Nano, AGX Orin) — step by step

Release rule: **nothing goes into the README until it has a row in
[jetson.md](jetson.md) measured on real hardware.** This page is how those
rows get made. The steps are the same on the Nano and the Orin; only the
setup differs (section 6).

Every command below was run on a laptop/CI machine before release; the
Jetson-only steps are marked. Expected results are given so you can tell a
real problem from normal output.

---

## 0. If `edgelens` crashed with "Bus error" (first Nano run)

Symptom: every `edgelens benchmark` / `doctor` / `pytest` died with
`Bus error (core dumped)`, while `edgelens selftest` worked, and a CUDA
benchmark hung for hours.

Cause: `pip install -e ".[dev,onnx,rosbag]"` installed the generic PyPI
`onnxruntime` **on top of** the Jetson `onnxruntime-gpu` wheel. Both write
into the same `onnxruntime/` folder, so the folder holds a mix of two builds,
and loading the CUDA provider crashes Python. (That install line was wrong for
a Jetson; the `[onnx]` extra is for laptops only.)

Repair, in the venv you use for EdgeLens:

```bash
pip list 2>/dev/null | grep -i onnxruntime        # expect: both onnxruntime and onnxruntime-gpu
pip uninstall -y onnxruntime onnxruntime-gpu
pip uninstall -y onnxruntime onnxruntime-gpu      # twice: removes leftovers of the second one
pip install /path/to/onnxruntime_gpu-*-cp38-*-linux_aarch64.whl   # the Jetson wheel you used for v0.1
python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

Expected last line: your version and
`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`.

Since v0.2, EdgeLens can no longer be taken down by this:
the CUDA check runs in a child process, `edgelens doctor` shows a red
**"ONNX Runtime installation is broken"** panel with the exact fix, and the
validation script stops at its preflight step instead of running for hours.

### 0a. Laptop: "2 onnxruntime packages are installed over each other"

Same problem as on the Jetson, on a laptop: `onnxruntime` and `onnxruntime-gpu`
in one environment. Keep ONE:

```bash
python3 -m pip uninstall -y onnxruntime onnxruntime-gpu
python3 -m pip uninstall -y onnxruntime onnxruntime-gpu   # twice
python3 -m pip install onnxruntime        # CPU; or onnxruntime-gpu if CUDA + cuDNN are installed
```

If `pip install -e .` reports "Successfully installed edgelens-0.1.0" for a
0.2 checkout, a stale `edgelens.egg-info` from v0.1 is in the folder:
`rm -rf edgelens.egg-info && python3 -m pip install -e ".[dev,onnx,rosbag]"`,
then `python3 -c "import edgelens; print(edgelens.__version__)"` must say 0.2.

### 0b. Still "SIGBUS" with only ONE onnxruntime installed (second Nano run)

Check in this order:

1. **The SD card.** A failing card makes memory-mapped libraries (the CUDA
   provider is hundreds of MB) crash with SIGBUS — it looks exactly like a
   broken install. `sudo dmesg | grep -iE "I/O error|mmcblk|EXT4-fs error"`;
   `edgelens doctor` shows this as the **Storage** row and the validation
   script as a WARN line. With storage errors, replace the card first, and
   measure nothing on it: its numbers are not evidence.
2. **EdgeLens's check.** v0.2.0.dev1 tested CUDA by allocating a bare CUDA
   tensor, which can crash the JetPack 4.6 onnxruntime-gpu 1.11 build even
   though sessions work (TensorRT ran on this Nano in v0.1). Since v0.2
   the check creates and runs a CUDA session on a tiny model, like a real
   benchmark, and names the step that failed.
3. **The wheel.** If the session-based check still crashes on a healthy card:
   `pip install --force-reinstall --no-deps /path/to/onnxruntime_gpu-1.11.0-cp38-*-linux_aarch64.whl`.

```bash
edgelens doctor --no-cache     # test again now; expect: no red panel, Storage and GPU rows ✓
```

Storage errors are remembered per SD card (its CID), so a reboot — which empties
the kernel log — does not make a failing card look healthy; a new card starts
clean. `edgelens doctor --forget-storage-errors` clears them after you have
tested the card.

The check caches only a success, keyed by the Python interpreter, the
onnxruntime install, the CUDA driver file and the boot: a reinstall, a JetPack
upgrade, a reflash or a reboot all test again, and a failure is re-tested on
every run. `doctor` says when a result came from the cache. The validation
script continues CPU-only (WARN) when only the GPU provider is unusable; it
still stops when two onnxruntime packages are installed over each other.

## 1. Install (once per board)

```bash
cd ~/edgelens && git checkout v0.2-dev
git am /path/to/0015-*.patch /path/to/0016-*.patch /path/to/0017-*.patch   # on top of 0012-0014
git am /path/to/0019-*.patch /path/to/0020-*.patch                         # Layer 3 (after 0018)
git am /path/to/0021-*.patch /path/to/0022-*.patch /path/to/0023-*.patch   # Nano fixes
git am /path/to/0024-*.patch                                                # fourth Nano run
source ~/edgelens-venv/bin/activate

# Jetson: NO [onnx] extra (keeps the Jetson onnxruntime-gpu wheel)
pip install -e ".[dev,rosbag]"
# Laptop: pip install -e ".[dev,onnx,rosbag]"

# Jetson Nano (Ubuntu 18.04) only: if the line above fails with
#   "No matching distribution found for rosbags>=0.9.20 ... (from versions: none)"
# your Python is 3.8.0 (Ubuntu 18.04's python3.8) and every rosbags release
# declares Python >= 3.8.2. Install it with that check off, then the rest:
python3 -c "import sys; print(sys.version)"                 # 3.8.0 -> do this
pip install --ignore-requires-python "rosbags==0.9.23"
pip install -e ".[dev]"
python3 -c "from rosbags.highlevel import AnyReader; print('rosbags OK')"

sudo apt install -y g++            # C/C++ marker checks
edgelens doctor                    # must show NO red panel
```

## 2. Prepare the board for clean numbers (Jetson)

```bash
sudo systemctl isolate multi-user.target     # stop the desktop (back: graphical.target)
sudo sysctl -w net.unix.max_dgram_qlen=512   # only if selftest's dgram_queue says WARN
# Nano only, power sensor (until reboot):
sudo chmod o+r /sys/bus/i2c/drivers/ina3221x/*/iio:device*/rail_name_* \
               /sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input
```

Why the desktop matters: on the first Nano run, `selftest` measured a 1 ms
reference workload finishing up to **2 ms late** in its worst 1% — the GUI
and its services taking the CPU away. Every p99 you measure carries that
noise. selftest now reports it as `scheduling_jitter` with this fix.

## 3. Self-test (1 minute)

```bash
edgelens selftest
```

| Check | Expected on a Jetson | If not |
| --- | --- | --- |
| clock | PASS | — |
| harness_accuracy | PASS, a few µs | FAIL = something disturbs timing: stop other load |
| scheduling_jitter | PASS (< 1 ms) with the desktop stopped | WARN: do step 2 |
| python_marker_cost | PASS: µs **beyond** the kernel's own sendto() | — |
| c_marker_cost | PASS: < 5 µs beyond sendto() | — |
| marker_delivery (Python, C) | **0 lost** at 4000 spans/s | FAIL = report it (send the JSON) |
| marker_accounting | PASS (burst: received + counted lost = produced); also shows the collector's capacity | FAIL = a real bug: send the JSON |
| dgram_queue | PASS (512) | WARN: `sysctl` line in step 2 |
| sampler_cpu | ideally PASS; the detail lists the most expensive parts and slow thermal zones | WARN is informational if telemetry_disturbance is PASS |
| telemetry_disturbance | PASS (< 3%) | — |
| chrony | PASS, bound in ms | only needed for ROS 2 across machines |

What changed since the first Nano run: the collector no longer parses while
receiving (it couldn't keep up on the Nano's CPU), the C sender no longer
re-sends a 32 KB batch on every span when the queue is full (that was the
234 µs per span), the "realistic rate" is now really one (4000 spans/s,
10x a 30 FPS 10-stage pipeline), marker costs are shown beyond the kernel's
own `sendto()`, and slow thermal zones are read only every 20th sample.

## 4. The validation run

```bash
EDGELENS_VALIDATE_QUICK=1 bash scripts/validate_on_jetson.sh       # ~15 min smoke run
EDGELENS_VALIDATE_SUDO=1  bash scripts/validate_on_jetson.sh       # full: ~100 min (30 min soak, 3 x 5 min contract runs)
```

**On a board you share** (another team's clocks and power mode): run the
script **without** `EDGELENS_VALIDATE_SUDO=1` and skip step 2 (don't stop
someone else's desktop or change their sysctl). Then nothing on the board is
changed: the violated-assumption gates (E1) edit the contract, not the
board. Only `T4 clocks` and `I1 trade-off` need the board to yourself; they
are reported as SKIP, and their rows in [jetson.md](jetson.md) say "not
possible on a shared board". Write the board's state (clocks not pinned,
power mode as found) into the row.

Each run writes `edgelens_validation_<board>_<date>/SUMMARY.txt` and a
`.tar.gz`. The script:
- stops at **preflight** if onnxruntime is broken (with the fix),
- gives every command step a 15-minute timeout (the soak gets its duration
  + 5 min; small inline Python checks have none, they take seconds),
- stops the whole run on a crash (a signal, including a SIGKILL from the
  out-of-memory killer before the timeout), with the Python traceback in
  `steps.log`, instead of running every later step into the same crash.

## 5. What each SUMMARY line proves

| Gate | Done when |
| --- | --- |
| preflight | one onnxruntime install; the CUDA provider loads (or is absent) without crashing |
| T1 unit tests | all pass on the board |
| T2 one clock | results say CLOCK_MONOTONIC; telemetry samples carry `t_mono_ns` |
| T3 selftest | no FAIL |
| T4 A/A | the same setup twice: no change beyond 2% (A and B interleaved) |
| T4 A/B known change | +1 ms injected into a 2 ms stage, interleaved repeats: REGRESSED, change near +50% (the statistics find a real change) |
| T4 clocks (sudo, INFO) | free vs `jetson_clocks` for TensorRT back to back — a **measurement**, not a gate: under sustained load DVFS ramps up anyway (Nano: no change), so whether pinning helps is a property of the workload. The 1 Hz case is the I1 trade-off. Still FAIL if the runs did not really alternate clock states |
| T5 schema | the v0.1.0 file loads in diagnose / validate / compare without a crash |
| S1 markers | C++ 300 and Python 200 iterations, 0 lost; import-log parses 300 lines |
| S2 watch | process table + board power; a deliberate leak detected; markers picked up |
| S4 bag | merged table with the IMU gap found; bag-replay pipeline within 3 s |
| **I3 sweep** | every available backend timed and compared with CPU; **TensorRT FP16 drift quantified** (max \|error\|, decision agreement) |
| I3 FP16 requirements | the FP16 result passes `validate`: decisions >= 99% equal, max \|error\| <= 0.05 |
| **I1 periodic** | latency and energy per period at 1 Hz; duty-cycle penalty checked |
| **I1 trade-off** (sudo) | pinned vs unpinned at 1 Hz: latency gained AND energy per period it costs |
| **I2 leak** | the deliberate leak demo is detected by `soak` |

What a soak can and cannot see in process memory (RSS): a leak shows only
once the memory the process already holds is used up. A program that freed a
lot before the soak (or in the warm-up window) refills those pages first, and
RSS stays flat while it does. Measured on an AGX Orin: soaks started after
another run in the same process stayed flat at 166 MB for 5 windows and then
jumped +20 MB — "no leak" was the honest answer for what RSS showed. Start a
soak in a fresh process, and run it long enough that the leak outgrows the
process's free memory (EdgeLens also needs >= 8 MB and >= 5% of growth).
| **I2 soak** | 30-minute bag-replay run at 1 Hz: drift, throttling, memory, JSONL streamed |
| I2 soak requirements | the soak result passes `validate`: p99 <= 3 s, no misses, drift <= 10%, memory growth <= 20 MB/h, no throttled window |
| **E1 contract PASS** | a contract generated from this board's state: every assumption holds, every guarantee met. The script first waits up to 2 min (`EDGELENS_QUIET_WAIT_S`) for background load < 20%; if the board stays busy it names the busiest processes (INFO line) and allows that load + 15% |
| **E1 violated: …** | board, power mode, clocks (contract-side and, with sudo, physically pinned), start temperature, unpaced input, background load (CPU hogs), provider: each gives INCONCLUSIVE with exactly that assumption VIOLATED |
| **E1** | all of the above |
| **E2 export / tracks** | two applications recorded at once + the bag-replay run: spans of several processes, iterations, CPU/GPU load, temperature, power |
| E2 by hand (TODO) | open the `.perfetto.json` at https://ui.perfetto.dev and look at every track |
| **E3 CI gate** | `examples/ci-jetson/ci/perf_gate.sh` exits 0 and writes summary, evidence, timeline |
| E3 by hand (TODO) | the example repository on GitHub with this board as a runner blocks a pull request |
| S3 ROS 2 | needs a ROS 2 system (section 7) |

## 6. Nano vs Orin: same steps, different setup

| | Jetson Nano (JetPack 4.6) | AGX Orin (JetPack 5/6) |
| --- | --- | --- |
| Python | system 3.6: use your **3.8 venv** (Ubuntu's 3.8.0: rosbags needs `--ignore-requires-python`, step 1) | system 3.8 / 3.10 works |
| onnxruntime | Jetson `onnxruntime-gpu` wheel only — never `[onnx]` | the onnxruntime-gpu wheel for your JetPack |
| Power sensor | INA3221 iio, root-only → `chmod` in step 2 | INA3221 hwmon, normally readable; AGX Orin reports the **sum of module rails** (under-reads the carrier board) — write that in the row |
| GPU power-gating | `railgate` shown by `edgelens periodic` | same |
| DLA | none: the `dla` sweep row says "unavailable" | 2 DLA engines: the `dla` row is a real result |
| CPU cores | 4 | 8 (AGX Orin 32GB) or 12 (64GB): single-core-bound stages show low system CPU % — use `watch`'s per-process % of one core |

Must come out the **same** on both: every gate PASS (except skipped ones),
0 lost markers at 4000 spans/s, NO DETECTABLE CHANGE in A/A, the leak found.

## 7. Layer 2 by hand (to understand the results, and for slides)

```bash
# Which backend? speed, energy AND whether the answer changed
edgelens sweep --model tests/fixtures/lstm_anomaly.onnx --backends cpu,cuda,tensorrt,tensorrt-fp16
#   -> table: p50 / p99 / energy per inference / max |error| / decisions agree / verdict
#   real data instead of synthetic inputs:  --inputs windows.npz

# The 1 Hz headline: "your benchmark says X ms; at 1 Hz it's Y ms"
edgelens periodic --model tests/fixtures/lstm_anomaly.onnx --provider TensorrtExecutionProvider \
    --period-ms 10 --period-ms 100 --period-ms 1000 --iterations 30 --save unpinned.json
sudo jetson_clocks
edgelens periodic ... --save pinned.json
edgelens compare unpinned.json pinned.json      # trade-off: latency gained, mJ per period it costs

# 30 minutes: drift, throttling, memory (process and Jetson GPU / NvMap), swap, time to OOM
edgelens soak --duration 30m --pipeline examples/robot_bag_replay.py --period-ms 1000 --save soak.json
edgelens soak --report soak.jsonl               # rebuild after a crash / power cut
```

## 7b. Layer 3 by hand

```bash
# E1 — requirements with their conditions
edgelens contract init my.yaml && edgelens contract check my.yaml
edgelens contract run examples/ci-jetson/perf/anomaly.contract.yaml --pipeline examples/ci-jetson/pipeline.py
#   -> pre-run check (board, power mode, clocks, start temperature, background load),
#      the run the contract needs, PASS / FAIL / INCONCLUSIVE + evidence.md
edgelens validate run.json --contract my.yaml --evidence evidence.md     # any existing result

# E2 — one timeline
edgelens export run.json --perfetto run.perfetto.json     # open at https://ui.perfetto.dev

# E3 — the CI gate, locally (as the GitHub job runs it)
cd examples/ci-jetson && bash ci/perf_gate.sh
```

To finish E3 on GitHub: copy `examples/ci-jetson/` into a new **private**
repository, follow its README (runner with the `jetson` label, sudoers line
for `jetson_clocks`, power-sensor service on the Nano, branch protection),
then open a pull request that makes `infer` slower than the deadline: the
check must turn red with FAIL and the limiting stage in the job summary.

ROS 2 (when a ROS 2 system is available):

```bash
source /opt/ros/humble/setup.bash
edgelens ros2 latency --from /imu --to /prediction --same-machine --duration 60
```

## 7c. Layer 4 by hand (windowed time-series pipelines, docs/stream.md)

```bash
cp -r ~/edgelens/examples/stream ~/toy && cd ~/toy
python3 make_toy_robot.py toy_data
edgelens stream fit-scaler toy.stream.yaml toy_data/train/*.csv --out scaler.json
edgelens envelope fit toy.stream.yaml toy_data/train/*.csv --out envelope.json
echo "envelope: {file: envelope.json}" >> toy.stream.yaml
edgelens replay toy.stream.yaml toy_data/test/*.csv --save-dir runs                 # compute per step
edgelens replay toy.stream.yaml toy_data/test/run_101.csv --features recompute --save rc.json
edgelens replay toy.stream.yaml toy_data/test/run_101.csv --speed 10 --save x10.json # paced, with telemetry
edgelens detect runs/*.json --save detect.json --markdown detect.md
edgelens perturb toy.stream.yaml toy_data/test/*.csv --faults faults.yaml --markdown rob.md
edgelens export x10.json --perfetto x10.perfetto.json
```

Pass when: the board's `detect` table equals the laptop's (same time to detect
per recording: decisions must not depend on the machine), `recompute` is
slower than `incremental` per step, and the paced run has no deadline misses.
Then run the same with your real model's spec and recordings.

## 8. Write the rows

For each PASS, add a row to the v0.2 table in [jetson.md](jetson.md): board,
JetPack, power mode, clocks pinned or not, the measured value, and the file
it came from. Send the `.tar.gz` for review.

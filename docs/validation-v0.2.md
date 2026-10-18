# Validating v0.2 Layers 0 and 1 on a Jetson (Nano and AGX Orin)

Release rule: **nothing goes into the README until it has a row in
[jetson.md](jetson.md) measured on real hardware.** This page is how those
rows get made. The same script runs on the Nano and the Orin; the steps
are identical, only the setup and what the rows show differ (below).

## 0. Once per board: setup

```bash
git clone https://github.com/Infinity-ops/edgelens && cd edgelens
git checkout -b v0.2-dev                    # then apply the v0.2 patches:
git am /path/to/0012-*.patch /path/to/0013-*.patch /path/to/0014-*.patch
python3 -m venv ~/el && source ~/el/bin/activate     # Nano: python3.8 -m venv (system 3.6 is too old)
pip install -e ".[dev,onnx,rosbag]"         # Orin: ONNX Runtime GPU wheel per JetPack, see jetson.md
sudo apt install -y g++                     # for the C/C++ marker checks
edgelens doctor
edgelens selftest                           # ~1 minute; read every WARN
```

Fix what `doctor` and `selftest` tell you **before** measuring anything:

| If you see | Do | Why it matters |
| --- | --- | --- |
| power sensor root-only (Nano) | the `chmod` from `doctor` | otherwise no energy rows |
| `dgram_queue` WARN (`max_dgram_qlen` < 64) | `sudo sysctl -w net.unix.max_dgram_qlen=512` | marker bursts are batched instead of queued (still no silent loss) |
| `chrony` WARN | ignore for Layers 0/1 on one board; install chrony before cross-machine ROS 2 tests | data age across machines needs clock bounds |
| `harness_accuracy` FAIL | stop: something on the board disturbs timing (load, governor) | every other number would be wrong |

## 1. Run the gate

```bash
EDGELENS_VALIDATE_QUICK=1 bash scripts/validate_on_jetson.sh     # 5 min smoke test first
EDGELENS_VALIDATE_SUDO=1 bash scripts/validate_on_jetson.sh      # full run (~25 min), incl. jetson_clocks A/B
```

It writes `edgelens_validation_<board>_<date>/SUMMARY.txt` and a `.tar.gz`
with every result file and the raw platform dump.

## 2. What each line of SUMMARY.txt proves ("done when")

| Gate | Done when | If it fails, look at |
| --- | --- | --- |
| **T1** unit tests on the board | all pass | `steps.log` |
| **T2** one clock | results say `CLOCK_MONOTONIC`, `perf_counter_is_monotonic: true`, every telemetry sample has `t_mono_ns` | `timeseries.json → clock` |
| **T3** selftest | no FAIL. Record: harness error (µs), Python and C span cost (µs), kernel `sendto()` cost, sampler CPU (% of a core), telemetry shift (%) | `selftest.json` |
| **T4** A/A | the same setup twice gives NO DETECTABLE CHANGE | `steps.log` (verdict, change, smallest detectable) |
| **T4** A/B (sudo) | unpinned → `jetson_clocks` gives IMPROVED on p99 | `unpinned.r*.json`, `pinned.r*.json` |
| **T5** schema | the v0.1.0 file loads in diagnose / validate / compare | `steps.log` |
| **S1** C++ / Python markers | 300 / 200 iterations, 0 lost | `cpp.json`, `py.json → markers` |
| **S1** import-log | 300 lines parsed | `log.json` |
| **S2** watch | process table + board power; a deliberate 10 MB/s leak is detected | `watch.json`, `leak.json → process` |
| **S2** watch + markers | attach mode also collects spans | `watch_markers.json` |
| **S4** bag merge + DEM2 SLA | merged table with the IMU gap found; every inference within 3 s | `dem2_merged.json`, `dem2.json` |
| **S3** ROS 2 | needs a ROS 2 system (see 4.) | |

## 3. Nano vs Orin: same steps, different setup and rows

The script and the commands are the same. What differs:

| | Jetson Nano (JetPack 4.6) | AGX Orin (JetPack 5/6) |
| --- | --- | --- |
| Python | system 3.6: make a **3.8 venv** | system 3.8 (JP5) / 3.10 (JP6) works |
| Power sensor | INA3221 iio, **root-only** → `chmod` from `doctor` | INA3221 hwmon, normally readable. AGX Orin has no whole-module input rail: power is the **sum of module rails** (method `sum_of_rails`), which under-reads the carrier board — say so in the row |
| GPU load path | `/sys/devices/gpu.0/load` | `/sys/devices/platform/…gpu/load` (JP5/6) — check `doctor`'s Telemetry sources |
| Power modes | MAXN / 5W | several (`nvpmodel -q` in `platform_raw.txt`); write the mode into every row |
| CPU cores | 4 | many more: a single-core-bound stage shows a much smaller system-wide CPU % — the process table in `watch` (% of one core) is the honest view |
| DLA | none | present; not used by Layers 0/1, but the raw dump records the DLA nodes for the v0.2 DLA work |
| Expected numbers | slower clocks: higher span cost and harness error than the Orin | lower; the selftest rows will show by how much |

What must come out the **same** on both: every gate PASS (except the
skipped ones), 0 lost markers at realistic rates, exact accounting in the
burst test, NO DETECTABLE CHANGE in A/A.

## 4. ROS 2 (S3), when a ROS 2 system is available

On the robot / PMA with ROS 2 sourced:

```bash
source /opt/ros/humble/setup.bash
# same machine first: data age is exact
edgelens ros2 latency --from /imu --to /prediction --same-machine --duration 60 --save ros_local.json
# then across machines (needs chrony on both; run `edgelens selftest` on the sensor machine)
edgelens ros2 latency --from /imr/imu --to /pm/prediction --source-bound-ms <bound> --duration 60
```

Done when: chain latency matches the timing measured with markers inside
the prediction node (run both at once: `edgelens record` collects the
markers, `ros2 latency` the topics) to within a millisecond, and a node that
does not copy the input stamp is reported as "can't be matched".

## 5. Write the rows

For each PASS gate, add a row to the v0.2 table in [jetson.md](jetson.md):
board, JetPack, power mode, clocks pinned or not, the measured value, and
the run file it came from. A feature without a row stays out of the README.

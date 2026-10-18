#!/usr/bin/env bash
# Release-gate validation for EdgeLens on a real Jetson board (Nano, Orin).
#
#   bash scripts/validate_on_jetson.sh                 # from the repo root, no root needed
#   EDGELENS_VALIDATE_SUDO=1 bash scripts/validate_on_jetson.sh   # + jetson_clocks A/B
#   EDGELENS_VALIDATE_QUICK=1 bash scripts/validate_on_jetson.sh  # ~5 min smoke run
#
# Runs every hardware-dependent code path that CI cannot exercise and packs
# the outputs into edgelens_validation_<board>_<date>.tar.gz. Every v0.2
# feature has a "done when" check below; SUMMARY.txt lists each with
# PASS / FAIL / SKIP so a row can go into docs/jetson.md. See
# docs/validation-v0.2.md for what each step proves.
set -u
BOARD="$(tr -d '\0' </proc/device-tree/model 2>/dev/null | tr ' /' '__' | cut -c1-40)"
OUT="edgelens_validation_${BOARD:-unknown}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
QUICK="${EDGELENS_VALIDATE_QUICK:-0}"
REPEATS=5; ITER=2000; [ "$QUICK" = "1" ] && { REPEATS=3; ITER=300; }
log() { echo "== $*" | tee -a "$OUT/steps.log"; }
run() { log "$*"; "$@" >>"$OUT/steps.log" 2>&1; local rc=$?; echo "   exit=$rc" | tee -a "$OUT/steps.log"; return $rc; }
gate() {  # gate "<feature>" "<done-when>" <command...>
  local name="$1" what="$2"; shift 2
  if run "$@"; then echo "PASS  $name — $what" >>"$OUT/SUMMARY.txt"
  else echo "FAIL  $name — $what (see steps.log)" >>"$OUT/SUMMARY.txt"; fi
}
skip() { echo "SKIP  $1 — $2" >>"$OUT/SUMMARY.txt"; log "skip: $1 ($2)"; }
py() { python3 - "$@"; }
# expect "<allowed exit codes>" cmd...: the exit code must be one of the allowed
# ones AND the output must not contain a Python traceback (a crash exits 1 too).
expect() {
  local allowed="$1"; shift
  log "$* (allowed exit: $allowed)"
  local tmp; tmp="$(mktemp)"; "$@" >"$tmp" 2>&1; local rc=$?
  cat "$tmp" >>"$OUT/steps.log"; echo "   exit=$rc" | tee -a "$OUT/steps.log"
  if grep -q "Traceback (most recent call last)" "$tmp"; then rm -f "$tmp"; return 1; fi
  rm -f "$tmp"
  case " $allowed " in *" $rc "*) return 0 ;; *) return 1 ;; esac
}

log "raw platform sources (what the parsers are written against)"
{ echo "--- /proc/device-tree/model"; tr -d '\0' </proc/device-tree/model; echo
  echo "--- /etc/nv_tegra_release"; cat /etc/nv_tegra_release
  echo "--- dpkg nvidia-l4t-core"; dpkg-query -W nvidia-l4t-core 2>/dev/null
  echo "--- nvpmodel -q"; nvpmodel -q
  echo "--- nvpmodel modes"; grep -E "POWER_MODEL ID" /etc/nvpmodel.conf 2>/dev/null
  echo "--- /var/lib/nvpmodel/status"; cat /var/lib/nvpmodel/status
  echo "--- GPU load nodes"; for f in /sys/devices/gpu.0/load /sys/devices/platform/gpu.0/load /sys/devices/platform/bus@0/*.gpu/load /sys/devices/platform/*.gpu/load; do [ -e "$f" ] && echo "$f: $(cat "$f")"; done
  echo "--- devfreq (GPU / DLA / EMC candidates)"; for d in /sys/class/devfreq/*; do [ -d "$d" ] && echo "$d: $(cat "$d"/cur_freq 2>/dev/null) [$(cat "$d"/min_freq 2>/dev/null)-$(cat "$d"/max_freq 2>/dev/null)] gov=$(cat "$d"/governor 2>/dev/null)"; done
  echo "--- DLA / NVDLA nodes"; ls -d /sys/devices/platform/*nvdla* /sys/devices/platform/bus@0/*nvdla* /dev/nvhost-*dla* 2>/dev/null
  echo "--- INA3221 (JetPack 4 iio)"; for d in /sys/bus/i2c/drivers/ina3221x/*/iio:device*; do [ -d "$d" ] && grep -H . "$d"/rail_name_* "$d"/in_power*_input; done
  echo "--- INA3221 (JetPack 5/6 hwmon)"; for d in /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*; do [ -d "$d" ] && grep -H . "$d"/in*_label "$d"/in*_input "$d"/curr*_input; done
  echo "--- thermal zones"; for z in /sys/class/thermal/thermal_zone*; do echo "$z $(cat "$z"/type 2>/dev/null) $(cat "$z"/temp 2>/dev/null)"; done
  echo "--- cooling devices (throttling state)"; for c in /sys/class/thermal/cooling_device*; do echo "$c $(cat "$c"/type 2>/dev/null) cur=$(cat "$c"/cur_state 2>/dev/null) max=$(cat "$c"/max_state 2>/dev/null)"; done
  echo "--- cpufreq"; grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_{min,max,cur}_freq
  echo "--- UNIX datagram queue (markers)"; cat /proc/sys/net/unix/max_dgram_qlen
  echo "--- time sync"; (chronyc -c tracking || timedatectl show) 2>&1
  echo "--- compilers"; (cc --version; c++ --version) 2>&1 | head -4
  echo "--- python"; python3 --version; python3 -c "import edgelens; print('edgelens', edgelens.__version__)"
  echo "--- one tegrastats line"; timeout 3 tegrastats --interval 500 | head -1
} >"$OUT/platform_raw.txt" 2>&1

: >"$OUT/SUMMARY.txt"

# ------------------------------------------------------------------ v0.1 paths
gate "unit tests" "all pass on the board" python3 -m pytest -q
run edgelens doctor
gate "v0.1 timeseries" "runs, deadline met" edgelens benchmark --pipeline examples/timeseries_vibration.py --iterations "$ITER" --save "$OUT/timeseries.json"
for P in CPUExecutionProvider CUDAExecutionProvider TensorrtExecutionProvider; do
  run edgelens benchmark --model tests/fixtures/small_cnn.onnx --provider "$P" --iterations 300 --deadline-ms 100 --save "$OUT/${P}.json"
done

# ------------------------------------------------------------------ T2 time integrity
gate "T2 one clock" "clock=CLOCK_MONOTONIC, perf_counter on it, telemetry on it" py "$OUT/timeseries.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
c = r["clock"]
assert c["clock"] == "CLOCK_MONOTONIC" and c["perf_counter_is_monotonic"], c
assert all(s.get("t_mono_ns") for s in r["telemetry_series"]), "telemetry not on the monotonic clock"
off = (c["telemetry_origin_mono_ns"] - c["trace_origin_mono_ns"]) / 1e6
print(f"telemetry starts {off:.3f} ms before/after the trace origin; samples line up exactly")
EOF

# ------------------------------------------------------------------ T3 selftest
if [ "$QUICK" = "1" ]; then SELF="--quick"; else SELF=""; fi
gate "T3 selftest" "no FAIL (WARNs are listed in selftest.json)" edgelens selftest $SELF --save "$OUT/selftest.json"

# ------------------------------------------------------------------ T4 statistics
# A and B alternate (A1 B1 A2 B2 ...) so slow drift (temperature) hits both
# groups equally instead of looking like a change between them.
for i in $(seq -w 1 "$REPEATS"); do
  run edgelens benchmark --pipeline examples/timeseries_vibration.py --iterations "$ITER" --save "$OUT/aa_1.r$i.json"
  run edgelens benchmark --pipeline examples/timeseries_vibration.py --iterations "$ITER" --save "$OUT/aa_2.r$i.json"
done
run edgelens compare "$OUT/aa_1.r*.json" "$OUT/aa_2.r*.json" --metric p50
py "$OUT" <<'EOF' >>"$OUT/steps.log" 2>&1 && echo "PASS  T4 A/A — identical setups: no change beyond 2%" >>"$OUT/SUMMARY.txt" || echo "FAIL  T4 A/A — a change over 2% was claimed between identical setups (see steps.log)" >>"$OUT/SUMMARY.txt"
import glob, json, sys
from edgelens.compare.stats import compare_groups, NO_CHANGE
load = lambda p: [json.load(open(f)) for f in sorted(glob.glob(p))]
d = sys.argv[1]
r = compare_groups(load(d + "/aa_1.r*.json"), load(d + "/aa_2.r*.json"), primary="p50")
row = r["metrics"]["p50"]
print(r["verdict"], row.get("change_pct"), row.get("min_detectable_change_pct"))
# Pass: no detectable change, or a statistically visible one below 2% (expected
# 5% of the time by design, and from residual drift).
sys.exit(0 if r["verdict"] == NO_CHANGE or abs(row.get("change_pct") or 0) < 2.0 else 1)
EOF
if [ "${EDGELENS_VALIDATE_SUDO:-0}" = "1" ]; then
  sudo jetson_clocks --store "$OUT/clocks_before.conf"
  for i in $(seq -w 1 "$REPEATS"); do          # interleaved: unpinned, pinned, unpinned, ...
    sudo jetson_clocks --restore "$OUT/clocks_before.conf"
    run edgelens benchmark --model tests/fixtures/small_cnn.onnx --provider TensorrtExecutionProvider --iterations 300 --save "$OUT/unpinned.r$i.json"
    sudo jetson_clocks
    run edgelens benchmark --model tests/fixtures/small_cnn.onnx --provider TensorrtExecutionProvider --iterations 300 --save "$OUT/pinned.r$i.json"
  done
  sudo jetson_clocks --restore "$OUT/clocks_before.conf"
  run edgelens compare "$OUT/unpinned.r*.json" "$OUT/pinned.r*.json"
  py "$OUT" <<'EOF' && echo "PASS  T4 A/B — jetson_clocks gives IMPROVED on p99" >>"$OUT/SUMMARY.txt" || echo "FAIL  T4 A/B — expected IMPROVED (see steps.log)" >>"$OUT/SUMMARY.txt"
import glob, json, sys
from edgelens.compare.stats import compare_groups, IMPROVED
load = lambda p: [json.load(open(f)) for f in sorted(glob.glob(p))]
d = sys.argv[1]
r = compare_groups(load(d + "/unpinned.r*.json"), load(d + "/pinned.r*.json"))
print(r["verdict"], r["metrics"][r["primary_metric"]].get("change_pct"))
sys.exit(0 if r["verdict"] == IMPROVED else 1)
EOF
else
  skip "T4 A/B" "set EDGELENS_VALIDATE_SUDO=1 to run the jetson_clocks before/after"
fi

# ------------------------------------------------------------------ T5 schema
cp tests/fixtures/result_v0_1_0.json "$OUT/v010.json"     # outputs land in $OUT, not in tests/
V010="$OUT/v010.json"
if expect "0" edgelens diagnose "$V010" && expect "0 1 2" edgelens validate "$V010" --p99-ms 1000 \
   && expect "0 1" edgelens compare "$V010" "$OUT/timeseries.json"; then
  echo "PASS  T5 schema — the v0.1.0 file loads in diagnose / validate / compare (no crash)" >>"$OUT/SUMMARY.txt"
else
  echo "FAIL  T5 schema — a command crashed or failed on the v0.1.0 file (see steps.log)" >>"$OUT/SUMMARY.txt"
fi

# ------------------------------------------------------------------ S1 markers
INC="$(edgelens include)"
if command -v c++ >/dev/null; then
  run c++ -O2 -std=c++11 -I"$INC" examples/markers/cpp_app.cpp -o "$OUT/cpp_app"
  gate "S1 C++ markers" "300 iterations received, 0 lost" edgelens record --deadline-ms 50 --save "$OUT/cpp.json" -- "$OUT/cpp_app" 300
  py "$OUT/cpp.json" <<'EOF' >>"$OUT/steps.log" 2>&1 && echo "PASS  S1 C++ accounting — iterations=300, lost=0" >>"$OUT/SUMMARY.txt" || echo "FAIL  S1 C++ accounting (see steps.log)" >>"$OUT/SUMMARY.txt"
import json, sys
r = json.load(open(sys.argv[1])); m = r["markers"]
print(r["iterations"], m["received"], m["lost"], r["latency"])
sys.exit(0 if r["iterations"] == 300 and m["lost"] == 0 else 1)
EOF
else
  skip "S1 C++ markers" "no C++ compiler (sudo apt install g++)"
fi
run edgelens record --deadline-ms 50 --save "$OUT/py.json" -- python3 examples/markers/python_app.py 200
py "$OUT/py.json" <<'EOF' >>"$OUT/steps.log" 2>&1 && echo "PASS  S1 Python markers — 200 iterations, 0 lost" >>"$OUT/SUMMARY.txt" || echo "FAIL  S1 Python markers (see steps.log)" >>"$OUT/SUMMARY.txt"
import json, sys
r = json.load(open(sys.argv[1])); m = r["markers"]
print(r["iterations"], m["received"], m["lost"])
sys.exit(0 if r["iterations"] == 200 and m["lost"] == 0 else 1)
EOF
printf 'start\n' >"$OUT/app.log"; for i in $(seq 1 300); do echo "[fortis] lstm inference took $((10 + i % 7)).5 ms" >>"$OUT/app.log"; done
gate "S1 import-log" "300 lines parsed" edgelens import-log "$OUT/app.log" --regex 'took (?P<ms>[0-9.]+) ms' --stage lstm --deadline-ms 20 --save "$OUT/log.json"

# ------------------------------------------------------------------ S2 attach
gate "S2 watch command" "per-process CPU / memory / ctx switches + board power" edgelens watch --save "$OUT/watch.json" -- edgelens benchmark --pipeline examples/timeseries_vibration.py --iterations "$ITER" --save "$OUT/watched_bench.json"
python3 -c "
import time
b = []
end = time.time() + 20
while time.time() < end:
    b.append(bytearray(1_000_000)); time.sleep(0.1)
" & LEAK=$!
sleep 1
run edgelens watch --pid "$LEAK" --save "$OUT/leak.json"
py "$OUT/leak.json" <<'EOF' >>"$OUT/steps.log" 2>&1 && echo "PASS  S2 leak — deliberate 10 MB/s leak detected" >>"$OUT/SUMMARY.txt" || echo "FAIL  S2 leak (see steps.log)" >>"$OUT/SUMMARY.txt"
import json, sys
p = json.load(open(sys.argv[1]))["process"]
print(p["rss_mb_start"], p["rss_mb_end"], p["rss_growth_mb_per_min"])
sys.exit(0 if (p["rss_growth_mb_per_min"] or 0) > 300 else 1)
EOF
gate "S2 watch + markers" "attach mode also picks up spans" edgelens watch --save "$OUT/watch_markers.json" -- python3 examples/markers/python_app.py 100

# ------------------------------------------------------------------ S4 bag replay
if python3 -c "import rosbags" 2>/dev/null; then
  run python3 tests/fixtures/make_dem2_bag.py "$OUT/dem2_synthetic.bag" --seconds 30 --imu-gap 10
  gate "S4 bag merge" "merged table, IMU gap found" python3 examples/rosbag_to_windows.py "$OUT/dem2_synthetic.bag" --out "$OUT/dem2_merged.npy"
  DEM2_MERGED="$OUT/dem2_merged.npy" run edgelens benchmark --pipeline examples/dem2_bag_replay.py --iterations "$ITER" --deadline-ms 3000 --save "$OUT/dem2.json"
  gate "S4 DEM2 SLA" "every inference within 3 s" edgelens validate "$OUT/dem2.json" --p99-ms 3000 --max-miss-ratio 0
  if [ "$QUICK" != "1" ]; then
    DEM2_MERGED="$OUT/dem2_merged.npy" run edgelens benchmark --pipeline examples/dem2_bag_replay.py --iterations 60 --pace --period-ms 1000 --deadline-ms 3000 --save "$OUT/dem2_1hz.json"
  fi
  [ -n "${DEM2_BAG:-}" ] && run python3 examples/rosbag_to_windows.py "$DEM2_BAG" --out "$OUT/dem2_real.npy"
else
  skip "S4 bag" 'pip install "edgelens[rosbag]"'
fi

# ------------------------------------------------------------------ S3 ROS 2
if python3 -c "import rclpy" 2>/dev/null && [ -n "${EDGELENS_ROS_FROM:-}" ]; then
  gate "S3 ROS 2 latency" "chain latency matched" edgelens ros2 latency --from "$EDGELENS_ROS_FROM" --to "${EDGELENS_ROS_TO:-}" --duration 30 --save "$OUT/ros2.json"
else
  skip "S3 ROS 2" "needs a sourced ROS 2 + EDGELENS_ROS_FROM=/topic [EDGELENS_ROS_TO=/topic]"
fi

echo; echo "---------------- SUMMARY ----------------"; cat "$OUT/SUMMARY.txt"
tar czf "$OUT.tar.gz" "$OUT" && echo "Wrote $OUT.tar.gz — attach it to the validation issue."

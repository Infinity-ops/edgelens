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
STEP_TIMEOUT="${EDGELENS_STEP_TIMEOUT:-900}"   # no single step may run longer (s)
export PYTHONFAULTHANDLER=1                     # a crash prints WHERE it happened
log() { echo "== $*" | tee -a "$OUT/steps.log"; }
crashed() {   # stop the whole run on a crash: everything after it would be noise
  echo "!! '$1' was killed by signal $(( $2 - 128 )) ($(kill -l $(( $2 - 128 )) 2>/dev/null))." | tee -a "$OUT/steps.log"
  echo "!! The Python environment is broken; stopping instead of running every step into" | tee -a "$OUT/steps.log"
  echo "!! the same crash. The traceback is at the end of $OUT/steps.log. Run 'edgelens doctor'." | tee -a "$OUT/steps.log"
  echo "STOP  crashed in: $1 (signal $(( $2 - 128 )))" >>"$OUT/SUMMARY.txt"
  cat "$OUT/SUMMARY.txt"; tar czf "$OUT.tar.gz" "$OUT"; exit 3
}
run() {
  log "$*"
  local t0=$SECONDS timed=0
  if [ "$(type -t "$1")" = function ]; then   # shell helpers (py): timeout can't run them
    "$@" >>"$OUT/steps.log" 2>&1
  else
    timed=1
    timeout --kill-after=15 "$STEP_TIMEOUT" "$@" >>"$OUT/steps.log" 2>&1
  fi
  local rc=$? took=$(( SECONDS - t0 ))
  echo "   exit=$rc (${took}s)" | tee -a "$OUT/steps.log"
  # 124 = timeout; 137 (SIGKILL) is OUR kill only if the step really ran out of
  # time — a SIGKILL earlier is the kernel's out-of-memory killer: a crash.
  if [ "$timed" = 1 ] && { [ "$rc" = 124 ] || { [ "$rc" = 137 ] && [ "$took" -ge "$STEP_TIMEOUT" ]; }; }; then
    echo "   TIMEOUT after ${STEP_TIMEOUT}s" | tee -a "$OUT/steps.log"; return $rc
  fi
  if [ "$rc" = 137 ]; then
    echo "   KILLED (SIGKILL) after ${took}s: probably out of memory — see 'dmesg | grep -i oom'" | tee -a "$OUT/steps.log"
  fi
  if [ "$rc" = 127 ]; then echo "   COMMAND NOT FOUND: $1" | tee -a "$OUT/steps.log"; fi
  if [ "$rc" -gt 128 ]; then crashed "$*" "$rc"; fi
  return $rc
}
gate() {  # gate "<feature>" "<done-when>" <command...>
  local name="$1" what="$2"; shift 2
  if run "$@"; then echo "PASS  $name — $what" >>"$OUT/SUMMARY.txt"
  else echo "FAIL  $name — $what (see steps.log)" >>"$OUT/SUMMARY.txt"; fi
}
skip() { echo "SKIP  $1 — $2" >>"$OUT/SUMMARY.txt"; log "skip: $1 ($2)"; }
py() { python3 - "$@"; }
# ---- clock state (sudo runs). `jetson_clocks --store` saves the CURRENT state:
# on a board that is already pinned, "--restore" pins again, and every
# "unpinned" run is really pinned (found on the Nano: A/B gave NO CHANGE).
clocks_state() {   # pinned | free | mixed | unknown — read from sysfs, no root needed
  python3 -c "from edgelens.core.identity import detect_clock_locking as d; c,g=d(); print('unknown' if c is None or g is None else 'pinned' if c and g else 'free' if not c and not g else 'mixed')"
}
# shellcheck disable=SC2024  # the log is the user's file: no sudo for the redirect
clocks_pin() { sudo jetson_clocks >>"$OUT/steps.log" 2>&1; [ "$(clocks_state)" = pinned ]; }
clocks_unpin() {   # back to the power mode's own DVFS limits
  local mode; mode=$(nvpmodel -q 2>/dev/null | grep -E '^[0-9]+$' | tail -1)
  # shellcheck disable=SC2024
  [ -n "$mode" ] && sudo nvpmodel -m "$mode" >>"$OUT/steps.log" 2>&1
  [ "$(clocks_state)" = free ] && return 0
  # nvpmodel kept them pinned: lower bounds back to the hardware minimum
  for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_min_freq; do
    sudo sh -c "cat '${f%scaling_min_freq}cpuinfo_min_freq' > '$f'" 2>>"$OUT/steps.log"
  done
  for d in /sys/devices/gpu.0/devfreq/* /sys/devices/platform/*gpu*/devfreq/* /sys/devices/platform/bus@0/*gpu*/devfreq/*; do
    [ -f "$d/available_frequencies" ] || continue
    sudo sh -c "tr ' ' '\n' < '$d/available_frequencies' | grep -v '^$' | sort -n | head -1 > '$d/min_freq'" 2>>"$OUT/steps.log"
  done
  [ "$(clocks_state)" = free ]
}
clocks_restore() { case "$CLOCKS_AT_START" in pinned) clocks_pin ;; free) clocks_unpin ;; esac; }
CLOCKS_AT_START=""
if [ "${EDGELENS_VALIDATE_SUDO:-0}" = "1" ]; then
  CLOCKS_AT_START=$(clocks_state)
  trap clocks_restore EXIT
fi
# both_states <unpinned.json> <pinned.json>: the two runs really differ in clocks
both_states() {
  python3 - "$@" <<'EOF'
import glob, json, sys
def states(pattern):
    out = set()
    for f in sorted(glob.glob(pattern)):
        e = json.load(open(f)).get("environment") or {}
        out.add((e.get("cpu_clocks_locked"), e.get("gpu_clocks_locked")))
    return out
a, b = states(sys.argv[1]), states(sys.argv[2])
print("clock state (cpu, gpu locked) — unpinned runs:", a, "pinned runs:", b)
sys.exit(0 if a == {(False, False)} and b == {(True, True)} else 1)
EOF
}
# expect "<allowed exit codes>" cmd...: the exit code must be one of the allowed
# ones AND the output must not contain a Python traceback (a crash exits 1 too).
expect() {
  local allowed="$1"; shift
  log "$* (allowed exit: $allowed)"
  local tmp; tmp="$(mktemp)"; timeout --kill-after=15 "$STEP_TIMEOUT" "$@" >"$tmp" 2>&1; local rc=$?
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

# ------------------------------------------------------------------ preflight
log "preflight: storage errors in the kernel log?"
python3 - >>"$OUT/steps.log" 2>&1 <<'EOF'
from edgelens.hardware.storage import kernel_storage_errors
s = kernel_storage_errors()
print("storage:", s["count"] if s["readable"] else "kernel log not readable", s["source"])
for ln in s["lines"]:
    print("  ", ln)
raise SystemExit(4 if s.get("count") else 0)
EOF
if [ $? = 4 ]; then
  echo "WARN  storage — I/O errors in the kernel log (see steps.log): a failing SD card / disk; results from this card are NOT evidence — replace it" >>"$OUT/SUMMARY.txt"
  log "!! storage errors in the kernel log — replace the SD card before trusting results"
fi
log "preflight: is this Python environment sound?"
timeout 300 python3 - >>"$OUT/steps.log" 2>&1 <<'EOF'
import sys
from edgelens.benchmark.onnx_pipeline import cuda_probe, cuda_probe_kind, ort_install_problems
problems = ort_install_problems()
for p in problems:
    print("PROBLEM:", p["problem"]); print("FIX:    ", p["fix"])
kind = cuda_probe_kind()
print("onnxruntime CUDA probe:", kind, cuda_probe()[1] or "")
# 2 = two packages over each other (everything is unreliable): stop.
# 3 = the GPU provider is unusable, CPU fine: EdgeLens never opens a GPU
#     session then, so the run continues CPU-only and the GPU rows say why.
sys.exit(2 if problems else (3 if kind in ("crash", "hang") else 0))
EOF
PRE=$?
if [ "$PRE" != 0 ] && [ "$PRE" != 3 ]; then
  echo "!! Preflight failed: the onnxruntime installation is broken (details and the fix:" | tee -a "$OUT/SUMMARY.txt"
  echo "!! tail $OUT/steps.log, or run 'edgelens doctor'). Fix it, then rerun this script." | tee -a "$OUT/SUMMARY.txt"
  # only the check's own findings (steps.log also holds the "==" lines already on screen)
  grep -E "^(PROBLEM|FIX|onnxruntime CUDA probe):|Traceback|Error" "$OUT/steps.log" | tail -8
  exit 2
elif [ "$PRE" = 3 ]; then
  echo "WARN  preflight — GPU providers unusable (see steps.log: 'CUDA probe'); continuing CPU-only, GPU rows are SKIP/unavailable. After a fix: edgelens doctor --no-cache" >>"$OUT/SUMMARY.txt"
  log "GPU providers unusable — continuing CPU-only: $(grep 'CUDA probe' "$OUT/steps.log" | tail -1)"
else
  echo "PASS  preflight — one onnxruntime install, CUDA provider loads or is absent cleanly" >>"$OUT/SUMMARY.txt"
fi

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
if [ "${EDGELENS_VALIDATE_SUDO:-0}" = "1" ] && clocks_unpin; then
  log "clocks at start: $CLOCKS_AT_START; A/B from free clocks"
  for i in $(seq -w 1 "$REPEATS"); do          # interleaved: unpinned, pinned, unpinned, ...
    clocks_unpin
    run edgelens benchmark --model tests/fixtures/small_cnn.onnx --provider TensorrtExecutionProvider --iterations 300 --save "$OUT/unpinned.r$i.json"
    clocks_pin
    run edgelens benchmark --model tests/fixtures/small_cnn.onnx --provider TensorrtExecutionProvider --iterations 300 --save "$OUT/pinned.r$i.json"
  done
  clocks_restore
  run edgelens compare "$OUT/unpinned.r*.json" "$OUT/pinned.r*.json"
  if ! both_states "$OUT/unpinned.r*.json" "$OUT/pinned.r*.json" >>"$OUT/steps.log" 2>&1; then
    echo "FAIL  T4 A/B — the runs did not alternate between free and pinned clocks (see steps.log)" >>"$OUT/SUMMARY.txt"
  else
  py "$OUT" <<'EOF' && echo "PASS  T4 A/B — jetson_clocks gives IMPROVED on p99" >>"$OUT/SUMMARY.txt" || echo "FAIL  T4 A/B — expected IMPROVED (see steps.log)" >>"$OUT/SUMMARY.txt"
import glob, json, sys
from edgelens.compare.stats import compare_groups, IMPROVED
load = lambda p: [json.load(open(f)) for f in sorted(glob.glob(p))]
d = sys.argv[1]
r = compare_groups(load(d + "/unpinned.r*.json"), load(d + "/pinned.r*.json"))
print(r["verdict"], r["metrics"][r["primary_metric"]].get("change_pct"))
sys.exit(0 if r["verdict"] == IMPROVED else 1)
EOF
  fi
elif [ "${EDGELENS_VALIDATE_SUDO:-0}" = "1" ]; then
  skip "T4 A/B" "could not unpin the clocks (nvpmodel / sysfs); reboot so the board starts unpinned"
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

# ------------------------------------------------------------------ Layer 2: I3 backend sweep
LSTM=tests/fixtures/lstm_anomaly.onnx
gate "I3 sweep runs" "every available backend timed and compared with CPU" \
  edgelens sweep --model "$LSTM" --backends cpu,cuda,tensorrt,tensorrt-fp16,dla \
  --iterations 300 --eval-samples 100 --export-inputs "$OUT/sweep_inputs.npz" --save "$OUT/sweep.json"
py "$OUT/sweep.json" <<'EOF' >>"$OUT/steps.log" 2>&1 && echo "PASS  I3 FP16 drift — TensorRT FP16 output drift quantified vs CPU" >>"$OUT/SUMMARY.txt" || echo "SKIP  I3 FP16 drift — no TensorRT FP16 row (see sweep.json)" >>"$OUT/SUMMARY.txt"
import json, sys
s = json.load(open(sys.argv[1]))
rows = {r["backend"]: r for r in s["backends"]}
for r in s["backends"]:
    print(r["backend"], r.get("status"), r.get("verdict"), r.get("max_abs_error"),
          r.get("decision_agreement"), r.get("p50_ms"), r.get("reason", ""))
fp16 = rows.get("tensorrt-fp16") or {}
sys.exit(0 if fp16.get("status") == "ok" and fp16.get("max_abs_error") is not None else 1)
EOF
if [ -f "$OUT/sweep.tensorrt-fp16.json" ]; then
  gate "I3 FP16 requirements" "TensorRT FP16 result meets: decisions >= 99% equal, max |error| <= 0.05" \
    edgelens validate "$OUT/sweep.tensorrt-fp16.json" --min-decision-agreement 0.99 --max-output-error 0.05
fi

# ------------------------------------------------------------------ Layer 2: I1 periodic inference
if [ "$QUICK" = "1" ]; then PI=10; else PI=30; fi
run edgelens periodic --model "$LSTM" --provider CPUExecutionProvider --period-ms 10 --period-ms 100 --period-ms 1000 \
  --iterations "$PI" --save "$OUT/periodic_cpu.json"
# best provider that actually LOADS here (TensorRT > CUDA > CPU), named in the summary
TRTP=$(python3 -c "from edgelens.benchmark.onnx_pipeline import usable_providers as u; p=u(); print(next((x for x in ('TensorrtExecutionProvider','CUDAExecutionProvider') if x in p), 'CPUExecutionProvider'))")
log "periodic provider: $TRTP"
if [ "${EDGELENS_VALIDATE_SUDO:-0}" = "1" ]; then clocks_unpin || log "!! could not unpin clocks"; fi
gate "I1 periodic 1 Hz ($TRTP)" "latency + energy per period at 1 Hz, duty-cycle check" \
  edgelens periodic --model "$LSTM" --provider "$TRTP" --period-ms 1000 --iterations "$PI" --save "$OUT/periodic_unpinned.json"
if [ "${EDGELENS_VALIDATE_SUDO:-0}" = "1" ]; then
  clocks_pin
  run edgelens periodic --model "$LSTM" --provider "$TRTP" --period-ms 1000 --iterations "$PI" --save "$OUT/periodic_pinned.json"
  clocks_restore
  run edgelens compare "$OUT/periodic_unpinned.json" "$OUT/periodic_pinned.json"
  if ! both_states "$OUT/periodic_unpinned.json" "$OUT/periodic_pinned.json" >>"$OUT/steps.log" 2>&1; then
    echo "FAIL  I1 trade-off — the two runs did not differ in clock state (see steps.log)" >>"$OUT/SUMMARY.txt"
  elif tail -n 60 "$OUT/steps.log" | grep -q "Trade-off"; then
    echo "PASS  I1 trade-off — free vs pinned clocks at 1 Hz: latency AND energy per period" >>"$OUT/SUMMARY.txt"
  else
    echo "FAIL  I1 trade-off — no trade-off line (see steps.log)" >>"$OUT/SUMMARY.txt"
  fi
else
  skip "I1 trade-off" "set EDGELENS_VALIDATE_SUDO=1 for the pinned/unpinned pair"
fi

# ------------------------------------------------------------------ Layer 2: I2 soak + memory
if [ "$QUICK" = "1" ]; then LEAK_D=60s; LEAK_W=6; SOAK_D=2m; SOAK_W=15; else LEAK_D=3m; LEAK_W=15; SOAK_D=30m; SOAK_W=60; fi
run edgelens soak --duration "$LEAK_D" --window "$LEAK_W" --pipeline examples/soak_leak_demo.py --period-ms 50 --save "$OUT/soak_leak.json"
py "$OUT/soak_leak.json" <<'EOF' >>"$OUT/steps.log" 2>&1 && echo "PASS  I2 leak — deliberate leak detected by soak" >>"$OUT/SUMMARY.txt" || echo "FAIL  I2 leak — not detected (see steps.log)" >>"$OUT/SUMMARY.txt"
import json, sys
m = json.load(open(sys.argv[1]))["soak"]["memory"]
print(m)
sys.exit(0 if m["leak_suspected"] else 1)
EOF
if [ -f "$OUT/dem2_merged.npy" ]; then
  SOAK_T=$(( $(python3 -c "from edgelens.soak import parse_duration as p; print(int(p('$SOAK_D')))") + 300 ))
  EDGELENS_STEP_TIMEOUT=$SOAK_T DEM2_MERGED="$OUT/dem2_merged.npy" STEP_TIMEOUT=$SOAK_T \
    gate "I2 soak $SOAK_D" "DEM2 pipeline at 1 Hz: drift, throttling, memory, JSONL streamed" \
    edgelens soak --duration "$SOAK_D" --window "$SOAK_W" --pipeline examples/dem2_bag_replay.py --period-ms 1000 --deadline-ms 3000 --save "$OUT/soak_dem2.json"
  gate "I2 soak requirements" "p99 <= 3 s, no misses, drift <= 10%, memory growth <= 20 MB/h, no throttled window" \
    edgelens validate "$OUT/soak_dem2.json" --p99-ms 3000 --max-miss-ratio 0 --max-latency-drift-pct 10 --max-mem-growth-mb-per-h 20 --max-throttled-windows 0
else
  skip "I2 soak DEM2" "needs the S4 bag merge (pip install edgelens[rosbag])"
fi

# ------------------------------------------------------------------ Layer 3: E1 assumption contracts
# Contracts are generated from THIS board's state, so the base contract can
# PASS on a Nano and on an Orin; each variant then breaks exactly one
# assumption — physically on the board where possible — and must give
# INCONCLUSIVE (exit 2) with that assumption VIOLATED and the others holding.
CON="$OUT/contracts"; mkdir -p "$CON"
OUTABS="$(cd "$OUT" && pwd)"
if [ "$QUICK" = "1" ]; then E1_P=100; E1_D=30s; else E1_P=1000; E1_D=5m; fi
run py "$CON" "$E1_P" "$E1_D" <<'EOF'
import copy, json, os, sys
from edgelens.core.identity import capture_environment
out, period, dur = sys.argv[1], float(sys.argv[2]), sys.argv[3]
env = capture_environment()
pm = env.get("power_mode") or {}
cpu, gpu = env.get("cpu_clocks_locked"), env.get("gpu_clocks_locked")
board = (env.get("board") or "unknown").replace("NVIDIA ", "")
base = {"contract": "e1-this-board", "version": "1", "requirements_doc": "validation-v0.2",
        "min_duration": dur,
        "assumes": {"board": {"value": board, "id": "A-board"},
                    "input_period_ms": {"value": period, "id": "A-period"},
                    "max_background_cpu_pct": {"value": 20, "id": "A-load"}},
        "guarantees": {"p99_ms": {"limit": period, "id": "R-p99"},
                       "max_miss_ratio": {"limit": 0.01, "id": "R-miss"}}}
from edgelens.hardware.telemetry import read_temps
if read_temps():                       # every Jetson; not every laptop / VM
    base["assumes"]["max_start_temp_c"] = {"value": 75, "id": "A-temp"}
    base["guarantees"]["max_temp_c"] = {"limit": 95, "id": "R-temp"}
if pm.get("name"):
    base["assumes"]["power_mode"] = {"value": pm["name"], "id": "A-power"}
if cpu is not None and gpu is not None:
    base["assumes"]["clocks_pinned"] = {"value": bool(cpu and gpu), "id": "A-clocks"}
short = copy.deepcopy(base); short["min_duration"] = "30s"
short["assumes"]["input_period_ms"]["value"] = 100.0
short["guarantees"]["p99_ms"]["limit"] = 100.0
def write(name, doc):
    json.dump(doc, open(os.path.join(out, name + ".json"), "w"), indent=2)
write("base", base); write("short", short)
def variant(name, key, value):
    d = copy.deepcopy(short); d["assumes"].setdefault(key, {"id": "A-" + key})["value"] = value
    write(name, d)
variant("v_board", "board", "Raspberry Pi 5")
if "max_start_temp_c" in short["assumes"]:
    variant("v_start_temp", "max_start_temp_c", 1)
if pm.get("name"):
    variant("v_power", "power_mode", "NOT_" + pm["name"])
if "clocks_pinned" in short["assumes"]:
    variant("v_clocks_flipped", "clocks_pinned", not short["assumes"]["clocks_pinned"]["value"])
d = copy.deepcopy(short); d["assumes"]["provider"] = "TensorrtExecutionProvider"; write("v_provider", d)
print("board:", board, "| power mode:", pm.get("name"), "| clocks pinned:", cpu, gpu)
print("contracts:", sorted(os.listdir(out)))
EOF
for f in "$CON"/*.json; do run edgelens contract check "$f"; done

# e1_violated <validation.json> <assumption>: that row VIOLATED, every other one HOLDS
e1_violated() {
  py "$1" "$2" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
rows = {a["assumption"]: a for a in v["assumptions"]}
for k, a in rows.items():
    print(f"  {k:24s} {a['status']:13s} {a.get('measured')}  {a.get('reason') or ''}")
bad = rows[sys.argv[2]]["status"] == "VIOLATED"
others = all(a["status"] == "HOLDS" for k, a in rows.items() if k != sys.argv[2])
print("verdict:", v["verdict"], "| guarantees alone:", v["guarantees_verdict"])
sys.exit(0 if bad and others and v["verdict"] == "INCONCLUSIVE" else 1)
EOF
}
E1_OK=1
e1_gate() {   # e1_gate <name> <validation.json> <assumption>
  if e1_violated "$2" "$3" >>"$OUT/steps.log" 2>&1; then
    echo "PASS  E1 violated: $1 — INCONCLUSIVE, only '$3' VIOLATED" >>"$OUT/SUMMARY.txt"
  else E1_OK=0; echo "FAIL  E1 violated: $1 — expected INCONCLUSIVE with '$3' VIOLATED (see steps.log)" >>"$OUT/SUMMARY.txt"; fi
}

# PASS path (also the E3 reference run): the DEM2-style pipeline under this board's contract
gate "E1 contract PASS" "every assumption holds on this board, every guarantee met (needs a quiet board)" \
  edgelens contract run "$CON/base.json" --pipeline examples/ci-jetson/pipeline.py \
  --save "$OUT/e1_base.json" --evidence "$OUT/e1_base.evidence.md" --perfetto "$OUT/e1_base.perfetto.json"
run edgelens contract run "$CON/short.json" --pipeline examples/ci-jetson/pipeline.py --save "$OUT/e1_short.json"

# contract-side violations, checked against this board's REAL readings
for V in v_board v_start_temp v_power v_clocks_flipped; do
  [ -f "$CON/$V.json" ] || { skip "E1 violated: $V" "not readable on this board"; continue; }
  cp "$OUT/e1_short.json" "$OUT/e1_$V.json"
  expect "2" edgelens validate "$OUT/e1_$V.json" --contract "$CON/$V.json" --evidence "$OUT/e1_$V.evidence.md"
  case $V in v_board) K=board;; v_start_temp) K=max_start_temp_c;; v_power) K=power_mode;; v_clocks_flipped) K=clocks_pinned;; esac
  e1_gate "$V" "$OUT/e1_$V.validation.json" "$K"
done

# input period, physically: a back-to-back run is not "one input every period"
run edgelens benchmark --pipeline examples/ci-jetson/pipeline.py --iterations 400 --save "$OUT/e1_unpaced.json"
expect "2" edgelens validate "$OUT/e1_unpaced.json" --contract "$CON/short.json"
py "$OUT/e1_unpaced.validation.json" <<'EOF' >>"$OUT/steps.log" 2>&1 && echo "PASS  E1 violated: unpaced input — input_period_ms VIOLATED" >>"$OUT/SUMMARY.txt" || { E1_OK=0; echo "FAIL  E1 violated: unpaced input (see steps.log)" >>"$OUT/SUMMARY.txt"; }
import json, sys
v = json.load(open(sys.argv[1]))
row = next(a for a in v["assumptions"] if a["assumption"] == "input_period_ms")
print(row); sys.exit(0 if row["status"] == "VIOLATED" and v["verdict"] == "INCONCLUSIVE" else 1)
EOF

# background load, physically: CPU hogs on ~1/3 of the cores during the run
NCPU=$(nproc); NHOG=$(( NCPU / 3 + 1 )); HOGS=""
for _ in $(seq "$NHOG"); do python3 -c "while True: pass" & HOGS="$HOGS $!"; done
log "background load: $NHOG CPU hog(s) on $NCPU cores"
expect "2" edgelens contract run "$CON/short.json" --pipeline examples/ci-jetson/pipeline.py --run-anyway --save "$OUT/e1_busy.json"
# shellcheck disable=SC2086
kill $HOGS 2>/dev/null; wait $HOGS 2>/dev/null
e1_gate "background load ($NHOG hogs)" "$OUT/e1_busy.validation.json" max_background_cpu_pct

# provider, before the run: the contract wants TensorRT, the run would use the CPU
expect "2" edgelens contract run "$CON/v_provider.json" --model "$LSTM" --provider CPUExecutionProvider --save "$OUT/e1_provider.json" --evidence "$OUT/e1_provider.evidence.md"
if [ ! -f "$OUT/e1_provider.json" ] && grep -q "VIOLATED: ran on CPUExecutionProvider" "$OUT/e1_provider.evidence.md" 2>/dev/null; then
  echo "PASS  E1 violated: provider — stopped BEFORE the run, evidence says why" >>"$OUT/SUMMARY.txt"
else E1_OK=0; echo "FAIL  E1 violated: provider (see steps.log)" >>"$OUT/SUMMARY.txt"; fi

# clocks, physically (sudo): put the board in the OTHER state than the contract assumes
assumed_pinned() { python3 -c "import json,sys; a=json.load(open(sys.argv[1]))['assumes']; v=a.get('clocks_pinned', {}).get('value'); print('none' if v is None else 'pinned' if v else 'free')" "$1"; }
if [ "${EDGELENS_VALIDATE_SUDO:-0}" = "1" ] && [ "$(assumed_pinned "$CON/short.json")" != none ]; then
  if [ "$(assumed_pinned "$CON/short.json")" = pinned ]; then FLIP=clocks_unpin; WHAT="unpinned (nvpmodel)"; else FLIP=clocks_pin; WHAT="pinned by jetson_clocks"; fi
  if $FLIP; then
    expect "2" edgelens contract run "$CON/short.json" --pipeline examples/ci-jetson/pipeline.py --run-anyway --save "$OUT/e1_clocks.json"
    clocks_restore
    e1_gate "clocks $WHAT" "$OUT/e1_clocks.validation.json" clocks_pinned
  else
    clocks_restore; E1_OK=0
    echo "FAIL  E1 violated: clocks (physical) — could not change the clock state" >>"$OUT/SUMMARY.txt"
  fi
else
  skip "E1 violated: clocks (physical)" "EDGELENS_VALIDATE_SUDO=1 on a board whose clock state is readable"
fi
if [ "$E1_OK" = 1 ]; then echo "PASS  E1 — each assumption, deliberately violated, gives INCONCLUSIVE" >>"$OUT/SUMMARY.txt"
else echo "FAIL  E1 — at least one violated assumption did not give INCONCLUSIVE" >>"$OUT/SUMMARY.txt"; fi

# ------------------------------------------------------------------ Layer 3: E2 Perfetto timeline
# Two applications at once into ONE recording (several processes, one clock),
# then that recording + the DEM2 contract run (board power) on one timeline.
SOCK="$(mktemp -u /tmp/edgelens-e2-XXXXXX.sock)"     # AF_UNIX paths max 107 bytes
EDGELENS_SOCKET="$SOCK" edgelens record --duration 12 --deadline-ms 50 --save "$OUT/e2_two_apps.json" >>"$OUT/steps.log" 2>&1 &
REC=$!; sleep 2
[ -x "$OUT/cpp_app" ] && EDGELENS_SOCKET="$SOCK" "$OUT/cpp_app" 300 >>"$OUT/steps.log" 2>&1 &
EDGELENS_SOCKET="$SOCK" python3 examples/markers/python_app.py 200 >>"$OUT/steps.log" 2>&1
wait "$REC"
E2_IN=""; for f in "$OUT/e2_two_apps.json" "$OUT/e1_base.json"; do [ -f "$f" ] && E2_IN="$E2_IN $f"; done
# shellcheck disable=SC2086
gate "E2 export" "Perfetto JSON written" edgelens export $E2_IN --perfetto "$OUT/e2_timeline.perfetto.json"
gate "E2 tracks" "spans of >= 1 process per file, iterations, CPU/GPU load, temperature, power (if the sensor reads)" py "$OUT/e2_timeline.perfetto.json" "$OUT/e1_base.json" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
ev = t["traceEvents"]
pids = {e["pid"] for e in ev if e["ph"] == "X" and e.get("cat") == "stage"}
counters = {f"{e['name']} {k}" for e in ev if e["ph"] == "C" for k in e["args"]}
names = {e["name"] for e in ev if e["ph"] == "X"}
print("processes with spans:", sorted(pids)); print("counter tracks:", sorted(counters))
print("span names:", sorted(names)[:20]); print("alignment:", t["metadata"]["alignment"])
need = {"CPU load %"}
base = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
if any(s.get("max_temp_c") is not None for s in base.get("telemetry_series") or []):
    need.add("Temperature max C")
if (base.get("energy") or {}).get("available"):
    need.add("Board power W")
if any(s.get("gpu_percent") is not None for s in base.get("telemetry_series") or []):
    need.add("GPU load %")
missing = need - counters
print("missing tracks:", missing or "none")
sys.exit(0 if len(pids) >= 2 and "iteration" in names and not missing else 1)
EOF
echo "TODO  E2 by hand — open $OUT/e2_timeline.perfetto.json at https://ui.perfetto.dev: every process, iterations, counters visible" >>"$OUT/SUMMARY.txt"

# ------------------------------------------------------------------ Layer 3: E3 performance CI
# The example repository's gate script, as the CI job runs it (minus GitHub).
mkdir -p "$OUTABS/e3"
gate "E3 CI gate" "examples/ci-jetson/ci/perf_gate.sh: exit 0 (PASS), job summary + evidence + timeline written" \
  env GITHUB_STEP_SUMMARY="$OUTABS/e3/summary.md" EDGELENS_OUT="$OUTABS/e3" \
      EDGELENS_CONTRACT="$OUTABS/contracts/base.json" PYTHON=python3 \
      bash -c 'cd examples/ci-jetson && bash ci/perf_gate.sh'
ls "$OUT/e3" >>"$OUT/steps.log" 2>&1
echo "TODO  E3 by hand — push examples/ci-jetson to a private repo, register this board as a runner (its README), open a PR: the check must run and block on FAIL / INCONCLUSIVE" >>"$OUT/SUMMARY.txt"

# ------------------------------------------------------------------ S3 ROS 2
if python3 -c "import rclpy" 2>/dev/null && [ -n "${EDGELENS_ROS_FROM:-}" ]; then
  gate "S3 ROS 2 latency" "chain latency matched" edgelens ros2 latency --from "$EDGELENS_ROS_FROM" --to "${EDGELENS_ROS_TO:-}" --duration 30 --save "$OUT/ros2.json"
else
  skip "S3 ROS 2" "needs a sourced ROS 2 + EDGELENS_ROS_FROM=/topic [EDGELENS_ROS_TO=/topic]"
fi

echo; echo "---------------- SUMMARY ----------------"; cat "$OUT/SUMMARY.txt"
tar czf "$OUT.tar.gz" "$OUT" && echo "Wrote $OUT.tar.gz — attach it to the validation issue."

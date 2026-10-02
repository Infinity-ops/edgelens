#!/usr/bin/env bash
# Release-gate validation for EdgeLens on a real Jetson board.
#
#   bash scripts/validate_on_jetson.sh            # from the repo root
#
# Runs every hardware-dependent code path that CI cannot exercise and packs
# the outputs into edgelens_validation_<board>_<date>.tar.gz. Attach that
# archive to the v0.1.0 validation issue. Nothing here needs root.
set -u
OUT="edgelens_validation_$(tr -d '\0' </proc/device-tree/model 2>/dev/null | tr ' /' '__' | cut -c1-40)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
log() { echo "== $*" | tee -a "$OUT/steps.log"; }
run() { log "$*"; "$@" >>"$OUT/steps.log" 2>&1; echo "   exit=$?" | tee -a "$OUT/steps.log"; }

log "raw platform sources (what the parsers are written against)"
{ echo "--- /proc/device-tree/model"; tr -d '\0' </proc/device-tree/model; echo
  echo "--- /etc/nv_tegra_release"; cat /etc/nv_tegra_release
  echo "--- nvpmodel -q"; nvpmodel -q
  echo "--- /var/lib/nvpmodel/status"; cat /var/lib/nvpmodel/status
  echo "--- GPU load nodes"; for f in /sys/devices/gpu.0/load /sys/devices/platform/gpu.0/load /sys/devices/platform/bus@0/*.gpu/load; do [ -e "$f" ] && echo "$f: $(cat "$f")"; done
  echo "--- INA3221 (JetPack 4 iio)"; for d in /sys/bus/i2c/drivers/ina3221x/*/iio:device*; do [ -d "$d" ] && grep -H . "$d"/rail_name_* "$d"/in_power*_input; done
  echo "--- INA3221 (JetPack 5/6 hwmon)"; for d in /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*; do [ -d "$d" ] && grep -H . "$d"/in*_label "$d"/in*_input "$d"/curr*_input; done
  echo "--- cpufreq / devfreq"; grep -H . /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq
  echo "--- one tegrastats line"; timeout 3 tegrastats --interval 500 | head -1
} >"$OUT/platform_raw.txt" 2>&1

run python3 -m pytest -q
run edgelens doctor
run edgelens monitor --duration 3 --interval 0.5
run edgelens benchmark --model tests/fixtures/small_cnn.onnx --provider CPUExecutionProvider --iterations 300 --deadline-ms 100 --idle-baseline 2 --save "$OUT/cpu.json"
for P in CUDAExecutionProvider TensorrtExecutionProvider; do
  run edgelens benchmark --model tests/fixtures/small_cnn.onnx --provider "$P" --iterations 300 --deadline-ms 100 --save "$OUT/${P}.json"
done
run edgelens benchmark --model tests/fixtures/multi_input_signal.onnx --input-shape vib:1x8x2048 --iterations 200 --save "$OUT/multi_input.json"
run edgelens benchmark --pipeline examples/timeseries_vibration.py --iterations 2000 --save "$OUT/timeseries.json"
run edgelens benchmark --pipeline examples/timeseries_vibration.py --iterations 200 --period-ms 5 --pace --save "$OUT/timeseries_paced.json"
run edgelens diagnose "$OUT/cpu.json"
run edgelens report "$OUT/cpu.json" --output "$OUT/report.html"
run edgelens compare "$OUT/cpu.json" "$OUT/CUDAExecutionProvider.json"

tar czf "$OUT.tar.gz" "$OUT" && echo "Wrote $OUT.tar.gz — attach it to the validation issue."

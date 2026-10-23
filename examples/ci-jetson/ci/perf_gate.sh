#!/usr/bin/env bash
# EdgeLens performance gate: measure what the contract needs, validate against
# it, publish the evidence. The exit code IS the verdict:
#   0 PASS · 1 FAIL · 2 INCONCLUSIVE (an assumption violated / not verifiable,
#   too little evidence) — any non-zero code fails the CI job.
#
# Settings (environment variables, all optional):
#   EDGELENS_CONTRACT   contract file            (perf/anomaly.contract.yaml)
#   EDGELENS_TARGET     what to measure          (--pipeline pipeline.py)
#                       e.g. "--model model.onnx --provider TensorrtExecutionProvider"
#   EDGELENS_OUT        output folder            (edgelens-out)
#   EDGELENS_COOLDOWN_S wait up to this many seconds for the pre-run check to
#                       pass (a hot board cools down, a previous job's load ends) (0)
#   EDGELENS_BASELINE   a run.json from main: adds a before/after comparison to the
#                       summary (informational, never blocks)
#   PYTHON              interpreter with edgelens installed (python3)
set -u
CONTRACT="${EDGELENS_CONTRACT:-perf/anomaly.contract.yaml}"
TARGET="${EDGELENS_TARGET:---pipeline pipeline.py}"
OUT="${EDGELENS_OUT:-edgelens-out}"
COOLDOWN="${EDGELENS_COOLDOWN_S:-0}"
PY="${PYTHON:-python3}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"
mkdir -p "$OUT"
el() { "$PY" -m edgelens "$@"; }

el contract check "$CONTRACT" || { echo "### ❌ Contract file is invalid: $CONTRACT" >>"$SUMMARY"; exit 2; }

# Wait (bounded) until the assumptions that can be checked before the run hold.
waited=0
# shellcheck disable=SC2086
while ! el contract run "$CONTRACT" $TARGET --dry-run >/dev/null 2>&1; do
  if [ "$waited" -ge "$COOLDOWN" ]; then break; fi
  echo "pre-run check not passing yet (hot board / background load?) — waiting 15 s"
  sleep 15; waited=$((waited + 15))
done

# shellcheck disable=SC2086
el contract run "$CONTRACT" $TARGET --save "$OUT/run.json" --evidence "$OUT/evidence.md" \
  --perfetto "$OUT/timeline.perfetto.json" 2>&1 | tee "$OUT/console.txt"
rc=${PIPESTATUS[0]}

{
  if [ -f "$OUT/evidence.md" ]; then cat "$OUT/evidence.md"
  else echo "### ⚠️ No evidence record"; echo; echo '```'; tail -n 40 "$OUT/console.txt"; echo '```'
  fi
  if [ -f "$OUT/run.json" ]; then
    echo
    echo "Artifacts: \`run.json\` (result), \`run.validation.json\` (verdict), \`evidence.md\`,"
    echo "\`timeline.perfetto.json\` (open at https://ui.perfetto.dev)."
  fi
} >>"$SUMMARY"

if [ -n "${EDGELENS_BASELINE:-}" ] && [ -f "$EDGELENS_BASELINE" ] && [ -f "$OUT/run.json" ]; then
  { echo; echo "## Compared with the baseline (informational)"; echo; echo '```'
    el compare "$EDGELENS_BASELINE" "$OUT/run.json" 2>&1 | tail -n 40
    echo '```'; } >>"$SUMMARY" || true
fi
exit "$rc"

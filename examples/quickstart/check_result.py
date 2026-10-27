"""Check a quick-start result file: the parts that must be the same on every
machine (numbers like latency vary; their structure and the verdicts do not).

    python check_result.py run.json
"""
import json
import sys

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "run.json"))
problems = []


def need(cond, what):
    if not cond:
        problems.append(what)


need(r.get("schema_version") == 1, "schema_version is 1")
need(r.get("mode") == "hardware", "mode is hardware (a real measurement, not --demo)")
need([s["name"] for s in r["pipeline"]["stages"]] ==
     ["acquire", "fft", "features", "model", "decision"], "the five stages, in order")
need(r["iterations"] == 1000, "1000 iterations")
lat = r["latency"]
need(all(lat.get(k) is not None for k in ("p50", "p95", "p99")), "p50 / p95 / p99 reported")
need(lat.get("p99_9") is not None, "p99.9 reported (1000 samples)")
need(lat["min"] <= lat["p50"] <= lat["p99"] <= lat["max"], "min <= p50 <= p99 <= max")
need(r["requirements"]["deadline_ms"] == 32.0, "deadline = hop period, 32 ms")
need(r["deadline"]["iterations"] == 1000, "deadline checked on every iteration")
need(abs(sum(r["stage_avg_ms"].values()) - r["latency"]["mean"]) < 0.5 * r["latency"]["mean"] + 0.05,
     "stage times add up to the end-to-end time (harness overhead small)")
need(r["clock"]["clock"] == "CLOCK_MONOTONIC", "timestamps on CLOCK_MONOTONIC")
need(len(r["trace"]["events"]) == 5000, "one trace event per stage and iteration")
for k in ("environment_id", "experiment_id", "run_id"):
    need(bool((r.get("identity") or {}).get(k)), f"identity.{k}")

if problems:
    print("NOT OK:\n  - " + "\n  - ".join(problems))
    sys.exit(1)
print(f"OK: {r['iterations']} iterations, p50 {lat['p50']} ms, p99 {lat['p99']} ms, "
      f"deadline {'met' if r['deadline']['met'] else 'MISSED'} "
      f"({r['deadline']['misses']} misses at {r['deadline']['deadline_ms']} ms)")

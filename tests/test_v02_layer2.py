"""v0.2 Layer 2: backend sweep + output equivalence (I3), periodic inference
(I1), soak mode + memory (I2)."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import edgelens as el
from edgelens import periodic as per
from edgelens import soak as sk
from edgelens import sweep as sw
from edgelens.cli import app
from edgelens.validate.engine import validate

FIX = Path(__file__).parent / "fixtures"
LSTM = FIX / "lstm_anomaly.onnx"
CNN = FIX / "small_cnn.onnx"
runner = CliRunner()

try:
    import onnxruntime as ort
except ImportError:            # pragma: no cover
    ort = None
needs_ort = pytest.mark.skipif(ort is None, reason="onnxruntime not installed")


# ================================================================ I3 sweep

def test_compare_outputs_verdicts():
    ref = {"score": np.array([[0.2], [0.7], [0.49], [0.9]], np.float32)}
    same = sw.compare_outputs(ref, {"score": ref["score"].copy()})
    assert same["verdict"] == sw.EQUIVALENT and same["max_abs_error"] == 0.0
    assert same["decision_agreement"] == 1.0
    drift = sw.compare_outputs(ref, {"score": ref["score"] + 0.005})
    assert drift["verdict"] == sw.NUMERIC_DRIFT
    assert drift["max_abs_error"] == pytest.approx(0.005, rel=1e-3)
    flip = sw.compare_outputs(ref, {"score": ref["score"] + np.array([[0], [0], [0.02], [0]],
                                                                     np.float32)})
    assert flip["verdict"] == sw.DECISIONS_DIFFER and flip["decisions_flipped"] == 1
    assert flip["outputs"]["score"]["decision"] == "> 0.5"


def test_compare_outputs_argmax_for_class_outputs():
    ref = {"logits": np.array([[1.0, 2.0, 0.5], [3.0, 0.1, 0.2]])}
    cand = {"logits": np.array([[1.0, 0.9, 0.5], [3.0, 0.1, 0.2]])}
    r = sw.compare_outputs(ref, cand)
    assert r["outputs"]["logits"]["decision"] == "argmax"
    assert r["decision_agreement"] == 0.5 and r["verdict"] == sw.DECISIONS_DIFFER


@needs_ort
def test_sweep_cpu_reference_unavailable_backends_and_export(tmp_path):
    summary, results = sw.run_sweep(str(LSTM), ["cpu", "cuda", "tensorrt-fp16"],
                                    iterations=60, eval_samples=20,
                                    export_inputs_path=str(tmp_path / "inp.npz"),
                                    save_stem=str(tmp_path / "s"))
    rows = {r["backend"]: r for r in summary["backends"]}
    assert rows["cpu"]["status"] == "ok" and rows["cpu"]["verdict"] == sw.EQUIVALENT
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        assert rows["cuda"]["status"] == "unavailable"
    assert (tmp_path / "s.json").exists() and (tmp_path / "s.cpu.json").exists()
    inp = np.load(tmp_path / "inp.npz")
    assert inp["window"].shape == (20, 1, 50, 16)
    r = results["cpu"]
    assert r["equivalence"]["reference"] and r["backend"]["name"] == "cpu"


@needs_ort
def test_real_fp16_style_drift_is_quantified_via_external_outputs(tmp_path):
    """No GPU here: emulate what FP16 does (inputs and outputs rounded to half
    precision) with the REAL model, and feed it as an external runtime —
    exactly the path EDDL outputs take."""
    inp_path = tmp_path / "inp.npz"
    sw.run_sweep(str(LSTM), ["cpu"], iterations=20, eval_samples=40,
                 export_inputs_path=str(inp_path))
    sess = ort.InferenceSession(str(LSTM), providers=["CPUExecutionProvider"])
    x = np.load(inp_path)["window"]
    outs = np.stack([sess.run(None, {"window": (xi * 7.3).astype(np.float16)
                                     .astype(np.float32) / 7.3})[0] for xi in x])
    outs = outs.astype(np.float16).astype(np.float32)
    np.savez(tmp_path / "fp16sim.npz", score=outs)
    rec = {"latency": {"p50": 1.5, "p99": 2.0, "mean": 1.6}, "energy": {"available": False},
           "mode": "markers"}
    (tmp_path / "timing.json").write_text(json.dumps(rec))
    summary, _ = sw.run_sweep(str(LSTM), ["cpu"], iterations=20, eval_samples=40,
                              inputs_path=str(inp_path),
                              external=[("fp16sim", str(tmp_path / "fp16sim.npz"),
                                         str(tmp_path / "timing.json"))])
    row = [r for r in summary["backends"] if r["backend"] == "fp16sim"][0]
    assert row["status"] == "ok" and row["verdict"] in (sw.NUMERIC_DRIFT, sw.DECISIONS_DIFFER)
    assert 0 < row["max_abs_error"] < 0.05
    assert row["p50_ms"] == 1.5 and row["timing_source"] == "markers"
    assert summary["inputs"].startswith("file:")


@needs_ort
def test_cli_sweep_and_validate_a_backend_file(tmp_path):
    res = runner.invoke(app, ["sweep", "--model", str(LSTM), "--backends", "cpu,cuda",
                              "--iterations", "50", "--eval-samples", "10",
                              "--save", str(tmp_path / "sw.json")])
    assert res.exit_code == 0, res.output
    assert "EQUIVALENT" in res.output and "synthetic" in res.output
    f = tmp_path / "sw.cpu.json"
    res = runner.invoke(app, ["validate", str(f), "--max-output-error", "0",
                              "--min-decision-agreement", "1"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["sweep", "--model", str(LSTM), "--backends", "nope"])
    assert res.exit_code == 2 and "Unknown backend" in res.output


def test_validate_output_requirements_not_measured_without_sweep():
    r = {"stage_avg_ms": {}, "latency": {"n": 0}, "iterations": 0, "mode": "hardware"}
    v = validate(r, {"max_output_error": 1e-3})
    assert v["verdict"] == "INCONCLUSIVE" and "sweep" in v["checks"][0]["reason"]


def test_output_drift_finding():
    from edgelens.diagnose.insights import findings
    r = {"equivalence": {"verdict": "DECISIONS_DIFFER", "max_abs_error": 0.03,
                         "decision_agreement": 0.97, "decisions_flipped": 3,
                         "decisions_compared": 100, "reference_backend": "cpu",
                         "inputs": "file:x"},
         "backend": {"name": "tensorrt-fp16"}}
    f = findings(r)[0]
    assert f["type"] == "OUTPUT_DRIFT" and "3 of 100" in f["detail"]


# ================================================================ I1 periodic

def _cold_pipeline(penalty_ms=3.0, base_ms=0.5, idle_threshold_s=0.03):
    """A stage that costs more after idle — like a GPU waking up."""
    state = {"last": 0.0}
    p = el.Pipeline("cold")

    def model(x=None):
        now = time.perf_counter()
        cost = base_ms + (penalty_ms if now - state["last"] > idle_threshold_s else 0.0)
        end = now + cost / 1000.0
        while time.perf_counter() < end:
            pass
        state["last"] = time.perf_counter()
    p.add_stage("model", model)
    return p


def test_periodic_detects_duty_cycle_penalty_and_separates_wakeup():
    r = per.run_periodic(_cold_pipeline(), [60.0], iterations=12, warm_iterations=60,
                         idle_baseline_s=0)
    p = r["periodic"]
    row = p["periods"][0]
    assert p["back_to_back"]["p50_ms"] == pytest.approx(0.5, abs=0.3)
    assert row["p50_ms"] == pytest.approx(3.5, abs=0.6)
    assert row["penalty_ratio_p50"] > 3 and p["duty_cycle_penalty"]
    assert row["wake_delay_p50_ms"] is not None and row["wake_delay_p50_ms"] >= 0
    assert row["inferences"] == 12
    from edgelens.diagnose.engine import diagnose
    assert diagnose(r)["primary"]["type"] == "DUTY_CYCLE_PENALTY"


def test_periodic_without_penalty():
    r = per.run_periodic(_cold_pipeline(penalty_ms=0.0), [40.0], iterations=10,
                         warm_iterations=40, idle_baseline_s=0)
    assert not r["periodic"]["duty_cycle_penalty"]
    assert "No readable power sensor" in " ".join(r["periodic"]["notes"]) or \
        r["periodic"]["periods"][0]["energy_per_period_mj"] is not None


def test_tradeoff_both_ways():
    def res(p99, e):
        return {"periodic": {"primary": {"period_ms": 1000.0, "p99_ms": p99, "p50_ms": p99 / 2,
                                         "energy_per_period_mj": e, "power_w_mean": e / 1000}}}
    t = per.tradeoff(res(12.0, 2100.0), res(4.0, 2600.0))
    assert t["latency_delta_ms"] == -8.0 and t["energy_delta_mj_per_period"] == 500.0
    s = per.tradeoff_sentence(t, "pinning the clocks")
    assert "from 12.000 to 4.000 ms" in s and "costs 500.0 mJ per period" in s
    assert per.tradeoff(res(1, 1), {"periodic": {"primary": {"period_ms": 10}}}) is None


def test_cli_periodic_and_compare_tradeoff(tmp_path):
    script = tmp_path / "p.py"
    script.write_text(
        "import time, edgelens as el\n"
        "def build_pipeline():\n"
        "    p = el.Pipeline('x')\n"
        "    p.add_stage('model', lambda x=None: time.sleep(0.001))\n"
        "    return p\n")
    for name in ("a", "b"):
        res = runner.invoke(app, ["periodic", "--pipeline", str(script), "--period-ms", "30",
                                  "--iterations", "5", "--warm-iterations", "20",
                                  "--idle-baseline", "0", "--save", str(tmp_path / f"{name}.json")])
        assert res.exit_code == 0, res.output
        saved = json.loads((tmp_path / f"{name}.json").read_text())
        assert saved["periodic"]["periods"][0]["period_ms"] == 30.0
    res = runner.invoke(app, ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert "Trade-off" in res.output
    # a limit every board meets (30 ms at 100 W = 3000 mJ): 100 mJ FAILED on an
    # AGX Orin, rightly — it idles at ~5.7 W, ~170 mJ per 30 ms period
    res = runner.invoke(app, ["validate", str(tmp_path / "a.json"), "--max-energy-per-period-mj",
                              "3000"])
    assert res.exit_code in (0, 2)          # 2 without a power sensor: NOT_MEASURED, never PASS


# ================================================================ I2 soak

def test_parse_duration():
    assert sk.parse_duration("30m") == 1800 and sk.parse_duration("1h30m") == 5400
    assert sk.parse_duration("90s") == 90 and sk.parse_duration("45") == 45
    with pytest.raises(ValueError):
        sk.parse_duration("ten minutes")


def _rows(n=10, p50=lambda i: 5.0, rss=lambda i: 100.0, avail=lambda i: 3000.0, extra=None):
    rows = []
    for i in range(n):
        r = {"kind": "window", "window": i, "t_start_s": i * 60.0, "t_end_s": (i + 1) * 60.0,
             "iterations": 60, "p50_ms": p50(i), "p99_ms": p50(i) * 2, "max_temp_c": 50.0,
             "rss_mb": rss(i), "mem_available_mb": avail(i), "swap_used_mb": 0.0,
             "cpu_freq_mhz_mean": 1400.0, "cpu_freq_mhz_min": 1400.0, "cooling_active": {}}
        r.update((extra or (lambda i: {}))(i))
        rows.append(r)
    return rows


def test_analysis_finds_drift_leak_oom_and_throttling():
    rng = np.random.default_rng(0)
    stable = sk.analyze(_rows(p50=lambda i: 5.0 + rng.normal(0, 0.05)))
    assert not stable["drift"]["p50"]["drifting"]
    assert not stable["memory"]["leak_suspected"]
    drift = sk.analyze(_rows(p50=lambda i: 5.0 * (1 + 0.05 * i)))
    assert drift["drift"]["p50"]["drifting"] and drift["drift"]["p50"]["change_pct"] > 30
    leak = sk.analyze(_rows(rss=lambda i: 100.0 + 10 * i, avail=lambda i: 3000.0 - 300 * i))
    assert leak["memory"]["leak_suspected"]
    assert leak["memory"]["process_rss_mb_per_h"] == pytest.approx(600, rel=0.01)
    assert 0 < leak["memory"]["hours_until_out_of_memory"] < 1
    def caps(i):
        r = {"cpu_cap_mhz_min": 1479.0, "cpu_freq_mhz_min": 102.0, "max_temp_c": 50.0 + i}
        if i == 7:
            r["cooling_active"] = {"gpu-balanced": 2}
        if i == 8:
            r["cpu_cap_mhz_min"] = 921.0
        return r
    hot = sk.analyze(_rows(extra=caps))
    assert hot["throttling"]["throttled_windows"] == 2
    reasons = " ".join(" ".join(w["reasons"]) for w in hot["throttling"]["windows"])
    assert "gpu-balanced" in reasons and "CPU maximum clock lowered to 921.0" in reasons
    # idle CPU at a low current clock while warming up is NOT throttling (review #5)
    idle = sk.analyze(_rows(extra=lambda i: {"cpu_cap_mhz_min": 1479.0, "cpu_freq_mhz_min": 102.0,
                                             "cpu_freq_mhz_mean": 900.0, "max_temp_c": 40.0 + i}))
    assert idle["throttling"]["throttled_windows"] == 0


def test_soak_jsonl_survives_a_crash(tmp_path):
    p = tmp_path / "s.jsonl"
    lines = [json.dumps({"kind": "header", "deadline_ms": 100})] + \
        [json.dumps(r) for r in _rows(n=8, rss=lambda i: 100.0 + 20 * i)]
    p.write_text("\n".join(lines) + '\n{"kind": "window", "window": 8, "t_st')   # torn last line
    r = sk.report_from_jsonl(str(p))
    assert r["soak"]["windows"] == 8 and r["soak"]["memory"]["leak_suspected"]
    v = validate(r, {"max_mem_growth_mb_per_h": 10, "p99_ms": 10})
    assert v["verdict"] == "FAIL"
    res = runner.invoke(app, ["soak", "--report", str(p), "--save", str(tmp_path / "r.json")])
    assert res.exit_code == 0, res.output and "LEAK SUSPECTED" in res.output


def test_soak_pipeline_detects_a_real_leak(tmp_path):
    keep = []
    p = el.Pipeline("leaky")

    def model(x=None):
        keep.append(bytearray(400_000))
        time.sleep(0.002)
    p.add_stage("model", model)
    # 9 s, not 7: a window costs its 1 s plus set-up (telemetry, the process
    # scan for background load). 7 s gave 6 windows here but 5 on an AGX Orin
    # in 4 runs of 5, and the test needs >= 6. 9 s gives >= 6 up to 1.5 s/window.
    r = sk.run_soak_pipeline(p, duration_s=9.0, window_s=1.0, period_ms=20,
                             jsonl_path=str(tmp_path / "s.jsonl"))
    s = r["soak"]
    assert s["windows"] >= 6
    assert s["memory"]["leak_suspected"], s["memory"]
    assert r["iterations"] > 200 and r["requirements"]["deadline_ms"] == 20
    lines = (tmp_path / "s.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "header" and len(lines) == s["windows"] + 1
    from edgelens.diagnose.engine import diagnose
    assert "MEMORY_LEAK" in [f["type"] for f in diagnose(r)["all_findings"]]
    v = validate(r, {"max_mem_growth_mb_per_h": 50})
    assert v["verdict"] == "FAIL"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_soak_observed_app_with_markers(tmp_path):
    app_py = tmp_path / "a.py"
    app_py.write_text("import time\nfrom edgelens import mark\n"
                      "end = time.time() + 6\n"
                      "while time.time() < end:\n"
                      "    with mark.iteration():\n"
                      "        with mark.span('lstm'): time.sleep(0.004)\n"
                      "    time.sleep(0.02)\n")
    r = sk.run_soak_observed(4.0, window_s=1.0, cmd=[sys.executable, str(app_py)],
                             deadline_ms=50, jsonl_path=str(tmp_path / "o.jsonl"))
    assert r["soak"]["windows"] >= 3
    assert r["iterations"] > 60 and r["deadline"]["misses"] == 0
    assert r["process"]["samples"] >= 2


def test_cli_soak_argument_checks():
    res = runner.invoke(app, ["soak", "--duration", "1m"])
    assert res.exit_code == 2 and "either" in res.output
    res = runner.invoke(app, ["soak", "--pipeline", "x.py"])
    assert res.exit_code == 2 and "--duration" in res.output


def _sourced_pipeline():
    """Like the DEM2 replay: the pipeline reads its own recorded data."""
    p = el.Pipeline("sourced", pack="timeseries", sample_rate_hz=100, window=10, hop=20)
    p.set_source(el.WindowSource(np.arange(1000, dtype=float), window=10, hop=20))
    p.add_stage("build_window", lambda w: float(w[0]) + 1)     # fails on None input
    return p


def test_periodic_and_soak_use_the_pipelines_own_source(tmp_path):
    r = per.run_periodic(_sourced_pipeline(), [20.0], iterations=5, warm_iterations=10,
                         idle_baseline_s=0)
    assert r["periodic"]["periods"][0]["inferences"] == 5
    r = sk.run_soak_pipeline(_sourced_pipeline(), duration_s=2.0, window_s=1.0, period_ms=50,
                             jsonl_path=str(tmp_path / "s.jsonl"))
    assert r["iterations"] > 20


def test_validation_script_run_helper_handles_shell_functions(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_on_jetson.sh"
    text = script.read_text()
    helpers = text[text.index("STEP_TIMEOUT="):text.index("gate() {")]
    h = tmp_path / "h.sh"
    h.write_text("set -u\nOUT=%s\nmkdir -p $OUT\n: > $OUT/SUMMARY.txt\n%s\n"
                 "py() { python3 - \"$@\"; }\n"
                 "run py <<'EOF'\nprint('helper ran')\nEOF\necho rc=$?\n" % (tmp_path / "o", helpers))
    import subprocess
    res = subprocess.run(["bash", str(h)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)
    assert "rc=0" in res.stdout, res.stdout
    assert "helper ran" in (tmp_path / "o" / "steps.log").read_text()


def test_short_soak_warm_up_growth_is_not_a_leak():
    """AGX Orin, 5-minute soak of a pipeline WITHOUT a leak: RSS rose in small
    warm-up steps (75.5 -> 76.4 MB, a significant slope of ~+11.6 MB/h) and
    was reported as LEAK SUSPECTED. Real growth was 0.9 MB."""
    steps = [74.74, 75.53, 75.79, 75.79, 75.79, 75.79, 75.79, 75.79, 75.92, 76.05,
             76.05, 76.18, 76.18, 76.18, 76.18, 76.31, 76.44, 76.44, 76.44, 76.44]
    rows = _rows(n=20, rss=lambda i: steps[i])
    for i, r in enumerate(rows):                      # 15 s windows, as on the Orin
        r["t_start_s"], r["t_end_s"] = i * 15.0, (i + 1) * 15.0
    m = sk.analyze(rows)["memory"]
    assert m["process_rss_mb_per_h_significant"] and m["process_rss_mb_per_h"] > 5
    assert not m["leak_suspected"], m
    assert m["rss_growing_inconclusive"] and m["process_rss_mb_grown"] < 2


def test_orin_leak_demo_is_still_a_leak():
    """The same Orin, examples/soak_leak_demo.py: ~1 MB/s for 3 minutes."""
    rss = [83.27, 99.0, 114.46, 129.8, 145.26, 160.86, 176.33, 191.93, 207.26, 222.73,
           238.19, 252.22]
    rows = _rows(n=12, rss=lambda i: rss[i])
    for i, r in enumerate(rows):
        r["t_start_s"], r["t_end_s"] = i * 15.0, (i + 1) * 15.0
    m = sk.analyze(rows)["memory"]
    assert m["leak_suspected"] and not m["rss_growing_inconclusive"]
    assert m["process_rss_mb_per_h"] == pytest.approx(3682, rel=0.05)

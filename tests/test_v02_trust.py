"""v0.2 Layer 0: statistical compare (T4), schema stability (T5), selftest (T3)."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from edgelens.cli import app
from edgelens.compare import stats as cs
from edgelens.core import schema

FIXTURE = Path(__file__).parent / "fixtures" / "result_v0_1_0.json"
runner = CliRunner()


def _run(p99, mean=None, p50=None, n=300, env="e1", fps=None):
    mean = mean if mean is not None else p99 * 0.7
    return {"latency": {"p99": p99, "mean": mean, "p50": p50 or mean, "max": p99 * 1.2,
                        "n": n},
            "fps": fps or round(1000.0 / mean, 2), "identity": {"environment_id": env},
            "energy": {"available": False}, "deadline": None}


# ---------------------------------------------------------------- T4

def test_t_quantiles_match_tables():
    assert cs.t975(4) == pytest.approx(2.776)
    assert cs.t975(4.5) == pytest.approx((2.776 + 2.571) / 2)
    assert cs.t975(10_000) == pytest.approx(1.96, abs=0.01)
    assert cs.t80(1e6) == pytest.approx(0.842, abs=0.01)


def test_a_a_comparison_rarely_claims_a_change():
    """Same distribution on both sides: at most ~5% false verdicts (+ slack)."""
    rng = np.random.default_rng(1)
    false = 0
    trials = 400
    for _ in range(trials):
        a = [_run(10 + rng.normal(0, 0.4)) for _ in range(5)]
        b = [_run(10 + rng.normal(0, 0.4)) for _ in range(5)]
        if cs.compare_groups(a, b, primary="p99")["verdict"] != cs.NO_CHANGE:
            false += 1
    assert false / trials <= 0.09


def test_real_change_is_detected_with_direction():
    rng = np.random.default_rng(2)
    a = [_run(10 + rng.normal(0, 0.2)) for _ in range(5)]
    worse = [_run(11 + rng.normal(0, 0.2)) for _ in range(5)]
    better = [_run(6 + rng.normal(0, 0.2)) for _ in range(5)]
    r = cs.compare_groups(a, worse, primary="p99")
    assert r["verdict"] == cs.REGRESSED and r["regression"]
    lo, hi = r["metrics"]["p99"]["ci95_pct"]
    assert lo > 0 and hi > lo
    r = cs.compare_groups(a, better, primary="p99")
    assert r["verdict"] == cs.IMPROVED and not r["regression"]
    # throughput: higher is better, so the slower group is a regression there too
    assert cs.compare_groups(a, worse, primary="fps")["verdict"] == cs.REGRESSED


def test_tiny_regression_below_min_effect_does_not_fail():
    a = [_run(10.0 + d) for d in (0.0, 0.001, -0.001, 0.002, -0.002)]
    b = [_run(10.05 + d) for d in (0.0, 0.001, -0.001, 0.002, -0.002)]
    r = cs.compare_groups(a, b, primary="p99", min_effect_pct=2.0)
    assert r["verdict"] == cs.REGRESSED and not r["regression"]
    assert "below --min-effect" in r["notes"][0]


def test_no_change_reports_minimum_detectable_effect_and_needs_repeats():
    rng = np.random.default_rng(3)
    a = [_run(10 + rng.normal(0, 1.0)) for _ in range(3)]
    b = [_run(10 + rng.normal(0, 1.0)) for _ in range(3)]
    row = cs.compare_groups(a, b, primary="p99")["metrics"]["p99"]
    assert row["min_detectable_change_pct"] > 10          # 3 noisy runs can't see small changes
    r = cs.compare_groups(a[:2], b, primary="p99")
    assert r["verdict"] == cs.INSUFFICIENT
    assert "--repeat" in r["metrics"]["p99"]["reason"]


def test_primary_falls_back_to_mean_when_p99_unsupported():
    a = [_run(10) for _ in range(3)]
    for r in a:
        r["latency"]["p99"] = None
    assert cs.compare_groups(a, a)["primary_metric"] == "mean"


def test_block_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(4)
    x = list(rng.gamma(4, 1.0, 3000))
    ci = cs.within_run_ci(x, "p99")
    est = float(np.percentile(x, 99))
    assert ci[0] <= est <= ci[1]
    assert cs.within_run_ci(x[:50], "p99") is None        # p99 needs >= 100 samples


def test_cli_repeat_and_group_compare(tmp_path):
    script = tmp_path / "p.py"
    script.write_text(
        "import edgelens as el\n"
        "def build_pipeline():\n"
        "    p = el.Pipeline('x')\n"
        "    p.add_stage('work', lambda x=None: sum(range(3000)))\n"
        "    return p\n")
    for name in ("a", "b"):
        res = runner.invoke(app, ["benchmark", "--pipeline", str(script), "--iterations", "120",
                                  "--repeat", "3", "--save", str(tmp_path / f"{name}.json")])
        assert res.exit_code == 0, res.output
    assert sorted(p.name for p in tmp_path.glob("a.r*.json")) == \
        ["a.r01.json", "a.r02.json", "a.r03.json"]
    res = runner.invoke(app, ["compare", str(tmp_path / "a.r*.json"), str(tmp_path / "b.r*.json"),
                              "--min-effect", "1000"])
    assert res.exit_code == 0, res.output
    assert "runs" in res.output
    res = runner.invoke(app, ["compare", str(tmp_path / "a.r01.json"), str(tmp_path / "b.r01.json"),
                              str(tmp_path / "b.r02.json")])
    assert res.exit_code == 2 and "QUOTE" in res.output
    res = runner.invoke(app, ["compare", str(tmp_path / "a.r01.json"), str(tmp_path / "b.r01.json")])
    assert res.exit_code in (0, 1) and "--repeat 5" in res.output
    res = runner.invoke(app, ["compare", str(tmp_path / "zz*.json"), str(tmp_path / "b.r01.json")])
    assert res.exit_code == 1 and "No files match" in res.output


# ---------------------------------------------------------------- T5

def test_v010_fixture_is_a_real_v1_document():
    doc = json.loads(FIXTURE.read_text())
    assert doc["edgelens_version"] == "0.1.0"
    assert schema.problems(doc) == [] and schema.missing_v1_fields(doc) == []
    assert "clock" not in doc and "latency_series_ms" not in doc      # written before v0.2


def test_v010_file_works_in_every_v02_command(tmp_path):
    f = tmp_path / "old.json"
    f.write_text(FIXTURE.read_text())
    assert runner.invoke(app, ["diagnose", str(f)]).exit_code == 0
    res = runner.invoke(app, ["validate", str(f), "--p99-ms", "1000"])
    assert res.exit_code == 0, res.output
    assert runner.invoke(app, ["report", str(f), "--output", str(tmp_path / "r.html")]).exit_code == 0
    res = runner.invoke(app, ["compare", str(f), str(f)])
    assert res.exit_code == 0, res.output
    # an old file next to a new one
    import edgelens as el
    p = el.Pipeline("x")
    p.add_stage("work", lambda x=None: None)
    new = tmp_path / "new.json"
    new.write_text(json.dumps(p.run(iterations=120, warmup=1)))
    assert runner.invoke(app, ["compare", str(f), str(new)]).exit_code in (0, 1)


def test_new_results_are_still_v1_and_only_add_fields():
    import edgelens as el
    p = el.Pipeline("x")
    p.add_stage("work", lambda x=None: None)
    r = p.run(iterations=10, warmup=1)
    old = json.loads(FIXTURE.read_text())
    assert r["schema_version"] == schema.SCHEMA_VERSION == 1
    missing = [k for k in old if k not in r and k not in ("pack_metrics", "backlog")]
    assert missing == [], f"v0.2 dropped v1 fields: {missing}"
    assert schema.problems(r) == [] and schema.missing_v1_fields(r) == []


def test_newer_schema_is_refused_clearly(tmp_path):
    doc = json.loads(FIXTURE.read_text())
    doc["schema_version"] = 99
    f = tmp_path / "future.json"
    f.write_text(json.dumps(doc))
    res = runner.invoke(app, ["diagnose", str(f)])
    assert res.exit_code == 1 and "newer" in res.output


# ---------------------------------------------------------------- T3

def test_selftest_quick_runs_and_reports_every_section(tmp_path):
    out = tmp_path / "self.json"
    res = runner.invoke(app, ["selftest", "--quick", "--save", str(out)])
    assert res.exit_code in (0, 1), res.output
    doc = json.loads(out.read_text())
    names = {c["check"] for c in doc["checks"]}
    for needed in ("clock_read_cost", "harness_accuracy", "python_marker_cost",
                   "marker_delivery", "sampler_cpu", "telemetry_disturbance"):
        assert needed in names, names
    for c in doc["checks"]:
        assert c["status"] in ("PASS", "WARN", "FAIL", "SKIP")

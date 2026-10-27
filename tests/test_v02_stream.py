"""v0.2 Layer 4: windowed time-series pipelines (edgelens.stream).

A synthetic robot: 20 Hz, three inputs (one dead), a health truth that
degrades in 10 % steps; the 'model' is a Python function of the window, so
these tests need no onnxruntime."""

import json
import math

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from edgelens.cli import app
from edgelens.stream import data as sdata
from edgelens.stream import detect as det
from edgelens.stream import envelope as envmod
from edgelens.stream import parity as par
from edgelens.stream import perturb as pt
from edgelens.stream.features import (OutputFilter, StreamFeaturizer, batch_features,
                                      feature_names, filter_series, windows)
from edgelens.stream.model import load_model
from edgelens.stream.runner import replay, simulate
from edgelens.stream.spec import SpecError, load_spec, parse

RATE = 20
BLOCK = 5
SEQ = 8

MODEL_PY = '''
import numpy as np
def predict(w):
    # health falls as the std of column a grows (stat-major: [min, std, max, mean, last])
    std_a = float(np.mean(w[0, -3:, 3]))
    return 1.0 / (1.0 + max(0.0, std_a - 0.5))
'''


def make_csv(path, seconds=400, onset=150, seed=0, date_time=False):
    rng = np.random.default_rng(seed)
    n = seconds * RATE
    t = np.arange(n) / RATE
    health = np.where(t < onset, 100.0, np.maximum(10.0, 100.0 - 10.0 * ((t - onset) // 20 + 1)))
    amp = 1.0 + 4.0 * (100.0 - health) / 100.0
    a = rng.standard_normal(n) * amp
    b = np.sin(t) + 0.1 * rng.standard_normal(n)
    c = np.zeros(n)                                   # a dead channel
    with open(path, "w") as f:
        f.write("timestamp,a,b,c,m1,m2\n")
        for k in range(n):
            ts = (f"2025-08-26 15:{int(t[k] // 60) % 60:02d}:{t[k] % 60:012.9f}" if date_time
                  else f"{1000.0 + t[k]:.6f}")
            f.write(f"{ts},{a[k]:.6f},{b[k]:.6f},{c[k]},{health[k]},{min(100.0, health[k] + 10)}\n")
    return path


def make_spec(tmp_path, **over):
    (tmp_path / "model.py").write_text(MODEL_PY)
    doc = {"name": "toy", "data": {"time_column": "timestamp", "rate_hz": RATE,
                                   "columns": ["a", "b", "c"],
                                   "target": {"columns": ["m1", "m2"], "reduce": "min",
                                              "scale": 0.01}},
           "features": {"block": BLOCK, "stats": ["min", "std", "max", "mean", "last"],
                        "sequence": SEQ, "padding": "zeros"},
           "model": {"python": "model.py:predict"},
           "output": {"slew_per_step": 0.05, "mean_steps": 4},
           "decision": {"threshold": 0.5, "alarm_when": "below", "healthy_level": 1.0},
           "deployment": {"period_ms": BLOCK * 1000 / RATE}}
    for k, v in over.items():
        doc[k] = v
    p = tmp_path / "toy.stream.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


@pytest.fixture
def toy(tmp_path):
    csv = make_csv(tmp_path / "rec.csv")
    spec_path = make_spec(tmp_path)
    spec, _ = load_spec(spec_path)
    return {"csv": csv, "spec_path": spec_path, "spec": spec, "dir": tmp_path,
            "rec": sdata.load_csv(csv, spec)}


# ---------------------------------------------------------------- spec

def test_spec_rejects_typos_and_impossible_settings(tmp_path):
    base = {"data": {"rate_hz": 20, "columns": ["a"]}, "features": {"block": 5}}
    with pytest.raises(SpecError, match="Did you mean 'sequence'"):
        parse(dict(base, features={"block": 5, "sequnce": 3}))
    with pytest.raises(SpecError, match="block of at least 2"):
        parse(dict(base, features={"block": 1, "stats": ["std"]}))
    with pytest.raises(SpecError, match="one number per column"):
        parse(dict(base, scaler={"center": [0, 1], "scale": [1, 1]}))
    with pytest.raises(SpecError, match="wrong column"):
        parse(dict(base, scaler={"center": [0], "scale": [1], "columns": ["b"]}))
    with pytest.raises(SpecError, match="divide by zero"):
        parse(dict(base, scaler={"center": [0], "scale": [0]}))
    s = parse(base)
    assert s["features"]["step_s"] == 0.25 and s["data"]["max_gap_s"] == 1.0


def test_spec_describes_what_it_implies(toy):
    from edgelens.stream.spec import describe
    text = " ".join(describe(toy["spec"]))
    assert "one model step every 5 rows (0.25 s)" in text
    assert "[1, 8, 15]" in text and "2 s of history" in text
    assert "warm-up" in text and "1.75 s" in text


# ---------------------------------------------------------------- data

def test_date_time_stamps_with_nanoseconds(tmp_path):
    csv = make_csv(tmp_path / "dt.csv", seconds=30, date_time=True)
    spec, _ = load_spec(make_spec(tmp_path))
    rec = sdata.load_csv(csv, spec)
    dt = np.diff(rec.time_s)
    assert np.allclose(dt, 1.0 / RATE, atol=1e-6)
    assert rec.truth[0] == pytest.approx(1.0)        # min(m1, m2) * 0.01


def test_missing_column_is_named(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("timestamp,a\n0,1\n")
    spec, _ = load_spec(make_spec(tmp_path))
    with pytest.raises(ValueError, match="missing column"):
        sdata.load_csv(p, spec)


def test_audit_finds_gaps_rate_and_dead_columns(toy):
    rec = toy["rec"].copy()
    keep = (rec.time_s < 1100) | (rec.time_s > 1130)       # a 30 s stall
    rec.time_s, rec.x, rec.truth = rec.time_s[keep], rec.x[keep], rec.truth[keep]
    a = sdata.audit(rec, toy["spec"])
    assert len(a["gaps"]) == 1 and a["gaps"][0]["length_s"] == pytest.approx(30.1, abs=0.01)
    assert a["constant_columns"] == ["c"] and not a["rate_mismatch"]
    rec.time_s = rec.time_s[0] + (rec.time_s - rec.time_s[0]) * 2     # 10 Hz
    assert sdata.audit(rec, toy["spec"])["rate_mismatch"]


def test_fit_scaler_is_sklearn_robust(toy):
    sc = sdata.fit_scaler([toy["rec"]], "robust")
    x = toy["rec"].x
    assert sc["center"][0] == pytest.approx(np.median(x[:, 0]))
    q1, q3 = np.percentile(x[:, 0], [25, 75])
    assert sc["scale"][0] == pytest.approx(q3 - q1)
    assert sc["scale"][2] == 1.0                     # constant column, like sklearn


# ---------------------------------------------------------------- features

def _reference_features(rec, spec):
    """Straight loops: what pandas rolling(W) sampled every W rows computes."""
    W = spec["features"]["block"]
    out = []
    for k in range(len(rec) // W):
        b = rec.x[k * W:(k + 1) * W]
        out.append(np.concatenate([b.min(0), b.std(0, ddof=1), b.max(0), b.mean(0), b[-1]]))
    return np.array(out)


def test_batch_features_match_the_training_definition(toy):
    F, info = batch_features(toy["rec"], toy["spec"])
    assert np.allclose(F, _reference_features(toy["rec"], toy["spec"]))
    assert F.shape[1] == 15 and len(info["t_end"]) == len(F)
    assert feature_names(toy["spec"])[3] == "a_std"


def test_column_major_layout(toy, tmp_path):
    spec, _ = load_spec(make_spec(tmp_path, features={"block": BLOCK, "stats": ["min", "max"],
                                                      "sequence": SEQ, "layout": "column-major"}))
    F, _ = batch_features(toy["rec"], spec)
    b = toy["rec"].x[:BLOCK]
    assert np.allclose(F[0], [b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max(),
                              b[:, 2].min(), b[:, 2].max()])


@pytest.mark.parametrize("padding", ["zeros", "none"])
def test_streaming_matches_batch(toy, tmp_path, padding):
    spec, _ = load_spec(make_spec(tmp_path, features={
        "block": BLOCK, "stats": ["min", "std", "max", "mean", "last"], "sequence": SEQ,
        "padding": padding}))
    F, _ = batch_features(toy["rec"], spec)
    ref = windows(F, spec)
    for mode in ("incremental", "recompute"):
        fz = StreamFeaturizer(spec, mode=mode)
        got = []
        for k in range(40):
            w = fz.push(toy["rec"].x[k * BLOCK:(k + 1) * BLOCK])
            if w is not None:
                got.append(w[0])
        if padding == "none":
            assert len(got) == 40 - SEQ + 1
        assert np.allclose(np.array(got), ref[:len(got)], atol=1e-5), mode
    assert ref[0, :SEQ - 1].sum() == 0 if padding == "zeros" else True


def test_output_filter_is_slew_then_rolling_mean():
    spec = {"output": {"slew_per_step": 0.1, "mean_steps": 3}}
    out = filter_series([1.0, 0.0, 0.0, 0.0, 1.0], spec)
    slewed = [1.0, 0.9, 0.8, 0.7, 0.8]
    expect = [np.mean(slewed[max(0, i - 2):i + 1]) for i in range(5)]
    assert np.allclose(out, expect)
    f = OutputFilter({"output": {"slew_per_step": None, "mean_steps": 1}})
    assert f(0.3) == 0.3 and f(0.9) == 0.9


# ---------------------------------------------------------------- detect

DEC = {"threshold": 0.5, "alarm_when": "below", "truth_threshold": 0.5, "healthy_level": 1.0}


def test_detection_metrics_separate_false_early_and_late():
    t = np.arange(20.0)
    truth = np.array([1.0] * 8 + [0.8] * 4 + [0.4] * 8)        # onset 8, event 12
    out = np.array([1, 0.4, 1, 1, 1, 1, 1, 1, 0.6, 0.45, 0.6, 0.6, 0.7, 0.7, 0.3, 0.3, 0.3,
                    0.3, 0.3, 0.3])
    s = det.score(t, truth, out, DEC, 1.0)
    assert s["false_alarms"] == 1 and s["false_alarm_at_s"] == [1.0]   # healthy, alarmed
    assert s["early_alarm_s"] == 1.0                                    # degraded, before event
    assert s["alarm_after_onset_s"] == 9.0
    assert s["time_to_detect_s"] == -3.0 and not s["missed"]
    assert s["false_alarms_per_h"] == pytest.approx(1 / (8 / 3600))
    s2 = det.score(t, truth, np.ones(20), DEC, 1.0)
    assert s2["missed"] and s2["time_to_detect_s"] is None


def test_alarm_above_threshold_direction():
    dec = {"threshold": 0.5, "alarm_when": "above", "truth_threshold": 0.5, "healthy_level": 0}
    t = np.arange(6.0)
    s = det.score(t, np.array([0, 0, 0.6, 0.6, 0.6, 0.6]), np.array([0, 0, 0, 0.7, 0.7, 0.7]),
                  dec, 1.0)
    assert s["time_to_detect_s"] == 1.0 and s["false_alarms"] == 0


def test_latency_budget_names_the_built_in_delays(toy):
    b = det.latency_budget(toy["spec"], start_level=1.0)
    assert b["step_s"] == 0.25 and b["warmup_s"] == 1.75
    assert b["slew_floor_s"] == pytest.approx(10 * 0.25)     # 0.5 / 0.05 steps
    assert b["mean_delay_s"] == pytest.approx(1.5 * 0.25)


# ---------------------------------------------------------------- replay

def test_replay_is_an_ordinary_result_with_decisions(toy):
    m = load_model(toy["spec"])
    res = replay(toy["rec"], toy["spec"], m, telemetry=False)
    assert [s["name"] for s in res["pipeline"]["stages"]] == [
        "ingest", "features", "inference", "postprocess", "decision"]
    assert res["iterations"] == len(toy["rec"]) // BLOCK
    assert res["deadline"]["deadline_ms"] == pytest.approx(250.0)   # deployed period
    d = res["decision"]
    assert d["output"]["event_s"] is not None and not d["output"]["missed"]
    assert d["guard"]["warming_up"] == SEQ - 1
    # the harness path and the batch path decide the same
    t, truth, raw, out, states = simulate(toy["rec"], toy["spec"], m)
    assert np.allclose(np.array(d["series"]["raw"], float), raw, atol=1e-4)
    assert res["stream"]["audit"]["constant_columns"] == ["c"]


def test_diagnosis_explains_decision_time_and_cadence(toy, tmp_path):
    from edgelens.diagnose.engine import diagnose
    spec, _ = load_spec(make_spec(tmp_path, deployment={"period_ms": 1000}))
    res = replay(toy["rec"], spec, load_model(spec), telemetry=False)
    types = [f["type"] for f in diagnose(res)["all_findings"]]
    assert "DECISION_LATENCY" in types and "CADENCE_MISMATCH" in types
    assert "WARMUP_PADDING" in types
    cad = next(f for f in diagnose(res)["all_findings"] if f["type"] == "CADENCE_MISMATCH")
    assert "250 ms" in cad["detail"] and "1000 ms" in cad["detail"]


def test_paced_replay_releases_on_the_accelerated_step(toy):
    res = replay(toy["rec"], toy["spec"], load_model(toy["spec"]), speed=50.0,
                 telemetry=False, max_steps=20)
    assert res["requirements"]["paced"] and res["requirements"]["period_ms"] == pytest.approx(5.0)
    assert res["iterations"] == 20


# ---------------------------------------------------------------- envelope

def test_envelope_flags_inputs_never_seen(toy):
    env = envmod.fit([toy["rec"]], toy["spec"], quantile=0.0, margin=0.05)
    assert "c_std" in env["constant_features"]
    g = envmod.Guard(env, toy["spec"])
    F, _ = batch_features(toy["rec"], toy["spec"])
    assert g.state(F[20])[0] == "valid"
    v = F[20].copy()
    v[feature_names(toy["spec"]).index("c_last")] = 0.5          # the dead channel moves
    st, idx = g.state(v)
    assert st == "degraded" and feature_names(toy["spec"])[idx[0]] == "c_last"
    assert g.state(F[20] + 100)[0] == "invalid"
    assert g.state(F[2], warming_up=True)[0] == "warming_up"
    rep = envmod.check([toy["rec"]], toy["spec"], env)
    assert rep["recordings"][0]["in_envelope_pct"] == 100.0


def test_envelope_must_match_the_spec(toy, tmp_path):
    env = envmod.fit([toy["rec"]], toy["spec"])
    p = tmp_path / "env.json"
    p.write_text(json.dumps(env))
    spec2, _ = load_spec(make_spec(tmp_path, features={"block": BLOCK, "stats": ["mean"],
                                                       "sequence": SEQ}))
    with pytest.raises(ValueError, match="refit"):
        envmod.load(p, spec2)


# ---------------------------------------------------------------- perturb

def test_fault_file_is_validated():
    with pytest.raises(pt.FaultError, match="kind"):
        pt.parse_faults({"faults": [{"kind": "melt"}]})
    with pytest.raises(pt.FaultError, match="unknown key"):
        pt.parse_faults({"faults": [{"kind": "noise", "sevrity": [1]}]})
    f = pt.parse_faults({"faults": [{"kind": "noise", "severity": [2, 1]},
                                    {"kind": "rate", "severity": [0.5]}]})
    assert f["faults"][0]["severity"] == [0.0, 1.0, 2.0]          # baseline added
    assert f["faults"][1]["severity"] == [0.5, 1.0]


def test_faults_change_the_data_as_documented(toy):
    rng = np.random.default_rng(0)
    rec, spec = toy["rec"], toy["spec"]
    def fault(kind, **kw):
        return dict({"name": kind, "kind": kind, "columns": ["a"], "start_s": 0.0,
                     "duration_s": None}, **kw)
    b = pt.apply(rec, fault("bias"), 2.0, spec, rng)
    assert np.allclose(b.x[:, 0] - rec.x[:, 0], 2.0 * 1.0)        # no scaler: scale 1
    assert np.array_equal(b.x[:, 1], rec.x[:, 1])
    s = pt.apply(rec, fault("stuck", start_s=10.0), 1.0, spec, rng)
    assert np.all(s.x[10 * RATE:, 0] == rec.x[10 * RATE, 0])
    g = pt.apply(rec, fault("gap", start_s=10.0), 5.0, spec, rng)
    assert len(g) == len(rec) - 5 * RATE and len(sdata.audit(g, spec)["gaps"]) == 1
    r = pt.apply(rec, fault("rate"), 0.5, spec, rng)
    assert len(r) == pytest.approx(len(rec) / 2, abs=1)
    assert pt.apply(rec, fault("noise"), 0.0, spec, rng).x.tolist() == rec.x.tolist()


def test_sweep_finds_where_the_decision_breaks(toy):
    m = load_model(toy["spec"])
    faults = pt.parse_faults({"faults": [{"name": "a-noise", "kind": "noise",
                                          "columns": ["a"], "severity": [0.5, 3, 6]}]})
    doc = pt.sweep([toy["rec"]], toy["spec"], m, faults)
    c = doc["curves"][0]
    assert [p["severity"] for p in c["points"]] == [0.0, 0.5, 3.0, 6.0]
    assert c["points"][0]["false_alarms"] == 0
    assert c["breaks_at"] is not None and "false alarm" in c["breaks_at"]["why"]
    assert "a-noise" in pt.markdown(doc)


def test_breaking_point_compares_rate_against_the_real_rate():
    pts = [{"severity": 0.5, "missed": 1, "false_alarms": 0, "time_to_detect_max_s": None},
           {"severity": 1.0, "missed": 0, "false_alarms": 0, "time_to_detect_max_s": 10.0},
           {"severity": 2.0, "missed": 0, "false_alarms": 0, "time_to_detect_max_s": 11.0}]
    assert pt._breaking_point(pts, 5.0, neutral=1.0) == {"severity": 0.5,
                                                          "why": ["fault missed"]}


# ---------------------------------------------------------------- parity

def test_parity_names_the_classic_bugs(toy):
    spec = toy["spec"]
    F, _ = batch_features(toy["rec"], spec)
    ours = windows(F, spec).astype(np.float64)
    assert par.compare_tensor(ours, ours.copy(), spec)["verdict"] == "match"
    # ddof=0 on the training side
    ref = ours.copy()
    std = [3 + c for c in range(3)]
    ref[:, :, std] *= math.sqrt((BLOCK - 1) / BLOCK)
    rep = par.compare_tensor(ours, ref, spec)
    assert rep["verdict"] == "MISMATCH" and any("ddof=0" in s for s in rep["signatures"])
    # one column scaled differently: every statistic of it differs
    ref = ours.copy()
    ref[:, :, [0, 3, 6, 9, 12]] = ref[:, :, [0, 3, 6, 9, 12]] * 2 + 1
    rep = par.compare_tensor(ours, ref, spec)
    assert any("every statistic of 'a'" in s for s in rep["signatures"])
    # shifted by one step
    rep = par.compare_tensor(ours[1:], ours[:-1], spec)
    assert any("shifted by" in s for s in rep["signatures"])
    st = par.self_check(toy["rec"], spec, steps=30)
    assert st["incremental"]["max_abs_error"] < 1e-5 and st["recompute"]["max_abs_error"] < 1e-5


def test_output_parity_counts_flipped_decisions(toy, tmp_path):
    raw = np.array([0.9, 0.6, 0.4, 0.3])
    p = tmp_path / "y.bin"
    np.array([0.9, 0.4, 0.4, 0.3], np.float32).tofile(p)
    rep = par.compare_outputs(raw, p, toy["spec"])
    assert rep["decisions_flipped"] == 1 and rep["max_abs_error"] == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------- CLI

def test_cli_end_to_end_replay_detect_validate(toy, tmp_path):
    r = CliRunner()
    sp = str(toy["spec_path"])
    out = r.invoke(app, ["stream", "check", sp, str(toy["csv"])])
    assert out.exit_code == 0, out.output
    assert "never change" in out.output
    runs = tmp_path / "runs"
    out = r.invoke(app, ["replay", sp, str(toy["csv"]), str(make_csv(tmp_path / "r2.csv",
                                                                       seed=1)),
                         "--save-dir", str(runs), "--no-telemetry"])
    assert out.exit_code == 0, out.output
    assert "Where decision time goes" in out.output
    files = sorted(str(p) for p in runs.glob("*.json"))
    assert len(files) == 2
    det_path = tmp_path / "detect.json"
    out = r.invoke(app, ["detect", *files, "--thresholds", "0.9,0.5", "--save", str(det_path),
                         "--markdown", str(tmp_path / "d.md")])
    assert out.exit_code == 0, out.output
    assert "Decision-latency curve" in out.output
    doc = json.loads(det_path.read_text())
    assert doc["decision"]["aggregate"]["output"]["events"] == 2
    # what-if post-processing without re-running the model
    out = r.invoke(app, ["detect", *files, "--mean-steps", "1", "--slew-per-step", "0"])
    assert out.exit_code == 0 and "what-if" in out.output
    ok = r.invoke(app, ["validate", str(det_path), "--max-time-to-detect-s", "1000",
                        "--max-missed-events", "0"])
    assert ok.exit_code == 0, ok.output
    bad = r.invoke(app, ["validate", str(det_path), "--max-time-to-detect-s", "-1000"])
    assert bad.exit_code == 1 and "DECISION_LATENCY" in bad.output
    assert "Limiting stage" not in bad.output


def test_contract_envelope_assumption(toy, tmp_path):
    r = CliRunner()
    run = tmp_path / "run.json"
    assert r.invoke(app, ["replay", str(toy["spec_path"]), str(toy["csv"]), "--save", str(run),
                          "--no-telemetry"]).exit_code == 0
    c = tmp_path / "c.yaml"
    c.write_text("contract: toy\nassumes:\n  min_in_envelope_pct: 99\n"
                 "guarantees:\n  max_missed_events: 0\n")
    out = r.invoke(app, ["validate", str(run), "--contract", str(c)])
    assert out.exit_code == 2 and "NOT VERIFIED" in out.output       # no envelope in the spec
    env = tmp_path / "envelope.json"
    assert r.invoke(app, ["envelope", "fit", str(toy["spec_path"]), str(toy["csv"]),
                          "--out", str(env)]).exit_code == 0
    spec2 = make_spec(tmp_path, envelope={"file": "envelope.json"})
    assert r.invoke(app, ["replay", str(spec2), str(toy["csv"]), "--save", str(run),
                          "--no-telemetry"]).exit_code == 0
    out = r.invoke(app, ["validate", str(run), "--contract", str(c)])
    assert out.exit_code == 0, out.output
    out = r.invoke(app, ["envelope", "check", str(spec2), str(toy["csv"])])
    assert out.exit_code == 0 and "Training envelope" in out.output


def test_cli_parity_exit_codes(toy, tmp_path):
    r = CliRunner()
    F, _ = batch_features(toy["rec"], toy["spec"])
    w = windows(F, toy["spec"])
    good, bad = tmp_path / "x.bin", tmp_path / "x0.bin"
    w.astype(np.float32).tofile(good)
    b = w.copy()
    b[:, :, 3:6] *= math.sqrt((BLOCK - 1) / BLOCK)
    b.astype(np.float32).tofile(bad)
    assert r.invoke(app, ["parity", str(toy["spec_path"]), str(toy["csv"]), "--reference",
                          str(good)]).exit_code == 0
    out = r.invoke(app, ["parity", str(toy["spec_path"]), str(toy["csv"]), "--reference",
                         str(bad)])
    assert out.exit_code == 1 and "ddof=0" in out.output


def test_cli_perturb_template_and_run(toy, tmp_path):
    r = CliRunner()
    f = tmp_path / "faults.yaml"
    assert r.invoke(app, ["perturb", str(toy["spec_path"]), "--init", str(f)]).exit_code == 0
    assert "kind: noise" in f.read_text()
    small = tmp_path / "small.yaml"
    small.write_text("faults:\n  - {name: a-bias, kind: bias, columns: [a], severity: [1]}\n")
    out = r.invoke(app, ["perturb", str(toy["spec_path"]), str(toy["csv"]), "--faults",
                         str(small), "--save", str(tmp_path / "rob.json")])
    assert out.exit_code == 0, out.output
    assert json.loads((tmp_path / "rob.json").read_text())["kind"] == "edgelens-robustness"

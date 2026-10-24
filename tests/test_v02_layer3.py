"""v0.2 Layer 3: assumption contracts (E1), Perfetto timeline (E2), CI gate (E3)."""

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edgelens import contract as ct
from edgelens import timeline as tl
from edgelens.cli import app
from edgelens.core.harness import run_harness
from edgelens.core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _pipe(ms=0.3):
    p = Pipeline("anomaly-like")

    def pre(x=None):
        t = time.perf_counter() + ms / 2000.0
        while time.perf_counter() < t:
            pass

    def infer(x=None):
        t = time.perf_counter() + ms / 2000.0
        while time.perf_counter() < t:
            pass
    p.add_stage("preprocess", pre)
    p.add_stage("inference", infer)
    return p


@pytest.fixture(scope="module")
def jetson_run():
    """A real paced run, with the environment of a Jetson AGX Orin set on top
    (this machine is not one): everything the contract checks is present."""
    r = run_harness(_pipe(), iterations=150, warmup=5, period_ms=5.0, pace=True,
                    deadline_ms=5.0)
    # a shared CI machine can stall for 100 ms: keep the guarantees about the
    # contract logic, not about this machine's scheduler
    r["deadline"].update(misses=0, miss_ratio=0.0, met=True)
    r["latency"]["p99"] = min(r["latency"]["p99"], 4.0)
    r["environment"].update(board="NVIDIA Jetson AGX Orin Developer Kit", is_jetson=True,
                            power_mode={"id": 0, "name": "MAXN"}, cpu_clocks_locked=True,
                            gpu_clocks_locked=True)
    r["telemetry"].update(start_temp_c=41.0, max_temp_c=52.0)
    r["telemetry"]["background"] = {"cpu_pct": 1.5, "cores": 0.18}
    r["pipeline"]["config"]["provider"] = "TensorrtExecutionProvider"
    r["identity"]["model_sha256"] = "1a2b3c4d5e6f7a8b"
    return r


CONTRACT = {
    "contract": "anomaly-detector", "version": "1.0", "requirements_doc": "SRS-2.1",
    "min_duration": "0.5s", "min_iterations": 100,
    "assumes": {
        "board": {"value": "AGX Orin", "id": "SRS-A1"},
        "power_mode": "MAXN",
        "clocks_pinned": True,
        "input_period_ms": {"value": 5, "tolerance_pct": 5},
        "max_start_temp_c": 50,
        "max_temp_c": 90,
        "max_background_cpu_pct": 10,
        "provider": "TensorrtExecutionProvider",
        "model_sha256": "1a2b3c4d",
    },
    "guarantees": {
        "p99_ms": {"limit": 50, "id": "SRS-R07"},
        "max_miss_ratio": {"limit": 0.5, "deadline_ms": 5, "id": "SRS-R08"},
        "max_temp_c": {"limit": 85, "id": "SRS-R14"},
    },
}


def _eval(result, doc=None):
    return ct.evaluate(result, ct.parse(copy.deepcopy(doc or CONTRACT)), "sha")


def test_all_assumptions_hold_and_the_verdict_is_the_guarantees(jetson_run):
    v = _eval(jetson_run)
    assert [a["status"] for a in v["assumptions"]] == [ct.HOLDS] * 9, v["assumptions"]
    assert v["verdict"] == ct.PASS and v["guarantees_verdict"] == ct.PASS
    ids = {r["requirement"]: r["id"] for r in v["checks"]}
    assert ids == {"p99_ms": "SRS-R07", "max_miss_ratio": "SRS-R08", "max_temp_c": "SRS-R14"}
    assert v["assumptions"][0]["id"] == "SRS-A1"


# E1 "done when": each assumption, deliberately violated, gives INCONCLUSIVE.
VIOLATIONS = {
    # board AND module: on a real Jetson the result also carries the module
    # read from the device tree ("Jetson AGX Orin 32GB"), which the board
    # assumption matches too — a fake Nano must not keep the Orin's module
    "board": lambda r: r["environment"].update(board="NVIDIA Jetson Nano Developer Kit",
                                               module="Jetson Nano (devkit)"),
    "power_mode": lambda r: r["environment"].update(power_mode={"id": 1, "name": "MODE_15W"}),
    "clocks_pinned": lambda r: r["environment"].update(gpu_clocks_locked=False),
    "input_period_ms": lambda r: r["requirements"].update(period_ms=10.0),
    "max_start_temp_c": lambda r: r["telemetry"].update(start_temp_c=63.0),
    "max_temp_c": lambda r: r["telemetry"].update(max_temp_c=95.0),
    "max_background_cpu_pct": lambda r: r["telemetry"]["background"].update(cpu_pct=37.0),
    "provider": lambda r: r["pipeline"]["config"].update(provider="CUDAExecutionProvider"),
    "model_sha256": lambda r: r["identity"].update(model_sha256="ffff0000ffff0000"),
}


@pytest.mark.parametrize("key", sorted(VIOLATIONS))
def test_each_violated_assumption_gives_inconclusive(jetson_run, key):
    r = copy.deepcopy(jetson_run)
    VIOLATIONS[key](r)
    doc = copy.deepcopy(CONTRACT)
    if key == "max_temp_c":          # keep the guarantee out of the way: 95 C would FAIL it
        doc["guarantees"].pop("max_temp_c")
    v = _eval(r, doc)
    row = next(a for a in v["assumptions"] if a["assumption"] == key)
    assert row["status"] == ct.VIOLATED, row
    others = [a["status"] for a in v["assumptions"] if a["assumption"] != key]
    assert set(others) == {ct.HOLDS}
    assert v["guarantees_verdict"] == ct.PASS
    assert v["verdict"] == ct.INCONCLUSIVE
    assert any(key in n for n in v["notes"])


@pytest.mark.parametrize("key,remove", [
    # power_mode_status too: an AGX Orin whose nvpmodel says "power mode is not
    # set" records that, and the contract then (rightly) says VIOLATED
    ("power_mode", lambda r: r["environment"].update(power_mode=None, power_mode_status=None)),
    ("clocks_pinned", lambda r: r["environment"].update(cpu_clocks_locked=None)),
    ("max_background_cpu_pct", lambda r: r["telemetry"].pop("background")),
    ("max_start_temp_c", lambda r: (r["telemetry"].pop("start_temp_c"),
                                    r.update(telemetry_series=[]))),
    ("input_period_ms", lambda r: r["requirements"].update(period_ms=None)),
])
def test_unverifiable_assumption_is_inconclusive_too(jetson_run, key, remove):
    r = copy.deepcopy(jetson_run)
    remove(r)
    v = _eval(r)
    row = next(a for a in v["assumptions"] if a["assumption"] == key)
    assert row["status"] == ct.NOT_VERIFIED
    assert v["verdict"] == ct.INCONCLUSIVE


def test_unpaced_run_violates_the_input_period(jetson_run):
    r = copy.deepcopy(jetson_run)
    r["requirements"]["paced"] = False
    v = _eval(r)
    row = next(a for a in v["assumptions"] if a["assumption"] == "input_period_ms")
    assert row["status"] == ct.VIOLATED and "back to back" in row["reason"]


def test_fail_under_the_assumptions_is_fail_but_outside_is_inconclusive(jetson_run):
    doc = copy.deepcopy(CONTRACT)
    doc["guarantees"]["p99_ms"] = {"limit": 0.001, "id": "SRS-R07"}
    v = _eval(jetson_run, doc)
    assert v["verdict"] == ct.FAIL and v["diagnosis"]["primary"]["type"]
    r = copy.deepcopy(jetson_run)
    r["environment"]["power_mode"] = {"id": 2, "name": "MODE_30W"}
    v = _eval(r, doc)
    assert v["guarantees_verdict"] == ct.FAIL and v["verdict"] == ct.INCONCLUSIVE
    assert "diagnosis" not in v
    assert any("outside the contract's assumptions" in n for n in v["notes"])


def test_too_little_evidence_is_inconclusive(jetson_run):
    doc = copy.deepcopy(CONTRACT)
    doc["min_duration"] = "30m"
    v = _eval(jetson_run, doc)
    assert v["verdict"] == ct.INCONCLUSIVE
    assert v["evidence_conditions"][0]["status"] == ct.NOT_MET
    doc = copy.deepcopy(CONTRACT)
    doc["min_iterations"] = 10_000
    assert _eval(jetson_run, doc)["verdict"] == ct.INCONCLUSIVE


def test_misses_against_another_deadline_are_not_measured(jetson_run):
    doc = copy.deepcopy(CONTRACT)
    doc["guarantees"]["max_miss_ratio"]["deadline_ms"] = 3000
    v = _eval(jetson_run, doc)
    row = next(r for r in v["checks"] if r["requirement"] == "max_miss_ratio")
    assert row["status"] == ct.NOT_MEASURED and "3000" in row["reason"]
    assert v["verdict"] == ct.INCONCLUSIVE


def test_demo_results_never_pass(jetson_run):
    r = copy.deepcopy(jetson_run)
    r["mode"] = "demo"
    assert _eval(r)["verdict"] == ct.INCONCLUSIVE


@pytest.mark.parametrize("bad,msg", [
    ({"asumes": {}}, "Did you mean 'assumes'"),
    ({"assumes": {"clock_pinned": True}}, "Did you mean 'clocks_pinned'"),
    ({"guarantees": {"p99": 3}}, "Did you mean 'p99_ms'"),
    ({"guarantees": {"p99_ms": {"limit": 3, "idd": "x"}}}, "Unknown key 'idd'"),
    ({"guarantees": {"p99_ms": "fast"}}, "expected a number"),
    ({"guarantees": {}}, "at least one"),
    ({"assumes": {"clocks_pinned": "yes"}}, "true or false"),
    ({"min_duration": "half an hour"}, "min_duration"),
    ({"assumes": {"input_period_ms": {"value": 1000, "limit": 1000}}}, "exactly one"),
])
def test_contract_typos_and_mistakes_are_errors(bad, msg):
    doc = {"contract": "x", "guarantees": {"p99_ms": 3}}
    doc.update(bad)
    with pytest.raises(ct.ContractError, match=msg):
        ct.parse(doc)


def test_yaml_and_json_contracts_load_with_a_hash(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(ct.TEMPLATE)
    c, sha = ct.load(y)
    assert c["contract"] == "my-pipeline" and len(sha) == 64
    assert c["assumes"]["input_period_ms"]["tolerance_pct"] == 5
    j = tmp_path / "c.json"
    j.write_text(json.dumps(CONTRACT))
    c2, _ = ct.load(j)
    assert c2["guarantees"]["max_miss_ratio"]["deadline_ms"] == 5


def test_markers_period_is_checked_against_the_observed_rate():
    base = {"mode": "markers", "markers": {"lost": 0}, "requirements": {"period_ms": 1000},
            "throughput_basis": "observed rate over the run",
            "latency": {"p50": 20.0}, "deadline": {"misses": 0}}
    spec = {"value": 1000.0, "tolerance_pct": 5.0}
    ok = ct.check_assumption(dict(base, fps=1.0), "input_period_ms", spec)
    assert ok["status"] == ct.HOLDS and ok["basis"] == "measured"
    fast = ct.check_assumption(dict(base, fps=2.0), "input_period_ms", spec)
    assert fast["status"] == ct.VIOLATED and "faster" in fast["reason"]
    slow = ct.check_assumption(dict(base, fps=0.5), "input_period_ms", spec)
    assert slow["status"] == ct.VIOLATED and "slower" in slow["reason"]
    # the app itself fell behind: timing can't tell, the configured period is used
    behind = ct.check_assumption(dict(base, fps=0.5, latency={"p50": 1900.0}),
                                 "input_period_ms", spec)
    assert behind["status"] == ct.HOLDS and behind["basis"] == "declared"


def test_background_load_is_measured_and_detects_a_busy_process():
    quiet = run_harness(_pipe(), iterations=60, warmup=2, period_ms=20.0, pace=True)
    bg = quiet["telemetry"]["background"]
    assert bg["cpu_pct"] >= 0 and os.getpid() in bg["workload_pids"]
    hog = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        time.sleep(0.3)
        busy = run_harness(_pipe(), iterations=60, warmup=2, period_ms=20.0, pace=True)
    finally:
        hog.kill()
        hog.wait()
    ncpu = os.cpu_count() or 1
    # one spinning process = one core of background load
    assert busy["telemetry"]["background"]["cores"] > 0.6
    assert busy["telemetry"]["background"]["cpu_pct"] > 60.0 / ncpu
    v = ct.evaluate(busy, ct.parse({"contract": "x",
                                    "assumes": {"max_background_cpu_pct": 50.0 / ncpu},
                                    "guarantees": {"p50_ms": 1000}}))
    assert v["verdict"] == ct.INCONCLUSIVE


def test_evidence_markdown_has_every_requirement_id(jetson_run):
    md = ct.evidence_markdown(_eval(jetson_run), "run.json")
    for rid in ("SRS-A1", "SRS-R07", "SRS-R08", "SRS-R14"):
        assert rid in md
    assert md.startswith("# ✅ PASS — anomaly-detector v1.0")
    assert "| Requirements | SRS-2.1 |" in md


# ---------------------------------------------------------------- CLI

def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj, default=str))
    return p


def test_cli_validate_contract_exit_codes_and_evidence(tmp_path, jetson_run):
    run = _write(tmp_path, "run.json", jetson_run)
    c = _write(tmp_path, "c.json", CONTRACT)
    ev = tmp_path / "evidence.md"
    res = runner.invoke(app, ["validate", str(run), "--contract", str(c), "--evidence", str(ev)])
    assert res.exit_code == 0, res.output
    assert "SRS-R07" in ev.read_text()
    saved = json.loads((tmp_path / "run.validation.json").read_text())
    assert saved["verdict"] == "PASS" and saved["contract"]["name"] == "anomaly-detector"

    bad = copy.deepcopy(jetson_run)
    bad["environment"]["cpu_clocks_locked"] = False
    run2 = _write(tmp_path, "run2.json", bad)
    res = runner.invoke(app, ["validate", str(run2), "--contract", str(c)])
    assert res.exit_code == 2

    res = runner.invoke(app, ["validate", str(run), "--contract", str(c), "--p99-ms", "3"])
    assert res.exit_code == 2 and "not both" in res.output

    typo = _write(tmp_path, "typo.json", {"contract": "x", "guarantee": {"p99_ms": 3}})
    res = runner.invoke(app, ["validate", str(run), "--contract", str(typo)])
    assert res.exit_code == 2 and "Did you mean 'guarantees'" in " ".join(res.output.split())


def test_cli_contract_init_check(tmp_path):
    p = tmp_path / "c.yaml"
    assert runner.invoke(app, ["contract", "init", str(p)]).exit_code == 0
    assert runner.invoke(app, ["contract", "init", str(p)]).exit_code == 1    # no overwrite
    res = runner.invoke(app, ["contract", "check", str(p)])
    assert res.exit_code == 0 and "601 inputs" in res.output.replace("\n", " ")


def test_cli_contract_run_measures_what_the_contract_needs(tmp_path):
    pipe = tmp_path / "p.py"
    pipe.write_text("import time\nfrom edgelens import Pipeline\n"
                    "def build_pipeline():\n    p = Pipeline('t')\n"
                    "    p.add_stage('infer', lambda x=None: time.sleep(0.0005))\n    return p\n")
    c = _write(tmp_path, "c.json", {
        "contract": "laptop", "min_duration": "1s",
        "assumes": {"input_period_ms": 10, "max_background_cpu_pct": 100},
        "guarantees": {"p99_ms": {"limit": 9, "id": "R1"},
                       "max_miss_ratio": {"limit": 0.05, "id": "R2"}}})
    save = tmp_path / "r.json"
    trace = tmp_path / "t.json"
    res = runner.invoke(app, ["contract", "run", str(c), "--pipeline", str(pipe),
                              "--save", str(save), "--perfetto", str(trace)])
    assert res.exit_code in (0, 1), res.output      # PASS (or FAIL on a very slow machine)
    r = json.loads(save.read_text())
    assert r["requirements"]["paced"] and r["requirements"]["period_ms"] == 10
    assert r["iterations"] >= 101 and r["deadline"]["deadline_ms"] == 10
    assert (tmp_path / "r.evidence.md").exists()
    assert json.loads(trace.read_text())["traceEvents"]


def test_cli_contract_run_stops_before_measuring_on_a_violated_assumption(tmp_path):
    c = _write(tmp_path, "c.json", {"contract": "x", "assumes": {"max_background_cpu_pct": -1},
                                    "guarantees": {"p50_ms": 10}})
    res = runner.invoke(app, ["contract", "run", str(c), "--pipeline", "does-not-matter.py",
                              "--save", str(tmp_path / "r.json")])
    assert res.exit_code == 2 and "VIOLATED BEFORE THE RUN" in res.output
    assert not (tmp_path / "r.json").exists()


# ---------------------------------------------------------------- E2 Perfetto

def _jetson_series(t0):
    return [{"t_s": k * 0.1, "t_mono_ns": t0 + k * 100_000_000, "cpu_percent": 30 + k,
             "gpu_percent": 60.0, "gpu_freq_mhz": 1300.5, "mem_percent": 40.0,
             "emc_percent": 12.0, "max_temp_c": 50 + k * 0.1, "power_w": 9.5,
             "rails_w": {"VDD_GPU_SOC": 4.1, "VDD_CPU_CV": 2.2}} for k in range(10)]


def test_perfetto_trace_has_every_track(jetson_run):
    r = copy.deepcopy(jetson_run)
    t0 = r["clock"]["trace_origin_mono_ns"]
    r["telemetry_series"] = _jetson_series(t0)
    r["latency_series_ms"] = [1.0] * len(r["latency_series_ms"])
    r["latency_series_ms"][3] = 99.0                      # one deadline miss (deadline 5 ms)
    trace, notes = tl.build([("run.json", r)])
    ev = trace["traceEvents"]
    json.dumps(trace)                                      # serializable
    names = {e["name"] for e in ev if e["ph"] == "X"}
    assert {"preprocess", "inference", "iteration"} <= names
    counters = {(e["name"], k) for e in ev if e["ph"] == "C" for k in e["args"]}
    assert {("CPU", "load %"), ("GPU", "load %"), ("GPU", "clock MHz"), ("EMC", "load %"),
            ("Memory", "used %"), ("Temperature", "max C"), ("Board", "power W"),
            ("Rail", "VDD_GPU_SOC W"), ("Rail", "VDD_CPU_CV W")} <= counters
    assert any(e["ph"] == "i" and "deadline miss" in e["name"] for e in ev)
    procs = [e["args"]["name"] for e in ev if e["ph"] == "M" and e["name"] == "process_name"]
    assert any("Board telemetry" in p for p in procs) and any("anomaly-like" in p for p in procs)
    xs = [e for e in ev if e["ph"] == "X"]
    assert min(e["ts"] for e in xs) >= 0 and all(e["dur"] >= 0 for e in xs)
    # spans and counters share one clock: the first counter is at the run start
    first_c = min(e["ts"] for e in ev if e["ph"] == "C")
    first_x = min(e["ts"] for e in xs)
    assert abs(first_c - first_x) < 50_000              # within 50 ms


def test_perfetto_merges_processes_on_one_clock(jetson_run):
    app_a = {"mode": "markers", "pipeline": {"name": "runtime-a"},
             "clock": {"trace_origin_mono_ns": 10_000_000_000,
                       "wall": {"wall_minus_mono_ns": 5_000_000_000_000}},
             "trace": {"events": [
                 {"iter": None, "stage": "@iteration", "start_ns": 0, "dur_ns": 3_000_000,
                  "thread": 101, "pid": 101, "flags": ""},
                 {"iter": 0, "stage": "can_decode", "start_ns": 0, "dur_ns": 1_000_000,
                  "thread": 101, "pid": 101, "flags": ""},
                 {"iter": 0, "stage": "lstm", "start_ns": 1_000_000, "dur_ns": 2_000_000,
                  "thread": 202, "pid": 202, "flags": "g"}]},
             "deadline": {"deadline_ms": 2.0}}
    app_b = copy.deepcopy(app_a)
    app_b["pipeline"]["name"] = "other"
    app_b["clock"]["trace_origin_mono_ns"] = 10_500_000_000      # 0.5 s later
    trace, _ = tl.build([("a.json", app_a), ("b.json", app_b)])
    assert trace["metadata"]["alignment"] == "mono"
    lstm = sorted(e["ts"] for e in trace["traceEvents"] if e.get("name") == "lstm")
    assert lstm == [1000.0, 501000.0]                        # µs, real offset kept
    assert {e["pid"] for e in trace["traceEvents"] if e.get("name") == "lstm"} == {202}
    assert any(e["ph"] == "i" for e in trace["traceEvents"])  # 3 ms iteration > 2 ms
    # different boots: aligned at each start, and said so
    app_b["clock"]["wall"]["wall_minus_mono_ns"] += 3600 * 10 ** 9
    trace, notes = tl.build([("a.json", app_a), ("b.json", app_b)])
    assert trace["metadata"]["alignment"] == "start" and notes


def test_perfetto_nothing_to_show_is_an_error():
    with pytest.raises(tl.TimelineError, match="Nothing to put on a timeline"):
        tl.build([("soak.json", {"soak": {}})])


def test_cli_export(tmp_path, jetson_run):
    run = _write(tmp_path, "run.json", jetson_run)
    out = tmp_path / "t.json"
    res = runner.invoke(app, ["export", str(run), "--perfetto", str(out)])
    assert res.exit_code == 0, res.output
    assert "ui.perfetto.dev" in res.output
    assert runner.invoke(app, ["export", str(run)]).exit_code == 2


TP = os.environ.get("EDGELENS_TRACE_PROCESSOR")


@pytest.mark.skipif(not TP, reason="set EDGELENS_TRACE_PROCESSOR=/path/to/trace_processor_shell")
def test_perfetto_trace_imports_in_the_real_trace_processor(tmp_path, jetson_run):
    """The engine behind ui.perfetto.dev reads the file with every track."""
    r = copy.deepcopy(jetson_run)
    r["telemetry_series"] = _jetson_series(r["clock"]["trace_origin_mono_ns"])
    out = tmp_path / "t.json"
    out.write_text(json.dumps(tl.build([("run.json", r)])[0]))
    q = ("select t.name as n, count(c.id) as k from counter_track t "
         "left join counter c on c.track_id = t.id group by t.name;")
    res = subprocess.run([TP, "-q", "/dev/stdin", str(out)], input=q, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    for track in ("CPU load %", "GPU load %", "GPU clock MHz", "Temperature max C",
                  "Board power W", "Rail VDD_GPU_SOC W"):
        assert track in res.stdout, res.stdout + res.stderr


# ---------------------------------------------------------------- E3 CI example

CI = ROOT / "examples" / "ci-jetson"


def test_ci_example_is_complete():
    for f in (".github/workflows/edgelens-perf.yml", "perf/anomaly.contract.yaml",
              "ci/perf_gate.sh", "pipeline.py", "README.md"):
        assert (CI / f).exists(), f
    c, _ = ct.load(CI / "perf" / "anomaly.contract.yaml")
    assert c["assumes"]["clocks_pinned"]["value"] is True
    wf = (CI / ".github/workflows/edgelens-perf.yml").read_text()
    assert "self-hosted" in wf and "concurrency" in wf and "perf_gate.sh" in wf


def _gate(tmp_path, contract_text, name):
    summary = tmp_path / f"{name}.summary.md"
    out = tmp_path / name
    (tmp_path / f"{name}.yaml").write_text(contract_text)
    env = dict(os.environ, GITHUB_STEP_SUMMARY=str(summary), EDGELENS_OUT=str(out),
               EDGELENS_CONTRACT=str(tmp_path / f"{name}.yaml"), PYTHON=sys.executable)
    res = subprocess.run(["bash", str(CI / "ci" / "perf_gate.sh")], cwd=str(CI), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True, timeout=300)
    return res, summary.read_text(), out


def test_ci_gate_passes_and_publishes_the_evidence(tmp_path):
    res, text, out = _gate(tmp_path, "contract: ci-smoke\nmin_duration: 1s\nassumes:\n"
                           "  input_period_ms: 20\nguarantees:\n  p99_ms: {limit: 1000, id: R-1}\n",
                           "ok")
    assert res.returncode == 0, res.stdout
    assert text.startswith("# ✅ PASS") and "R-1" in text and "ui.perfetto.dev" in text
    for f in ("run.json", "run.validation.json", "evidence.md", "timeline.perfetto.json"):
        assert (out / f).exists(), f


def test_ci_gate_inconclusive_when_the_board_is_not_the_assumed_one(tmp_path):
    # a board no test machine is: "AGX Orin" HOLDS when the suite runs on an
    # AGX Orin (found on one), and the gate then rightly measures and passes
    res, text, out = _gate(tmp_path, "contract: ci-smoke\nassumes:\n  board: AGX Thor\n"
                           "guarantees:\n  p99_ms: {limit: 20, id: R-1}\n", "orin")
    assert res.returncode == 2, res.stdout        # stopped before measuring: not that board
    assert "INCONCLUSIVE" in text and "board" in text.lower()
    assert not (out / "run.json").exists()


# ---------------------------------------------------------------- found by the validation run

def test_back_to_back_run_without_any_period_violates_the_input_period():
    r = run_harness(_pipe(), iterations=50, warmup=2)          # no period, no pace
    v = ct.evaluate(r, ct.parse({"contract": "x", "assumes": {"input_period_ms": 100},
                                 "guarantees": {"p50_ms": 1000}}))
    row = v["assumptions"][0]
    assert row["status"] == ct.VIOLATED and row["measured"] == "back-to-back"


def test_markers_rate_cross_check_needs_an_observed_rate():
    r = {"mode": "markers", "markers": {}, "requirements": {"period_ms": 1000},
         "fps": 50.0, "throughput_basis": "1 / mean latency (iterations back to back)",
         "latency": {"p50": 20.0}}
    row = ct.check_assumption(r, "input_period_ms", {"value": 1000.0, "tolerance_pct": 5.0})
    assert row["status"] == ct.HOLDS and row["basis"] == "declared"


def test_soak_with_a_period_counts_as_paced(tmp_path):
    from edgelens import soak as sk
    r = sk.run_soak_pipeline(_pipe(), 2.0, window_s=1.0, period_ms=20.0,
                             jsonl_path=str(tmp_path / "s.jsonl"))
    assert r["requirements"]["paced"] is True
    v = ct.evaluate(r, ct.parse({"contract": "x", "assumes": {"input_period_ms": 20,
                                                              "max_background_cpu_pct": 100},
                                 "guarantees": {"p50_ms": 1000}}))
    assert [a["status"] for a in v["assumptions"]] == [ct.HOLDS, ct.HOLDS]
    assert r["telemetry"]["background"]["windows"] >= 1


def test_validate_flags_with_evidence_on_fail_writes_only_the_requested_file(tmp_path, jetson_run,
                                                                             monkeypatch):
    # the FAIL branch has a local "evidence" (diagnosis text): it once became a file name
    monkeypatch.chdir(tmp_path)
    run = _write(tmp_path, "run.json", jetson_run)
    res = runner.invoke(app, ["validate", str(run), "--p99-ms", "0.0001", "--evidence", "e.md"])
    assert res.exit_code == 1, res.output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["e.md", "run.json",
                                                         "run.validation.json"]
    assert "requirements given as flags" in (tmp_path / "e.md").read_text()

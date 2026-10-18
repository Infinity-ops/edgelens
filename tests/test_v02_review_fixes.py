"""Regression tests for the issues found in the independent v0.2 review.
Each test reproduces the reported failure scenario."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from edgelens.attach import observe, to_result
from edgelens.core import integrity
from edgelens.markers import INCLUDE_DIR
from edgelens.markers.collector import MarkerCollector
from edgelens.markers.session import spans_to_result
from edgelens.validate.engine import validate

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
CC = shutil.which("cc") or shutil.which("gcc")


@pytest.fixture
def tmpdir_short():
    d = tempfile.mkdtemp(prefix="elr")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _py_app(tmp, n, period_s, work_s, name="app.py"):
    p = Path(tmp) / name
    p.write_text(
        "import time\nfrom edgelens import mark\n"
        f"for i in range({n}):\n"
        "    with mark.iteration():\n"
        f"        with mark.span('lstm'): time.sleep({work_s})\n"
        f"    time.sleep({period_s - work_s})\n")
    return str(p)


def _codes(result):
    return [i["code"] for i in integrity.check(result)]


# #1 throughput is the observed rate, not 1 / latency ------------------------

@linux_only
def test_observed_throughput_is_the_real_rate(tmpdir_short):
    run = observe(cmd=[sys.executable, _py_app(tmpdir_short, 12, 0.1, 0.02)],
                  process_interval_s=0.2)
    r = to_result(run, name="app")
    assert r["iterations"] == 12
    assert 7.0 < r["fps"] < 12.0, r["fps"]            # ~10 Hz, not ~45 (1 / 22 ms)
    assert r["throughput_basis"] == "observed rate over the run"
    assert validate(r, {"min_fps": 30})["verdict"] == "FAIL"


def test_paced_harness_throughput_is_the_release_rate():
    import edgelens as el
    p = el.Pipeline("x")
    p.add_stage("work", lambda x=None: time.sleep(0.002))
    r = p.run(iterations=25, warmup=1, period_ms=20, pace=True, telemetry=False)
    assert 40 < r["fps"] < 55, r["fps"]                # 50 Hz releases, not 1 / 2 ms


# #2 other processes' spans don't mix in -------------------------------------

@linux_only
def test_record_command_uses_a_private_socket(tmpdir_short, monkeypatch):
    shared = os.path.join(tmpdir_short, "shared.sock")
    monkeypatch.setenv("EDGELENS_SOCKET", shared)
    other = subprocess.Popen([sys.executable, _py_app(tmpdir_short, 60, 0.05, 0.03, "o.py")],
                             env=dict(os.environ, EDGELENS_SOCKET=shared))
    try:
        run = observe(cmd=[sys.executable, _py_app(tmpdir_short, 8, 0.05, 0.01)],
                      process_interval_s=0.2)
    finally:
        other.kill()
        other.wait()
    r = to_result(run, name="app", deadline_ms=25)
    assert r["iterations"] == 8 and r["markers"]["processes"] == [run.target["pid"]]
    assert r["deadline"]["misses"] == 0
    assert r["markers"]["private_socket"] is True


@linux_only
def test_watch_pid_excludes_foreign_spans_on_the_shared_socket(tmpdir_short):
    sock = os.path.join(tmpdir_short, "s.sock")
    env = dict(os.environ, EDGELENS_SOCKET=sock)
    target = subprocess.Popen([sys.executable, "-c",
                               "import time,sys\nfrom edgelens import mark\n"
                               "time.sleep(0.5)\n"
                               "for i in range(10):\n"
                               "    with mark.span('lstm'): time.sleep(0.01)\n"
                               "    time.sleep(0.05)\n"], env=env)
    other = subprocess.Popen([sys.executable, _py_app(tmpdir_short, 40, 0.03, 0.02, "o.py")],
                             env=env)
    try:
        run = observe(pid=target.pid, socket_path=sock, process_interval_s=0.1)
    finally:
        other.kill()
        other.wait()
        target.wait()
    r = to_result(run, name="watch", mode="attach")
    m = r["markers"]
    assert r["iterations"] == 10 and m["processes"] == [target.pid]
    assert m["foreign_spans"] > 0 and other.pid in m["foreign_pids"]
    assert "FOREIGN_SPANS_EXCLUDED" in _codes(r)


# #3 killed sender: tail losses can't be counted -> no PASS -------------------

C_BURST_THEN_WAIT = r"""
#include "edgelens_mark.h"
#include <unistd.h>
int main(void) {
    for (int i = 0; i < 300; i++) { el_span_t s = el_begin("lstm"); el_end(s); }
    pause();                       /* killed by SIGINT: no exit flush */
    return 0;
}
"""


@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_sender_killed_with_pending_spans_blocks_pass(tmpdir_short):
    src = Path(tmpdir_short) / "b.c"
    exe = Path(tmpdir_short) / "b"
    src.write_text(C_BURST_THEN_WAIT)
    subprocess.run([CC, "-O2", "-I", INCLUDE_DIR, str(src), "-o", str(exe)], check=True)
    run = observe(cmd=[str(exe)], duration_s=1.0, process_interval_s=0.2)
    r = to_result(run, name="burst", deadline_ms=100)
    snd = r["markers"]["senders"][0]
    assert snd["final_seen"] is False
    issues = {i["code"]: i for i in integrity.check(r)}
    # EdgeLens stopped the target and it never said goodbye: whether or not a
    # batch was ever seen, pending spans may have died with it -> no PASS.
    assert issues["SENDER_NOT_FINISHED_CLEANLY"]["severity"] == integrity.BLOCKS
    assert validate(r, {"max_latency_ms": 100})["verdict"] == "INCONCLUSIVE"


def test_watch_pid_still_running_at_the_end_only_warns():
    ms = 1_000_000
    spans = [{"kind": "X", "src": "a", "seq": i, "pid": 9, "tid": 9, "start_ns": i * 10 * ms,
              "end_ns": i * 10 * ms + ms, "flags": "", "name": "lstm"} for i in range(150)]
    r = spans_to_result(spans, {"received": 150, "lost": 0,
                                "senders": [{"pid": 9, "final_seen": False,
                                             "backpressure": False}]},
                        deadline_ms=5, attach_ids=False,
                        extra={"observation": {"target": {"kind": "pid", "pid": 9},
                                               "ended_by": "duration reached"}})
    sev = {i["code"]: i["severity"] for i in integrity.check(r)}
    assert sev["SENDER_NOT_FINISHED_CLEANLY"] == integrity.WARN
    assert validate(r, {"p99_ms": 5})["verdict"] == "PASS"


@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_cpp_example_exits_cleanly_on_sigint(tmpdir_short):
    root = Path(__file__).resolve().parents[1]
    exe = Path(tmpdir_short) / "cpp_app"
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    subprocess.run([cxx, "-O2", "-std=c++11", "-I", INCLUDE_DIR,
                    str(root / "examples" / "markers" / "cpp_app.cpp"), "-o", str(exe)],
                   check=True)
    run = observe(cmd=[str(exe), "100000"], duration_s=1.0, process_interval_s=0.2)
    r = to_result(run, name="cpp")
    assert run.exit_code == 0
    assert all(s["final_seen"] for s in r["markers"]["senders"])
    assert "SENDER_NOT_FINISHED_CLEANLY" not in _codes(r)


# #4 fork: child is a new sender ---------------------------------------------

C_FORK = r"""
#include "edgelens_mark.h"
#include <sys/wait.h>
#include <unistd.h>
int main(void) {
    for (int i = 0; i < 50; i++) { el_span_t s = el_begin("pre"); el_end(s); usleep(200); }
    pid_t p = fork();
    for (int i = 0; i < 50; i++) { el_span_t s = el_begin("post"); el_end(s); usleep(200); }
    if (p > 0) waitpid(p, 0, 0);
    return 0;
}
"""


@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_fork_gives_the_child_its_own_sequence(tmpdir_short):
    src = Path(tmpdir_short) / "f.c"
    exe = Path(tmpdir_short) / "f"
    src.write_text(C_FORK)
    subprocess.run([CC, "-O2", "-I", INCLUDE_DIR, str(src), "-o", str(exe)], check=True)
    sock = os.path.join(tmpdir_short, "s.sock")
    c = MarkerCollector(sock).start()
    subprocess.run([str(exe)], check=True, env=dict(os.environ, EDGELENS_SOCKET=sock))
    time.sleep(0.3)
    spans, st = c.stop()
    assert st["received"] == 150 and st["lost"] == 0 and st["duplicate_seq"] == 0
    assert len({s["pid"] for s in spans}) == 2
    assert len(st["senders"]) == 2 and all(x["final_seen"] for x in st["senders"])


@linux_only
def test_python_fork_gives_the_child_its_own_sequence(tmpdir_short):
    sock = os.path.join(tmpdir_short, "s.sock")
    code = ("import os, time\nfrom edgelens import mark\n"
            "for i in range(30):\n    with mark.span('pre'): pass\n    time.sleep(0.001)\n"
            "pid = os.fork()\n"
            "for i in range(30):\n    with mark.span('post'): pass\n    time.sleep(0.001)\n"
            "if pid: os.waitpid(pid, 0)\nelse:\n    mark.flush(); os._exit(0)\n")
    c = MarkerCollector(sock).start()
    subprocess.run([sys.executable, "-c", code], check=True,
                   env=dict(os.environ, EDGELENS_SOCKET=sock))
    time.sleep(0.3)
    spans, st = c.stop()
    assert st["received"] == 90 and st["lost"] == 0 and st["duplicate_seq"] == 0
    assert len({s["pid"] for s in spans}) == 2


def test_duplicate_sequence_blocks_pass():
    ms = 1_000_000
    spans = [{"kind": "X", "src": "a", "seq": i, "pid": 1, "tid": 1, "start_ns": i * 10 * ms,
              "end_ns": i * 10 * ms + ms, "flags": "", "name": "lstm"} for i in range(150)]
    r = spans_to_result(spans, {"received": 150, "lost": 0, "duplicate_seq": 50,
                                "senders": [{"pid": 1, "final_seen": True}]},
                        deadline_ms=5, attach_ids=False)
    assert validate(r, {"p99_ms": 5})["verdict"] == "INCONCLUSIVE"


# #5 energy only over the active window --------------------------------------

def test_energy_window_is_interpolated_and_never_falls_back():
    from edgelens.core.result import windowed_energy
    s = 1_000_000_000
    series = [{"t_mono_ns": t * s, "power_w": 10.0 if t < 99 else 2.0} for t in range(0, 101)]
    # 0.8 s active window between two samples: interpolated, flagged low confidence
    e = windowed_energy(series, int(99.1 * s), int(99.9 * s), iterations=2)
    assert e["available"] and e["power_w_mean"] == pytest.approx(2.0)
    assert e["energy_per_iteration_j"] == pytest.approx(0.8, rel=0.01)
    assert "low_confidence" in e
    # window outside the series: unavailable, never the whole-run figure
    e = windowed_energy(series, 200 * s, 201 * s, iterations=2)
    assert e["available"] is False and "cover" in e["reason"]


def test_energy_excludes_startup_before_the_first_span():
    from edgelens.core.result import windowed_energy
    s = 1_000_000_000
    series = [{"t_mono_ns": t * s // 10, "power_w": 10.0 if t < 600 else 2.0}
              for t in range(0, 1000)]                       # 60 s load at 10 W, then 2 W
    e = windowed_energy(series, 60 * s, 99 * s, iterations=39)
    assert e["power_w_mean"] == pytest.approx(2.0, abs=0.01)
    assert e["energy_per_iteration_j"] == pytest.approx(2.0, rel=0.02)   # not ~17 J


# #6 iterations across threads -----------------------------------------------

def test_spans_from_several_threads_without_iteration_markers_are_ambiguous():
    ms = 1_000_000
    spans = []
    for i in range(120):
        t = i * 20 * ms
        spans.append({"kind": "X", "src": "a", "seq": 2 * i, "pid": 1, "tid": 11,
                      "start_ns": t, "end_ns": t + ms, "flags": "", "name": "capture"})
        spans.append({"kind": "X", "src": "a", "seq": 2 * i + 1, "pid": 1, "tid": 12,
                      "start_ns": t + 5 * ms, "end_ns": t + 9 * ms, "flags": "",
                      "name": "lstm"})
    r = spans_to_result(spans, {"received": 240, "lost": 0, "senders": []},
                        deadline_ms=50, attach_ids=False)
    assert r["markers"]["iteration_streams"] == 2
    assert "ITERATIONS_AMBIGUOUS" in _codes(r)
    assert validate(r, {"p99_ms": 50})["verdict"] == "INCONCLUSIVE"
    assert r["stage_stats_ms"]["lstm"]["mean"] == pytest.approx(4.0)   # stage timing still fine


# #9 ROS: unmatched / missing messages block a chain PASS --------------------

def test_ros_chain_with_missing_messages_blocks_pass():
    from edgelens.ros2 import latency as rl
    from tests.test_v02_ros import FixedClock, _streams
    src, dst, ck = _streams(drop=range(50, 60))       # inputs dropped: outputs unmatched
    s, chain, ages = rl.summarize(src, dst, clock_map=ck, same_machine=True)
    r = rl.to_result(s, chain, ages, from_topic="/imu", to_topic="/pred", deadline_ms=20,
                     attach_ids=False)
    assert "ROS_MESSAGES_MISSING" in _codes(r)
    assert validate(r, {"max_latency_ms": 20})["verdict"] == "INCONCLUSIVE"
    assert 8 < r["fps"] < 12        # matched outputs arrive at ~10 Hz, not 1 / 7 ms
    del FixedClock


# #14 small items --------------------------------------------------------------

def test_repeated_runs_do_not_overwrite_each_others_verdicts(tmp_path):
    from typer.testing import CliRunner
    from edgelens.cli import app
    fixture = Path(__file__).parent / "fixtures" / "result_v0_1_0.json"
    for i in (1, 2):
        (tmp_path / f"base.r0{i}.json").write_text(fixture.read_text())
        CliRunner().invoke(app, ["validate", str(tmp_path / f"base.r0{i}.json"),
                                 "--p99-ms", "100"])
    assert (tmp_path / "base.r01.validation.json").exists()
    assert (tmp_path / "base.r02.validation.json").exists()


def test_python_dash_m_cli_knows_v02_commands():
    res = subprocess.run([sys.executable, "-m", "edgelens.cli", "record", "--help"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)
    assert res.returncode == 0 and "markers" in res.stdout.lower()


def test_unknown_gpu_load_is_not_reported_as_idle():
    ms = 1_000_000
    spans = [{"kind": "X", "src": "a", "seq": i, "pid": 1, "tid": 1, "start_ns": i * 10 * ms,
              "end_ns": i * 10 * ms + ms, "flags": "", "name": "lstm"} for i in range(10)]
    r = spans_to_result(spans, {"received": 10, "lost": 0, "senders": []},
                        telemetry_summary={"gpu_percent_mean": None}, attach_ids=False)
    issue = [i for i in integrity.check(r) if i["code"] == "GPU_TIMING_NOT_SYNCED"]
    assert issue and "unknown" in issue[0]["detail"]


@linux_only
def test_watch_pid_keeps_short_lived_worker_children(tmpdir_short):
    sock = os.path.join(tmpdir_short, "s.sock")
    env = dict(os.environ, EDGELENS_SOCKET=sock)
    worker = ("import time\nfrom edgelens import mark\n"
              "with mark.span('infer'): time.sleep(0.02)\n")
    parent = subprocess.Popen([sys.executable, "-c",
                               "import subprocess, sys, time\n"
                               "time.sleep(0.6)\n"
                               "for i in range(6):\n"
                               f"    subprocess.run([sys.executable, '-c', {worker!r}])\n"
                               "    time.sleep(0.05)\n"], env=env)
    run = observe(pid=parent.pid, socket_path=sock, process_interval_s=0.5)
    parent.wait()
    r = to_result(run, name="w", mode="attach")
    assert r["iterations"] == 6, r["markers"]
    assert not r["markers"].get("foreign_unverified")


def test_iteration_lookup_stays_fast_with_background_spans():
    ms = 1_000_000
    spans = []
    for i in range(16000):
        t = i * 30 * ms
        spans.append({"kind": "X", "src": "a", "seq": 0, "pid": 1, "tid": 1, "start_ns": t,
                      "end_ns": t + 10 * ms, "flags": "", "name": "@iteration"})
        spans.append({"kind": "X", "src": "a", "seq": 0, "pid": 1, "tid": 2,
                      "start_ns": t + 15 * ms, "end_ns": t + 16 * ms, "flags": "",
                      "name": "logger"})
    t0 = time.perf_counter()
    r = spans_to_result(spans, {"received": 32000, "lost": 0, "senders": []}, attach_ids=False)
    assert time.perf_counter() - t0 < 10.0
    assert r["iterations"] == 16000


@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_fork_fallback_without_pthread_atfork(tmpdir_short):
    src = Path(tmpdir_short) / "f.c"
    exe = Path(tmpdir_short) / "f"
    src.write_text(C_FORK)
    subprocess.run([CC, "-O2", "-DEDGELENS_TEST_NO_ATFORK", "-I", INCLUDE_DIR, str(src),
                    "-o", str(exe)], check=True)
    sock = os.path.join(tmpdir_short, "s.sock")
    c = MarkerCollector(sock).start()
    subprocess.run([str(exe)], check=True, env=dict(os.environ, EDGELENS_SOCKET=sock))
    time.sleep(0.3)
    spans, st = c.stop()
    assert st["received"] == 150 and st["lost"] == 0 and st["duplicate_seq"] == 0
    assert len({s["pid"] for s in spans}) == 2


def test_compare_glob_ignores_diagnose_and_validate_outputs(tmp_path):
    from typer.testing import CliRunner
    from edgelens.cli import app
    fixture = (Path(__file__).parent / "fixtures" / "result_v0_1_0.json").read_text()
    for side in ("a", "b"):
        for i in (1, 2, 3):
            (tmp_path / f"{side}.r0{i}.json").write_text(fixture)
    runner = CliRunner()
    runner.invoke(app, ["diagnose", str(tmp_path / "a.r01.json")])
    runner.invoke(app, ["validate", str(tmp_path / "a.r02.json"), "--p99-ms", "100"])
    res = runner.invoke(app, ["compare", str(tmp_path / "a.r*.json"), str(tmp_path / "b.r*.json")])
    assert res.exit_code == 0, res.output
    assert "3 vs 3 runs" in res.output


C_FORK_PENDING_CHILD_EXITS = r"""
#include "edgelens_mark.h"
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
int main(void) {
    for (int i = 0; i < 400; i++) { el_span_t s = el_begin("burst"); el_end(s); }
    pid_t p = fork();                  /* parent likely has a pending batch now */
    if (p == 0) exit(0);               /* child records nothing and exits */
    waitpid(p, 0, 0);
    return 0;
}
"""


@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
@pytest.mark.parametrize("flags", [[], ["-DEDGELENS_TEST_NO_ATFORK"]])
def test_forked_child_never_resends_parent_pending_spans(tmpdir_short, flags):
    src = Path(tmpdir_short) / "p.c"
    exe = Path(tmpdir_short) / "p"
    src.write_text(C_FORK_PENDING_CHILD_EXITS)
    subprocess.run([CC, "-O2", *flags, "-I", INCLUDE_DIR, str(src), "-o", str(exe)], check=True)
    sock = os.path.join(tmpdir_short, "s.sock")
    c = MarkerCollector(sock).start()
    subprocess.run([str(exe)], check=True, env=dict(os.environ, EDGELENS_SOCKET=sock))
    time.sleep(0.3)
    _, st = c.stop()
    assert st["duplicate_seq"] == 0
    assert st["received"] + st["lost"] == 400, st

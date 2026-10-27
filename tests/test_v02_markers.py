"""v0.2 Layer 1: markers (S1) and attach mode (S2), plus the clock (T2)."""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from edgelens.core import clock
from edgelens.markers import HEADER, INCLUDE_DIR, protocol
from edgelens.markers.collector import CollectorError, MarkerCollector
from edgelens.markers.logimport import LogImportError, parse_lines
from edgelens.markers.session import spans_to_result

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc + AF_UNIX")


@pytest.fixture
def sock_path():
    d = tempfile.mkdtemp()
    yield os.path.join(d, "el.sock")
    shutil.rmtree(d, ignore_errors=True)


def _send(path, payloads):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    for p in payloads:
        s.sendto(p, path)
    s.close()


def _span(seq, start, end, name, src="a", pid=1, flags="-"):
    return protocol.format_span(src, seq, pid, 7, start, end, flags, name)


# ---------------------------------------------------------------- T2 clock

def test_clock_is_monotonic_and_matches_perf_counter_on_linux():
    a, b = clock.now_ns(), clock.now_ns()
    assert b >= a
    if sys.platform.startswith("linux"):
        assert clock.perf_counter_is_monotonic()


def test_wall_offset_round_trips_with_small_uncertainty():
    m = clock.ClockMap()
    mono = clock.now_ns()
    assert abs(m.wall_to_mono(m.mono_to_wall(mono)) - mono) == 0
    assert m.uncertainty_ns < 5_000_000          # < 5 ms even on a loaded CI box
    assert abs(m.mono_to_wall(clock.now_ns()) - time.time_ns()) < 50_000_000


def test_chrony_csv_parse_and_bound():
    line = ("C0A80001,192.168.0.1,3,1760000000.123,0.000012,-0.000003,0.000020,"
            "-12.5,0.001,0.05,0.004000,0.000500,64.2,Normal")
    st = clock.parse_chrony_tracking_csv(line)
    assert st["synchronized"] and st["stratum"] == 3
    # |offset| + root_dispersion + root_delay / 2
    assert st["error_bound_s"] == pytest.approx(0.000012 + 0.0005 + 0.002)
    unsynced = line.replace("Normal", "Not synchronised")
    assert clock.parse_chrony_tracking_csv(unsynced)["synchronized"] is False
    assert clock.parse_chrony_tracking_csv("garbage") is None
    assert clock.cross_machine_bound_s(st, None) is None          # remote unknown
    assert clock.cross_machine_bound_s(st, 0.001) == pytest.approx(st["error_bound_s"] + 0.001)


def test_results_carry_clock_block_and_series(tmp_path):
    import edgelens as el
    pipe = el.Pipeline("p")
    pipe.add_stage("work", lambda x=None: sum(range(2000)))
    r = pipe.run(iterations=120, warmup=2)
    c = r["clock"]
    assert c["clock"] == "CLOCK_MONOTONIC"
    assert c["trace_origin_mono_ns"] and c["telemetry_origin_mono_ns"]
    assert len(r["latency_series_ms"]) == 120
    for s in r["telemetry_series"]:
        assert s["t_mono_ns"] is not None and s["t_s"] is not None


# ---------------------------------------------------------------- protocol

def test_protocol_roundtrip_and_malformed():
    msg = protocol.parse(_span(5, 100, 250, "lstm inference", flags="g"))
    assert msg == {"kind": "X", "src": "a", "seq": 5, "pid": 1, "tid": 7, "start_ns": 100,
                   "end_ns": 250, "flags": "g", "name": "lstm inference"}
    assert protocol.parse(b"EL1 X a 1 2 3 400 100 - backwards") is None      # end < start
    assert protocol.parse(b"hello") is None
    assert protocol.parse(b"EL2 X a 1 2 3 4 5 - x") is None
    assert protocol.parse(protocol.format_final("a", 9, 1)) == {"kind": "F", "src": "a",
                                                               "seq": 9, "pid": 1}


# ---------------------------------------------------------------- collector

@linux_only
def test_collector_counts_gaps_and_final_exactly(sock_path):
    c = MarkerCollector(sock_path).start()
    # seq 0..9 sent, 3 and 7 "dropped by the sender", final says last was 12
    msgs = [_span(i, i * 10, i * 10 + 5, "s") for i in range(10) if i not in (3, 7)]
    msgs += [b"junk", protocol.format_final("a", 12, 1)]
    _send(sock_path, msgs)
    time.sleep(0.3)
    spans, st = c.stop()
    assert st["received"] == 8
    assert st["lost"] == 2 + 3            # two gaps + 10, 11, 12 after the last span
    assert st["malformed"] == 1
    assert st["senders"][0]["final_seen"]
    assert not os.path.exists(sock_path)  # cleaned up


@linux_only
def test_collector_does_not_count_spans_before_it_started(sock_path):
    c = MarkerCollector(sock_path).start()
    _send(sock_path, [_span(i, i, i + 1, "s") for i in range(100, 110)])
    time.sleep(0.3)
    _, st = c.stop()
    assert st["lost"] == 0 and st["senders"][0]["seq_before_first"] == 100


@linux_only
def test_collector_refuses_a_live_collector_and_removes_a_stale_socket(sock_path):
    c = MarkerCollector(sock_path).start()
    with pytest.raises(CollectorError, match="already listening"):
        MarkerCollector(sock_path).start()
    c.stop()
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    stale.bind(sock_path)
    stale.close()                          # file left behind, nobody listening
    c2 = MarkerCollector(sock_path).start()
    c2.stop()
    Path(sock_path).write_text("not a socket")
    with pytest.raises(CollectorError, match="not a socket"):
        MarkerCollector(sock_path).start()


# ---------------------------------------------------------------- session

def _x(name, start, end, pid=1, flags=""):
    return {"kind": "X", "src": "a", "seq": 0, "pid": pid, "tid": 1, "start_ns": start,
            "end_ns": end, "flags": flags, "name": name}


def test_explicit_iterations_include_gaps_between_spans():
    ms = 1_000_000
    spans = []
    for i in range(3):
        t = i * 100 * ms
        spans += [_x("@iteration", t, t + 10 * ms), _x("pre", t, t + 2 * ms),
                  _x("lstm", t + 5 * ms, t + 9 * ms)]
    r = spans_to_result(spans, {"received": 9, "lost": 0}, deadline_ms=9.5, attach_ids=False)
    assert r["iterations"] == 3
    assert r["latency"]["mean"] == pytest.approx(10.0)       # not 2 + 4: gaps count
    assert r["stage_avg_ms"] == {"pre": 2.0, "lstm": 4.0}
    assert r["deadline"]["misses"] == 3
    assert {e["iter"] for e in r["trace"]["events"] if e["stage"] == "lstm"} == {0, 1, 2}
    assert r["markers"]["iterations_from"] == "@iteration spans"


def test_auto_iterations_per_process_and_gpu_sync_check():
    ms = 1_000_000
    spans = []
    for i in range(4):
        t = i * 50 * ms
        spans += [_x("window", t, t + ms), _x("lstm", t + ms, t + 3 * ms, flags="")]
        spans += [_x("window", t, t + ms, pid=2), _x("lstm", t + ms, t + 4 * ms, pid=2,
                                                     flags="g")]
    r = spans_to_result(spans, {}, attach_ids=False)
    assert r["iterations"] == 8                                # 4 per process
    g = r["markers"]["gpu_timing"]
    assert g["inference_spans"] == 8 and g["unsynced_inference_spans"] == 4
    assert g["unsynced_inference_stages"] == ["lstm"]
    assert r["markers"]["processes"] == [1, 2]


# ---------------------------------------------------------------- python client

@linux_only
def test_python_client_end_to_end(sock_path, tmp_path):
    c = MarkerCollector(sock_path).start()
    script = tmp_path / "app.py"
    script.write_text(
        "import time\nfrom edgelens import mark\n"
        "for i in range(50):\n"
        "    with mark.iteration():\n"
        "        with mark.span('pre'): time.sleep(0.001)\n"
        "        with mark.span('lstm', sync=lambda: None): time.sleep(0.002)\n")
    env = dict(os.environ, EDGELENS_SOCKET=sock_path)
    subprocess.run([sys.executable, str(script)], check=True, env=env, timeout=60)
    time.sleep(0.3)
    spans, st = c.stop()
    assert st["received"] == 150 and st["lost"] == 0
    assert st["senders"][0]["final_seen"]
    r = spans_to_result(spans, st, attach_ids=False)
    assert r["iterations"] == 50
    assert r["stage_stats_ms"]["lstm"]["mean"] >= 2.0
    assert r["markers"]["gpu_timing"]["unsynced_inference_spans"] == 0


@linux_only
def test_python_client_never_blocks_without_collector(sock_path, monkeypatch):
    from edgelens import mark
    mark.configure(sock_path)              # nobody listening there
    t0 = time.perf_counter()
    for _ in range(1000):
        with mark.span("x"):
            pass
    assert time.perf_counter() - t0 < 1.0
    assert mark.stats()["dropped"] >= 1000


# ---------------------------------------------------------------- C header

CC = shutil.which("cc") or shutil.which("gcc")
CXX = shutil.which("c++") or shutil.which("g++")

C_APP = r"""
#include "edgelens_mark.h"
#include <stdio.h>
static void sync_fn(void *ctx) { (void)ctx; }
int main(void) {
    int i;
    for (i = 0; i < 100; i++) {
        el_span_t it = el_begin("@iteration");
        el_span_t a = el_begin("preprocess"); el_end(a);
        el_span_t b = el_begin("lstm"); el_end_synced(b, sync_fn, 0);
        el_end(it);
    }
    return 0;
}
"""

CPP_APP = r"""
#include "edgelens_mark.h"
#include <cstdlib>
#include <thread>
#include <vector>
static void spin_us(long us) { uint64_t t = el_now_ns() + us * 1000ull; while (el_now_ns() < t) {} }
int main(int argc, char **argv) {
    long work_us = argc > 1 ? atol(argv[1]) : 0;      // 0 = pathological burst
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; t++) ts.emplace_back([work_us] {
        for (int i = 0; i < 250; i++) { EL_ITERATION(); { EL_SPAN("work"); spin_us(work_us); } }
    });
    for (auto &t : ts) t.join();
    return 0;
}
"""


def _build(compiler, src, tmp_path, name, flags):
    f = tmp_path / name
    f.write_text(src)
    out = tmp_path / (name + ".bin")
    subprocess.run([compiler] + flags + ["-Wall", "-Wextra", "-Werror", "-O2", "-I",
                                         INCLUDE_DIR, str(f), "-o", str(out)], check=True)
    return str(out)


def test_header_ships_with_package():
    assert os.path.isfile(HEADER)
    assert "EL1 X" in Path(HEADER).read_text()


@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_header_end_to_end(sock_path, tmp_path):
    exe = _build(CC, C_APP, tmp_path, "app.c", ["-std=c99"])
    c = MarkerCollector(sock_path).start()
    subprocess.run([exe], check=True, env=dict(os.environ, EDGELENS_SOCKET=sock_path))
    time.sleep(0.3)
    spans, st = c.stop()
    assert st["received"] == 300 and st["lost"] == 0
    assert st["senders"][0]["final_seen"]            # destructor sent the final message
    r = spans_to_result(spans, st, attach_ids=False)
    assert r["iterations"] == 100
    assert r["markers"]["gpu_timing"]["unsynced_inference_spans"] == 0
    assert all("g" in s["flags"] for s in spans if s["name"] == "lstm")


@linux_only
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_cpp_header_threads_and_disabled_build(sock_path, tmp_path):
    exe = _build(CXX, CPP_APP, tmp_path, "app.cpp", ["-std=c++11", "-pthread"])
    env = dict(os.environ, EDGELENS_SOCKET=sock_path)

    # Realistic rate (4 threads, a span every ~1 ms each = 4000 spans/s, far
    # above a 30 Hz or 1 Hz pipeline): nothing may be lost.
    c = MarkerCollector(sock_path).start()
    subprocess.run([exe, "1000"], check=True, env=env)
    time.sleep(0.3)
    spans, st = c.stop()
    assert st["received"] == 2000 and st["lost"] == 0, st
    assert spans_to_result(spans, st, attach_ids=False)["iterations"] == 1000

    # Pathological burst (2000 spans in well under a millisecond): some may be
    # dropped, but the accounting must be exact — every span is either
    # received or counted as lost, never silently missing.
    c = MarkerCollector(sock_path).start()
    subprocess.run([exe, "0"], check=True, env=env)
    time.sleep(0.3)
    _, st = c.stop()
    assert st["received"] + st["lost"] == 2000, st
    assert all(s["final_seen"] for s in st["senders"])

    off = _build(CXX, CPP_APP, tmp_path, "off.cpp", ["-std=c++11", "-pthread",
                                                     "-DEDGELENS_DISABLE"])
    subprocess.run([off], check=True)


@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_header_without_collector_is_silent_and_fast(tmp_path):
    exe = _build(CC, C_APP, tmp_path, "nocol.c", [])
    t0 = time.perf_counter()
    subprocess.run([exe], check=True, env=dict(os.environ,
                                               EDGELENS_SOCKET=str(tmp_path / "none.sock")))
    assert time.perf_counter() - t0 < 2.0


# ---------------------------------------------------------------- log import

def test_log_import_durations_and_stage_groups():
    lines = ["x", "LSTM took 12.5 ms", "LSTM took 13.5 ms", "pre took 900 us"]
    spans, info = parse_lines(lines, r"(?P<stage>\w+) took (?P<ms>[0-9.]+) ms", "inference")
    assert [s["name"] for s in spans] == ["LSTM", "LSTM"]
    assert spans[0]["end_ns"] - spans[0]["start_ns"] == 12_500_000
    r = spans_to_result(spans, info, attach_ids=False, source="log")
    assert r["iterations"] == 2 and r["markers"]["source"] == "log"
    spans, _ = parse_lines(lines, r"pre took (?P<us>[0-9.]+) us")
    assert spans[0]["end_ns"] - spans[0]["start_ns"] == 900_000
    with pytest.raises(LogImportError):
        parse_lines(lines, r"took ([0-9.]+)")
    with pytest.raises(LogImportError):
        parse_lines(lines, r"(?P<ms>[")


def test_log_import_with_timestamps():
    lines = ["t=1000 e=6000 infer", "t=9000 e=11000 infer"]
    spans, info = parse_lines(lines, r"t=(?P<start_ns>\d+) e=(?P<end_ns>\d+)")
    assert [(s["start_ns"], s["end_ns"]) for s in spans] == [(1000, 6000), (9000, 11000)]
    assert info["timestamps"] == "from log"


# ---------------------------------------------------------------- attach (S2)

@linux_only
def test_watch_measures_a_child_process_and_memory_growth():
    from edgelens.attach import observe, to_result
    code = ("import time\nb=[]\nt=time.time()\n"
            "while time.time()-t<2.0:\n b.append(bytearray(2_000_000)); time.sleep(0.05)\n")
    run = observe(cmd=[sys.executable, "-c", code], markers=False, process_interval_s=0.2)
    r = to_result(run, name="leaky", mode="attach")
    p = r["process"]
    assert run.ended_by == "command exited" and run.exit_code == 0
    assert p["samples"] >= 5
    assert p["rss_mb_end"] > p["rss_mb_start"] + 40
    assert p["rss_growth_mb_per_min"] > 600
    assert r["iterations"] == 0                         # latency honestly not measured
    assert "not measured" in r["markers"]["note"]
    json.dumps(r)                                       # serialisable


@linux_only
def test_watch_with_markers_combines_latency_and_process(sock_path, tmp_path):
    from edgelens.attach import observe, to_result
    script = tmp_path / "app.py"
    script.write_text("import time\nfrom edgelens import mark\n"
                      "for i in range(40):\n"
                      "    with mark.span('lstm'): time.sleep(0.005)\n")
    run = observe(cmd=[sys.executable, str(script)], socket_path=sock_path,
                  process_interval_s=0.2)
    r = to_result(run, name="app", deadline_ms=20, mode="attach")
    assert r["iterations"] == 40 and r["deadline"]["misses"] == 0
    assert r["process"]["samples"] >= 1
    assert r["markers"]["lost"] == 0


def test_find_process_errors_are_explicit():
    from edgelens.attach import TargetError, find_process
    with pytest.raises(TargetError, match="No running process"):
        find_process("definitely-not-a-process-name-xyz")


# ---------------------------------------------------------------- CLI

@linux_only
@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_cli_record_runs_a_command_and_saves(tmp_path):
    from typer.testing import CliRunner
    from edgelens.cli import app
    exe = _build(CC, C_APP, tmp_path, "cli.c", [])
    out = tmp_path / "rec.json"
    res = CliRunner().invoke(app, ["record", "--socket", str(tmp_path / "s.sock"),
                                   "--deadline-ms", "50", "--save", str(out), "--", exe])
    assert res.exit_code == 0, res.output
    r = json.loads(out.read_text())
    assert r["mode"] == "markers" and r["iterations"] == 100
    assert r["observation"]["exit_code"] == 0
    res = CliRunner().invoke(app, ["validate", str(out), "--p99-ms", "50"])
    assert res.exit_code == 0, res.output


def test_cli_import_log_and_include(tmp_path):
    from typer.testing import CliRunner
    from edgelens.cli import app
    log = tmp_path / "a.log"
    log.write_text("".join(f"lstm took {10 + i % 3} ms\n" for i in range(150)))
    out = tmp_path / "log.json"
    res = CliRunner().invoke(app, ["import-log", str(log), "--regex",
                                   r"took (?P<ms>\d+) ms", "--deadline-ms", "11.5",
                                   "--save", str(out)])
    assert res.exit_code == 0, res.output
    r = json.loads(out.read_text())
    assert r["iterations"] == 150 and r["deadline"]["misses"] == 50
    res = CliRunner().invoke(app, ["include"])
    assert res.output.strip() == INCLUDE_DIR


def test_cli_watch_needs_exactly_one_target():
    from typer.testing import CliRunner
    from edgelens.cli import app
    res = CliRunner().invoke(app, ["watch"])
    assert res.exit_code == 2
    res = CliRunner().invoke(app, ["watch", "--pid", "1", "--name", "x"])
    assert res.exit_code == 2


# ---------------------------------------------------------------- integrity

def test_lost_markers_turn_a_latency_pass_into_inconclusive_but_keep_fail():
    from edgelens.validate.engine import validate
    ms = 1_000_000
    spans = [_x("lstm", i * 10 * ms, i * 10 * ms + 2 * ms) for i in range(200)]
    clean = spans_to_result(spans, {"received": 200, "lost": 0}, deadline_ms=5,
                            attach_ids=False)
    assert validate(clean, {"p99_ms": 5})["verdict"] == "PASS"
    lossy = spans_to_result(spans, {"received": 200, "lost": 3, "lost_ratio": 0.015},
                            deadline_ms=5, attach_ids=False)
    v = validate(lossy, {"p99_ms": 5, "max_miss_ratio": 0})
    assert v["verdict"] == "INCONCLUSIVE"
    assert all(r["status"] == "NOT_MEASURED" for r in v["checks"])
    assert "MARKER_EVENTS_LOST" in v["checks"][0]["reason"]
    v = validate(lossy, {"p99_ms": 1})          # observed p99 2 ms > 1 ms: still a FAIL
    assert v["verdict"] == "FAIL"


def test_attach_without_markers_reports_latency_not_measured():
    from edgelens.diagnose.engine import diagnose
    from edgelens.core.result import build_result
    r = build_result(pipeline_desc={"name": "x", "pack": "custom", "config": {}, "stages": []},
                     stage_samples_ms={}, e2e_ms=[], mode="attach", pipeline_source="pid:1")
    codes = [i["code"] for i in diagnose(r)["measurement_integrity"]]
    assert "LATENCY_NOT_MEASURED" in codes

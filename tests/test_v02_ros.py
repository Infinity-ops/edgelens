"""v0.2: ROS 2 latency logic (S3) and the ROS bag example (S4)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from edgelens.core.clock import ClockMap
from edgelens.ros2 import latency as rl

ROOT = Path(__file__).resolve().parents[1]
MS = 1_000_000


class FixedClock(ClockMap):
    def __init__(self, offset_ns):
        self.offset_ns, self.uncertainty_ns, self.estimated_at_mono_ns = offset_ns, 1, 0


def _streams(n=200, period_ms=10, chain_ms=7.0, sensor_delay_ms=3.0, drop=(), offset=10**18):
    """IMU at 100 Hz stamped by the sensor; arrives sensor_delay later; the
    prediction (same stamp) arrives chain_ms after the IMU message."""
    src, dst = rl.TopicRecorder("/imu"), rl.TopicRecorder("/pred")
    for i in range(n):
        if i in drop:
            continue
        stamp_wall = offset + i * period_ms * MS
        arrival_mono = stamp_wall - offset + int(sensor_delay_ms * MS)
        src.add(arrival_mono, stamp_wall)
        if i % 10 == 0:
            dst.add(arrival_mono + int(chain_ms * MS), stamp_wall)
    return src, dst, FixedClock(offset)


def test_chain_latency_needs_no_clock_sync():
    src, dst, _ = _streams(chain_ms=7.0)
    lat, matched, unmatched = rl.chain_latency(src, dst)
    assert matched == 20 and unmatched == 0
    assert all(abs(v - 7.0) < 1e-9 for v in lat)


def test_data_age_same_machine_and_gap_detection():
    src, dst, ck = _streams(drop=range(50, 60))
    summary, chain, ages = rl.summarize(src, dst, same_machine=True, clock_map=ck)
    assert summary["data_age"]["valid"] and summary["data_age"]["uncertainty_ms"] == 0.0
    assert summary["data_age"]["from_ms"]["p50"] == pytest.approx(3.0)
    assert summary["data_age"]["at_output_ms"]["p50"] == pytest.approx(10.0)
    assert summary["from"]["gaps"] == 1 and summary["from"]["estimated_missing_messages"] == 10
    assert summary["from"]["rate_hz"] == pytest.approx(100.0)


def test_cross_machine_age_is_inconclusive_without_bounds():
    src, dst, ck = _streams()
    synced = {"available": True, "synchronized": True, "error_bound_s": 0.0004}
    s, _, _ = rl.summarize(src, dst, clock_map=ck, local_chrony={"available": False,
                                                                  "reason": "no chronyc"})
    assert s["data_age"]["valid"] is False
    s, _, _ = rl.summarize(src, dst, clock_map=ck, local_chrony=synced)
    assert s["data_age"]["valid"] is False and "--source-bound-ms" in s["data_age"]["uncertainty_basis"]
    s, _, _ = rl.summarize(src, dst, clock_map=ck, local_chrony=synced, source_bound_ms=0.6)
    assert s["data_age"]["uncertainty_ms"] == pytest.approx(1.0)


def test_results_validate_with_the_right_honesty():
    from edgelens.validate.engine import validate
    src, dst, ck = _streams()
    s, chain, ages = rl.summarize(src, dst, clock_map=ck, local_chrony={"available": False})
    r = rl.to_result(s, chain, ages, from_topic="/imu", to_topic="/pred", deadline_ms=20,
                     attach_ids=False)
    assert r["ros2"]["latency_basis"] == "chain" and r["iterations"] == 20
    assert validate(r, {"max_latency_ms": 20})["verdict"] == "PASS"   # chain: no sync needed
    # no matching stamps -> falls back to data age, which is cross-machine here
    dst2 = rl.TopicRecorder("/pred")
    for t in dst.recv_mono_ns:
        dst2.add(t, None)
    s, chain, ages = rl.summarize(src, dst2, clock_map=ck, local_chrony={"available": False})
    assert "copy the input stamp" in s["chain"]["note"]
    r = rl.to_result(s, chain, ages, from_topic="/imu", to_topic="/pred", deadline_ms=20,
                     attach_ids=False)
    v = validate(r, {"max_latency_ms": 20})
    assert r["ros2"]["latency_basis"] == "data_age"
    assert v["verdict"] == "INCONCLUSIVE"
    assert "CROSS_MACHINE_CLOCK_UNKNOWN" in v["checks"][0]["reason"]


def test_negative_data_age_is_flagged():
    from edgelens.core import integrity
    src, dst, _ = _streams(sensor_delay_ms=-50.0)                     # sensor clock ahead
    s, chain, ages = rl.summarize(src, None, clock_map=FixedClock(10**18), same_machine=True)
    r = rl.to_result(s, chain, ages, from_topic="/imu", to_topic=None, attach_ids=False)
    assert "NEGATIVE_DATA_AGE" in [i["code"] for i in integrity.check(r)]


def test_cli_without_ros_explains_how_to_source_it(monkeypatch):
    from typer.testing import CliRunner
    from edgelens.cli import app
    monkeypatch.setitem(sys.modules, "rclpy", None)
    res = CliRunner().invoke(app, ["ros2", "latency", "--from", "/imu", "--duration", "1"])
    assert res.exit_code == 1 and "source" in res.output.lower()


# ---------------------------------------------------------------- S4 bag example

rosbags = pytest.importorskip("rosbags")


@pytest.fixture(scope="module")
def dem2_bag(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
    from make_robot_bag import write_bag
    path = tmp_path_factory.mktemp("bag") / "dem2.bag"
    write_bag(str(path), seconds=6, imu_gap_at=3.0, battery=True)
    return path


def _run_example(*args):
    return subprocess.run([sys.executable, str(ROOT / "examples" / "rosbag_to_windows.py"),
                           *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=120)


def test_bag_merge_matches_sml_columns_and_finds_gaps(dem2_bag, tmp_path):
    out = tmp_path / "m.npy"
    res = _run_example(dem2_bag, "--out", out)
    assert res.returncode == 0, res.stderr
    info = json.loads(out.with_suffix(".json").read_text())
    a = np.load(out)
    assert a.shape == (len(info["columns"]), info["rows"])
    cols = info["columns"]
    for c in ("/imu:linear_acceleration.z", "/odom:pose.pose.position.x",
              "/odom:twist.twist.angular.z", "/cmd_vel_rc100:linear.x", "age_ms:/odom"):
        assert c in cols
    assert info["base_topic"] == "/imu"
    assert info["topics"]["/imu"]["gaps_over_2x_period"] == 1
    assert info["topics"]["/imu"]["longest_gap_ms"] == pytest.approx(210.0)
    # as-of join never looks into the future: ages are >= 0 and below one period + jitter
    age = a[cols.index("age_ms:/odom")]
    assert age.min() >= 0 and age.max() < 50.0 + 1e-6
    accel_z = a[cols.index("/imu:linear_acceleration.z")]
    assert np.allclose(accel_z, 9.81)


def test_bag_merge_header_time_generic_fields_and_errors(dem2_bag, tmp_path):
    out = tmp_path / "h.npy"
    res = _run_example(dem2_bag, "--out", out, "--time", "header",
                       "--topics", "/imu", "/odom", "/cmd_vel_rc100", "/battery")
    assert res.returncode == 0, res.stderr
    assert "no header.stamp on /cmd_vel_rc100" in res.stderr
    cols = json.loads(out.with_suffix(".json").read_text())["columns"]
    assert "/battery:voltage" in cols and "/battery:current" in cols
    res = _run_example(dem2_bag, "--out", out, "--topics", "/nope")
    assert res.returncode != 0 and "Available" in res.stderr


def test_dem2_replay_pipeline_runs_and_validates(dem2_bag, tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from edgelens.cli import app
    merged = tmp_path / "dem2_merged.npy"
    assert _run_example(dem2_bag, "--out", merged).returncode == 0
    monkeypatch.setenv("BAG_MERGED", str(merged))
    out = tmp_path / "dem2.json"
    res = CliRunner().invoke(app, ["benchmark", "--pipeline",
                                   str(ROOT / "examples" / "robot_bag_replay.py"),
                                   "--iterations", "120", "--deadline-ms", "3000",
                                   "--save", str(out)])
    assert res.exit_code == 0, res.output
    r = json.loads(out.read_text())
    assert list(r["stage_avg_ms"]) == ["build_window", "normalize", "lstm", "decide"]
    assert r["pipeline"]["config"]["hop_period_ms"] == 1000.0
    assert r["deadline"]["deadline_ms"] == 3000
    res = CliRunner().invoke(app, ["validate", str(out), "--p99-ms", "3000",
                                   "--max-miss-ratio", "0"])
    assert res.exit_code == 0, res.output

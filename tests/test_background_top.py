"""Background-load "top" list: who used the CPU during the window.

A laptop run listed `kworker/1:1H-kblockd` at 5.06 cores and missed the
test's busy loop: kernel workers rename themselves after the work they do,
the old code took a changed name for a reused PID and charged the worker
with its whole lifetime of CPU time. Identity is now the process start time.
"""
from edgelens.hardware.telemetry import TelemetryRecorder as T

W0, EL = 1000.0, 1.2          # window start (wall s), window length (s)


def used(before, now):
    return T._window_cpu(before, now, W0, EL)


def test_renamed_kernel_worker_counts_only_the_window():
    before = ("kworker/4:2-events", 300.0, 10.0)
    now = ("kworker/4:2-mm_percpu_wq", 300.05, 10.0)        # same start time
    assert abs(used(before, now) - 0.05) < 1e-9


def test_reused_pid_started_in_the_window_counts_fully():
    before = ("old", 300.0, 10.0)
    now = ("new", 0.4, W0 + 0.2)
    assert used(before, now) == 0.4


def test_reused_pid_started_before_the_window_is_skipped():
    before = ("old", 300.0, 10.0)
    now = ("other", 250.0, 500.0)                             # neither: unknown window share
    assert used(before, now) is None


def test_process_missing_from_first_snapshot_but_old_is_skipped():
    assert used(None, ("kworker/u16:2", 4000.0, 5.0)) is None


def test_new_process_in_the_window_counts_fully():
    assert used(None, ("hog", 1.1, W0 + 0.05)) == 1.1


def test_old_snapshots_without_create_time_fall_back_to_the_difference():
    assert abs(used(("p", 2.0), ("p", 2.5, None)) - 0.5) < 1e-9


def test_busy_loop_is_listed_at_one_core():
    import subprocess, sys, time
    hog = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        time.sleep(0.3)
        r = T(interval_s=0.2, use_tegrastats=False, use_power=False)
        r.start()
        time.sleep(1.2)
        bg = r.stop()["background"]
    finally:
        hog.kill()
        hog.wait()
    row = [r for r in bg["top"] if r["pid"] == hog.pid]     # listed, ~1 core;
    assert row and row[0]["cores"] > 0.6, bg["top"]          # first only on an idle box
    assert all(row["cores"] <= (__import__("psutil").cpu_count() or 1) for row in bg["top"])

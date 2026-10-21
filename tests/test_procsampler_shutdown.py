"""RSS of a process that exits by itself: leave out its shutdown.

Found on a laptop (and 3 runs in 12 here): `watch` recorded an ~80 MB
child as "17 -> 9 MB" because the last sample landed while Python's
finalizer returned its buffers to the OS. The reading is real, but it is
not the running process, and it bent the growth slope downwards.
"""
from edgelens.attach.procsampler import ProcessSampler


def _sampler(rss_mb, exited=True):
    s = ProcessSampler.__new__(ProcessSampler)
    s.pid, s.name, s.cmdline, s.children, s.interval_s = 1, "app", "app", True, 0.2
    s._costs = [0.1]
    s.exited_mono_ns = 123 if exited else None
    s.start_mono_ns = 0
    s.raw = [(int(i * 2e8), i * 0.1, int(v * 1e6), 1, 0, 0, 1)
             for i, v in enumerate([rss_mb[0]] + rss_mb)]
    return s


def test_shutdown_drop_is_left_out():
    p = _sampler([17.1, 25.1, 33.1, 41.1, 49.2, 57.2, 65.2, 73.2, 81.2, 9.1]).summary()
    assert p["rss_mb_start"] == 17.1 and p["rss_mb_end"] == 81.2 and p["rss_mb_peak"] == 81.2
    assert p["rss_mb_at_exit"] == 9.1
    assert p["rss_growth_mb_per_min"] > 2000          # ~40 MB/s, not bent down


def test_still_running_process_keeps_its_last_sample():
    p = _sampler([80.0, 81.0, 40.0], exited=False).summary()
    assert p["rss_mb_end"] == 40.0 and p["rss_mb_at_exit"] is None


def test_exit_without_a_drop_changes_nothing():
    p = _sampler([17.0, 25.0, 33.0, 41.0]).summary()
    assert p["rss_mb_end"] == 41.0 and p["rss_mb_at_exit"] is None

"""
edgelens.ros2.latency
-----------------------
Sensor-to-prediction latency from ROS 2 header timestamps — the measurement
logic, independent of rclpy (so it is unit-tested everywhere; the rclpy
adapter in edgelens.ros2.node only feeds it messages).

Two different numbers, kept apart on purpose:

  chain latency   arrival of the OUTPUT message - arrival of the INPUT
                  message it was computed from, matched by identical
                  header.stamp (DEM2 agreed that nodes keep the source
                  timestamp). Both arrivals are taken on THIS machine on
                  CLOCK_MONOTONIC: no clock sync needed, valid across any
                  number of nodes and machines in between.

  data age        arrival of a message on this machine - its header.stamp:
                  how old the sensor reading is when it gets here. The stamp
                  was taken by the SENSOR's machine, so this compares two
                  clocks. It is exact only on one machine; across machines
                  its uncertainty is the sum of both machines' chrony error
                  bounds. Without those bounds it is reported, but marked
                  INCONCLUSIVE — never as a confident number.

Per topic: rate, interval jitter, gaps (an interval > 2x the median period
counts as missing messages) — dropped sensor messages are as much a
predictive-maintenance problem as slow inference.
"""

import numpy as np

from ..core.clock import ClockMap, chrony_status

GAP_FACTOR = 2.0


class TopicRecorder:
    def __init__(self, name):
        self.name = name
        self.recv_mono_ns = []
        self.stamp_ns = []           # header.stamp in ROS time ns, or None

    def add(self, recv_mono_ns, stamp_ns):
        self.recv_mono_ns.append(int(recv_mono_ns))
        self.stamp_ns.append(None if stamp_ns is None else int(stamp_ns))

    def stats(self):
        n = len(self.recv_mono_ns)
        out = {"topic": self.name, "messages": n}
        if n < 2:
            return out
        dt = np.diff(np.asarray(self.recv_mono_ns, dtype=np.int64)) / 1e6
        med = float(np.median(dt))
        out.update(rate_hz=round(1000.0 / med, 3) if med > 0 else None,
                   period_ms_median=round(med, 3),
                   interval_jitter_ms=round(float(np.std(dt)), 3),
                   interval_ms_p99=round(float(np.percentile(dt, 99)), 3),
                   interval_ms_max=round(float(dt.max()), 3))
        if med > 0:
            gaps = dt[dt > GAP_FACTOR * med]
            out["gaps"] = int(len(gaps))
            out["estimated_missing_messages"] = int(sum(round(g / med) - 1 for g in gaps))
        stamps = [s for s in self.stamp_ns if s is not None]
        out["has_header_stamp"] = len(stamps) == n
        if stamps:
            sd = np.diff(np.asarray(stamps, dtype=np.int64)) / 1e6
            out["stamp_period_ms_median"] = round(float(np.median(sd)), 3) if len(sd) else None
            out["stamps_out_of_order"] = int((sd < 0).sum()) if len(sd) else 0
        return out


def chain_latency(src, dst):
    """Match dst messages to src messages with the same header.stamp.
    Returns (latencies_ms list in dst order, matched, unmatched)."""
    first_arrival = {}
    for t, s in zip(src.recv_mono_ns, src.stamp_ns):
        if s is not None and s not in first_arrival:
            first_arrival[s] = t
    lat, unmatched = [], 0
    for t, s in zip(dst.recv_mono_ns, dst.stamp_ns):
        t0 = first_arrival.get(s) if s is not None else None
        if t0 is None:
            unmatched += 1
            continue
        lat.append((t - t0) / 1e6)
    return lat, len(lat), unmatched


def data_age(rec, clock_map):
    """Age (ms) of each message at arrival: arrival (as wall/ROS time) - stamp."""
    out = []
    for t, s in zip(rec.recv_mono_ns, rec.stamp_ns):
        if s is not None:
            out.append((clock_map.mono_to_wall(t) - s) / 1e6)
    return out


def age_uncertainty(same_machine, source_bound_ms=None, local=None):
    """(bound_ms or None, explanation). None = cannot be known -> INCONCLUSIVE."""
    if same_machine:
        return 0.0, "sensor and EdgeLens on the same machine: one clock, no sync error"
    local = local if local is not None else chrony_status()
    if not local.get("available") or not local.get("synchronized"):
        return None, ("cross-machine data age without a synchronised local clock "
                      f"({local.get('reason') or 'chrony not synchronised'})")
    if source_bound_ms is None:
        return None, ("cross-machine data age: this machine's chrony bound is "
                      f"{local['error_bound_s'] * 1000:.3f} ms, but the sensor machine's "
                      "bound is unknown. Run `edgelens selftest` there and pass "
                      "--source-bound-ms")
    total = local["error_bound_s"] * 1000.0 + source_bound_ms
    return round(total, 3), (f"this machine ±{local['error_bound_s'] * 1000:.3f} ms + sensor "
                             f"machine ±{source_bound_ms} ms (chrony error bounds)")


def summarize(src, dst, same_machine=False, source_bound_ms=None, clock_map=None,
              local_chrony=None):
    """Everything `edgelens ros2 latency` reports, as one dict."""
    clock_map = clock_map or ClockMap()
    chain, matched, unmatched = chain_latency(src, dst) if dst is not None else ([], 0, 0)
    age_src = data_age(src, clock_map)
    age_dst = data_age(dst, clock_map) if dst is not None else []
    bound, why = age_uncertainty(same_machine, source_bound_ms, local_chrony)
    out = {
        "from": src.stats(),
        "to": dst.stats() if dst is not None else None,
        "chain": {"matched": matched, "unmatched": unmatched,
                  "method": "same header.stamp on input and output; both arrivals on this "
                            "machine's CLOCK_MONOTONIC (no clock sync needed)"},
        "data_age": {"uncertainty_ms": bound, "uncertainty_basis": why,
                     "valid": bound is not None},
        "clock": clock_map.describe(),
    }

    def pct(v, p):
        return round(float(np.percentile(v, p)), 3) if v else None

    for key, vals in (("from_ms", age_src), ("at_output_ms", age_dst)):
        out["data_age"][key] = {"n": len(vals), "p50": pct(vals, 50), "p99": pct(vals, 99),
                                "max": round(max(vals), 3) if vals else None}
    # Throughput window over exactly the messages that make up the latency
    # series (matched outputs for chain latency, stamped messages for age).
    if chain:
        stamps_in = {s for s in src.stamp_ns if s is not None}
        times = [t for t, s in zip(dst.recv_mono_ns, dst.stamp_ns) if s in stamps_in]
    else:
        rec = dst if age_dst else src
        times = [t for t, s in zip(rec.recv_mono_ns, rec.stamp_ns) if s is not None]
    if len(times) > 1 and times[-1] > times[0]:
        n = len(times)
        out["window_ms"] = (times[-1] - times[0]) / 1e6 * n / (n - 1)
    if dst is not None and dst.recv_mono_ns and not matched:
        out["chain"]["note"] = ("No output message carried an input's header.stamp, so the "
                                "chain can't be matched. The output node must copy the input "
                                "stamp into its header (DEM2 convention).")
    return out, chain, (age_dst or age_src)


def to_result(summary, chain_ms, age_ms, *, from_topic, to_topic, deadline_ms=None,
              telemetry_summary=None, telemetry_series=None, attach_ids=True):
    """Standard result document, so validate / compare / diagnose work."""
    from ..core.identity import attach_identity
    from ..core.result import build_result
    if chain_ms:
        basis, stage, series = "chain", f"chain {from_topic} -> {to_topic}", chain_ms
    else:
        basis, stage, series = "data_age", f"data age at {to_topic or from_topic}", age_ms
    summary = dict(summary, latency_basis=basis)
    result = build_result(
        pipeline_desc={"name": f"ros2:{from_topic}->{to_topic or from_topic}", "pack": "custom",
                       "config": {"from": from_topic, "to": to_topic},
                       "stages": [{"name": stage, "role": "other"}]},
        stage_samples_ms={stage: series}, e2e_ms=list(series), mode="ros2",
        pipeline_source=f"ros2:{from_topic}->{to_topic}", deadline_ms=deadline_ms,
        telemetry_summary=telemetry_summary, telemetry_series=telemetry_series,
        extra={"ros2": summary}, throughput_window_ms=summary.get("window_ms"))
    if attach_ids:
        attach_identity(result)
    return result

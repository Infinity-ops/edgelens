"""
Turn a ROS bag into a merged, time-aligned table that EdgeLens can replay.

Typical case: a mobile-robot recording (.bag / .db3 / .mcap) with /imu,
/odom and a velocity command topic, and a predictive-maintenance model that
gets the last X readings of all sources, merged by timestamp.

    pip install "edgelens[rosbag]"           # rosbags: reads ROS 1 .bag and
                                              # ROS 2 .db3/.mcap, no ROS needed
    python examples/rosbag_to_windows.py robot.bag --out merged.npy
    # optional: --topics /imu /odom /cmd_vel /battery  --base /imu

Writes:
    merged.npy              float64 [channels, rows]  (time on the last axis,
                            the layout edgelens.WindowSource replays)
    merged.json             column names, per-topic rates, gaps, data age

How the merge works (and why): one row per message of the BASE topic
(default: the fastest one). Every other topic contributes its LATEST message
at or before that row's time — an "as-of" join. That is exactly what an
online model sees at inference time: it can only use readings that have
already arrived. Rows before every topic has produced a message are
dropped. For each row and topic the AGE of the value used is kept
(`age_ms:<topic>` columns), because a model fed a 400 ms old odometry value
is not seeing what you think it is.

Columns per message type (the usual numeric fields; everything else falls back to all
numeric fields, so a battery topic just works):
    sensor_msgs/Imu          orientation.xyzw, angular_velocity.xyz,
                             linear_acceleration.xyz
    nav_msgs/Odometry        pose.position.xyz, pose.orientation.xyzw,
                             twist.linear.xyz, twist.angular.xyz
    geometry_msgs/Twist(+Stamped)   linear.xyz, angular.xyz

Time: the bag's receive timestamp by default (always present) — the only
mode in which "only already-arrived data" is guaranteed. --time header uses
header.stamp where messages have one (where nodes keep source timestamps), but
then rows are aligned on the SENSOR clocks: a stamp before the row time does
not prove the message had arrived, and topics without a header fall back to
bag time, mixing two clocks. Use it to study source timing, not to emulate
what an online model sees.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_TOPICS = ("/imu", "/odom", "/cmd_vel")

_FIELDS = {
    "sensor_msgs/msg/Imu": (
        "orientation.x", "orientation.y", "orientation.z", "orientation.w",
        "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"),
    "nav_msgs/msg/Odometry": (
        "pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z",
        "pose.pose.orientation.x", "pose.pose.orientation.y", "pose.pose.orientation.z",
        "pose.pose.orientation.w",
        "twist.twist.linear.x", "twist.twist.linear.y", "twist.twist.linear.z",
        "twist.twist.angular.x", "twist.twist.angular.y", "twist.twist.angular.z"),
    "geometry_msgs/msg/Twist": ("linear.x", "linear.y", "linear.z",
                                "angular.x", "angular.y", "angular.z"),
    "geometry_msgs/msg/TwistStamped": ("twist.linear.x", "twist.linear.y", "twist.linear.z",
                                       "twist.angular.x", "twist.angular.y",
                                       "twist.angular.z"),
}


def _get(msg, dotted):
    for part in dotted.split("."):
        msg = getattr(msg, part)
    return float(msg)


def numeric_fields(msg, prefix=""):
    """All scalar numeric fields of a message (fallback for unknown types)."""
    out = []
    names = getattr(msg, "__dataclass_fields__", None) or getattr(msg, "__slots__", ())
    for name in names:
        if name in ("header", "__msgtype__") or name.startswith("_"):
            continue
        v = getattr(msg, name)
        key = f"{prefix}{name}"
        if isinstance(v, bool):
            out.append(key)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out.append(key)
        elif hasattr(v, "__dataclass_fields__") or hasattr(v, "__slots__"):
            out.extend(numeric_fields(v, key + "."))
    return out


def _stamp_ns(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return None
    st = h.stamp
    return int(st.sec) * 1_000_000_000 + int(getattr(st, "nanosec", getattr(st, "nsec", 0)))


def read_topics(bag, topics, time_source="bag"):
    """{topic: (times_ns int64[N], values float64[N, F], field_names, msgtype)}"""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        sys.exit('This example needs rosbags >= 0.9.20:  pip install -U "edgelens[rosbag]"')
    out = {}
    paths = [Path(bag)]
    with AnyReader(paths, default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        available = {c.topic: c.msgtype for c in reader.connections}
        missing = [t for t in topics if t not in available]
        if missing:
            listing = "\n  ".join(f"{t}  ({m})" for t, m in sorted(available.items()))
            sys.exit(f"Topic(s) not in {bag}: {', '.join(missing)}\nAvailable:\n  {listing}")
        conns = [c for c in reader.connections if c.topic in topics]
        rows = {t: ([], []) for t in topics}
        fields = {}
        header_missing = set()
        for conn, t_bag, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            topic = conn.topic
            if topic not in fields:
                fields[topic] = _FIELDS.get(conn.msgtype) or tuple(numeric_fields(msg))
            t = t_bag
            if time_source == "header":
                st = _stamp_ns(msg)
                if st is None:
                    header_missing.add(topic)
                else:
                    t = st
            rows[topic][0].append(t)
            rows[topic][1].append([_get(msg, f) for f in fields[topic]])
        for topic in topics:
            ts, vals = rows[topic]
            if not ts:
                sys.exit(f"Topic {topic} has no messages.")
            order = np.argsort(np.asarray(ts, dtype=np.int64), kind="stable")
            out[topic] = (np.asarray(ts, dtype=np.int64)[order],
                          np.asarray(vals, dtype=np.float64)[order],
                          list(fields[topic]), available[topic])
        if header_missing:
            print(f"note: no header.stamp on {', '.join(sorted(header_missing))}; "
                  f"used the bag time for those topics", file=sys.stderr)
        if time_source == "header":
            print("warning: --time header aligns rows on sensor clocks; the merge no longer "
                  "guarantees 'only data that had already arrived'", file=sys.stderr)
    return out


def topic_stats(ts):
    dt = np.diff(ts) / 1e6
    if len(dt) == 0:
        return {"messages": int(len(ts))}
    med = float(np.median(dt))
    gaps = dt > 2.0 * med if med > 0 else np.zeros_like(dt, dtype=bool)
    return {"messages": int(len(ts)), "rate_hz": round(1000.0 / med, 2) if med else None,
            "period_ms_median": round(med, 3), "period_ms_p99": round(float(np.percentile(dt, 99)), 3),
            "gaps_over_2x_period": int(gaps.sum()),
            "longest_gap_ms": round(float(dt.max()), 3)}


def merge_asof(data, base=None):
    """As-of join onto the base topic's timeline. Returns (matrix [rows, cols],
    column names, times_ns, info)."""
    if base is None:
        base = max(data, key=lambda t: len(data[t][0]))
    bt = data[base][0]
    start = max(ts[0] for ts, _, _, _ in data.values())        # every topic present
    keep = bt >= start
    bt = bt[keep]
    cols, names = [], []
    names.append("time_s")
    cols.append(((bt - bt[0]) / 1e9)[:, None])
    for topic, (ts, vals, fields, _) in data.items():
        if topic == base:
            idx = np.nonzero(keep)[0]
            age = np.zeros(len(bt))
        else:
            idx = np.searchsorted(ts, bt, side="right") - 1         # latest at or before
            age = (bt - ts[idx]) / 1e6
        cols.append(vals[idx])
        names.extend(f"{topic}:{f}" for f in fields)
        cols.append(age[:, None])
        names.append(f"age_ms:{topic}")
    matrix = np.hstack(cols)
    info = {"base_topic": base, "rows": int(len(bt)),
            "rows_dropped_before_all_topics_present": int((~keep).sum()),
            "duration_s": round(float((bt[-1] - bt[0]) / 1e9), 3) if len(bt) else 0.0}
    ages = {t: {"p50_ms": round(float(np.percentile(matrix[:, names.index(f"age_ms:{t}")], 50)), 3),
                "max_ms": round(float(matrix[:, names.index(f"age_ms:{t}")].max()), 3)}
            for t in data if t != base}
    info["data_age_of_merged_values"] = ages
    return matrix, names, bt, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="ROS 1 .bag file, or ROS 2 bag directory / .mcap")
    ap.add_argument("--topics", nargs="+", default=list(DEFAULT_TOPICS))
    ap.add_argument("--base", default=None, help="timeline topic (default: the fastest)")
    ap.add_argument("--time", choices=("bag", "header"), default="bag")
    ap.add_argument("--out", default="merged.npy")
    args = ap.parse_args(argv)

    data = read_topics(args.bag, args.topics, args.time)
    matrix, names, _, info = merge_asof(data, args.base)
    info["topics"] = {t: dict(topic_stats(ts), msgtype=mt, fields=len(f))
                      for t, (ts, _, f, mt) in data.items()}
    info["columns"] = names
    info["time_source"] = args.time
    info["layout"] = "[channels, rows]: time on the last axis, for edgelens.WindowSource"
    out = Path(args.out)
    np.save(out, matrix.T.astype(np.float64))
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))

    print(f"{info['rows']} rows x {len(names)} columns over {info['duration_s']} s "
          f"(base {info['base_topic']}) -> {out}")
    for t, st in info["topics"].items():
        line = (f"  {t:<22} {st['messages']:>7} msgs  {st.get('rate_hz')} Hz  "
                f"gaps>2x period: {st.get('gaps_over_2x_period')}  "
                f"longest gap {st.get('longest_gap_ms')} ms")
        print(line)
    for t, a in info["data_age_of_merged_values"].items():
        print(f"  age of {t} values in merged rows: p50 {a['p50_ms']} ms, max {a['max_ms']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

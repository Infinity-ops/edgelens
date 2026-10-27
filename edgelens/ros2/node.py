"""
edgelens.ros2.node
--------------------
The rclpy side of `edgelens ros2 latency`: subscribe to two topics of ANY
message type that has a std_msgs/Header, timestamp each arrival on
CLOCK_MONOTONIC as the first thing in the callback, and hand everything to
edgelens.ros2.latency.

Needs a sourced ROS 2 installation (rclpy is not on PyPI):
    source /opt/ros/humble/setup.bash
    edgelens ros2 latency --from /sensor/imu --to /model/prediction --duration 60

QoS: best-effort, depth 10 — compatible with both reliable and best-effort
publishers, and it never slows a publisher down.

Arrival time includes rclpy's own delivery (deserialisation, executor
wake-up), typically well under a millisecond; it is the same for both
topics, so chain latency is unaffected to first order.
"""

import time

from ..core.clock import now_ns
from .latency import TopicRecorder


class Ros2Unavailable(RuntimeError):
    pass


def _import_rclpy():
    try:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        raise Ros2Unavailable(
            "rclpy is not importable. Source your ROS 2 installation first, e.g. "
            "`source /opt/ros/humble/setup.bash`, then run edgelens from that shell "
            f"(and from a Python that ROS 2 supports). ({e})") from e
    return rclpy, QoSProfile, ReliabilityPolicy, HistoryPolicy, get_message


def _stamp_ns(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return None
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def record(from_topic, to_topic=None, duration_s=30.0, discovery_timeout_s=10.0):
    """Subscribe and record arrivals. Returns (src, dst, info)."""
    rclpy, QoSProfile, Rel, Hist, get_message = _import_rclpy()
    rclpy.init()
    node = rclpy.create_node("edgelens_latency")
    try:
        wanted = [t for t in (from_topic, to_topic) if t]
        types = {}
        end = time.monotonic() + discovery_timeout_s
        while time.monotonic() < end and len(types) < len(wanted):
            for name, tlist in node.get_topic_names_and_types():
                if name in wanted and tlist:
                    types[name] = tlist[0]
            if len(types) < len(wanted):
                rclpy.spin_once(node, timeout_sec=0.2)
        missing = [t for t in wanted if t not in types]
        if missing:
            raise Ros2Unavailable(f"Topic(s) not found within {discovery_timeout_s:.0f} s: "
                                  f"{', '.join(missing)}. Check `ros2 topic list` and "
                                  f"ROS_DOMAIN_ID.")
        qos = QoSProfile(depth=10, reliability=Rel.BEST_EFFORT, history=Hist.KEEP_LAST)
        recs = {t: TopicRecorder(t) for t in wanted}
        no_header = set()

        def make_cb(topic):
            rec = recs[topic]

            def cb(msg):
                t = now_ns()                       # first: arrival time
                s = _stamp_ns(msg)
                if s is None:
                    no_header.add(topic)
                rec.add(t, s)
            return cb

        subs = [node.create_subscription(get_message(types[t]), t, make_cb(t), qos)
                for t in wanted]
        end = time.monotonic() + duration_s
        try:
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass
        del subs
        info = {"types": types, "topics_without_header": sorted(no_header),
                "duration_s": duration_s}
        return recs[from_topic], recs.get(to_topic), info
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

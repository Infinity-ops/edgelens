"""ROS 2 latency (experimental in v0.2): sensor-to-prediction timing from
header timestamps. edgelens.ros2.latency holds the logic (no ROS needed);
edgelens.ros2.node subscribes via rclpy."""

from .latency import TopicRecorder, chain_latency, data_age, summarize

__all__ = ["TopicRecorder", "chain_latency", "data_age", "summarize"]

"""Write a small synthetic ROS 1 bag shaped like the DEM2 Infineon recording:
/imu (sensor_msgs/Imu, 100 Hz), /odom (nav_msgs/Odometry, 20 Hz),
/cmd_vel_rc100 (geometry_msgs/Twist, 10 Hz, no header), and an optional
/battery (sensor_msgs/BatteryState) for the generic-fields path.

    python tests/fixtures/make_dem2_bag.py out.bag --seconds 5 [--imu-gap 1.0]

--imu-gap drops IMU messages for 200 ms starting at that second, so tests can
check that gaps are detected. Needs:  pip install "edgelens[rosbag]"
"""

import argparse
import math

import numpy as np


def write_bag(path, seconds=5.0, imu_gap_at=None, battery=False, t0_ns=1_700_000_000_000_000_000):
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    Header, Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
    Quat, Vec3 = T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Vector3"]
    Imu, Odom, Twist = T["sensor_msgs/msg/Imu"], T["nav_msgs/msg/Odometry"], \
        T["geometry_msgs/msg/Twist"]
    Pose, PoseCov = T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/PoseWithCovariance"]
    TwistCov, Point = T["geometry_msgs/msg/TwistWithCovariance"], T["geometry_msgs/msg/Point"]
    cov9, cov36 = np.zeros(9), np.zeros(36)

    def header(t_ns, seq, frame="base_link"):
        return Header(seq=seq, stamp=Time(sec=t_ns // 1_000_000_000,
                                          nanosec=t_ns % 1_000_000_000), frame_id=frame)

    events = []
    for i in range(int(seconds * 100)):
        t = t0_ns + i * 10_000_000
        if imu_gap_at is not None and imu_gap_at <= i / 100.0 < imu_gap_at + 0.2:
            continue
        a = math.sin(i / 10.0)
        events.append((t, "/imu", Imu(
            header=header(t - 2_000_000, i),            # stamped 2 ms before arrival
            orientation=Quat(x=0.0, y=0.0, z=math.sin(i / 200.0), w=math.cos(i / 200.0)),
            orientation_covariance=cov9,
            angular_velocity=Vec3(x=0.01 * a, y=0.0, z=0.1),
            angular_velocity_covariance=cov9,
            linear_acceleration=Vec3(x=a, y=0.1 * a, z=9.81),
            linear_acceleration_covariance=cov9), "sensor_msgs/msg/Imu"))
    for i in range(int(seconds * 20)):
        t = t0_ns + 3_000_000 + i * 50_000_000
        events.append((t, "/odom", Odom(
            header=header(t - 5_000_000, i, "odom"), child_frame_id="base_link",
            pose=PoseCov(pose=Pose(position=Point(x=0.05 * i, y=0.0, z=0.0),
                                   orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0)),
                         covariance=cov36),
            twist=TwistCov(twist=Twist(linear=Vec3(x=1.0, y=0.0, z=0.0),
                                       angular=Vec3(x=0.0, y=0.0, z=0.1)),
                           covariance=cov36)), "nav_msgs/msg/Odometry"))
    for i in range(int(seconds * 10)):
        t = t0_ns + 7_000_000 + i * 100_000_000
        events.append((t, "/cmd_vel_rc100", Twist(linear=Vec3(x=1.0, y=0.0, z=0.0),
                                                  angular=Vec3(x=0.0, y=0.0, z=0.1)),
                       "geometry_msgs/msg/Twist"))
    if battery:
        Bat = T["sensor_msgs/msg/BatteryState"]
        for i in range(int(seconds)):
            t = t0_ns + 11_000_000 + i * 1_000_000_000
            events.append((t, "/battery", Bat(
                header=header(t, i), voltage=24.0 - 0.01 * i, temperature=30.0,
                current=-2.5, charge=10.0, capacity=20.0, design_capacity=20.0,
                percentage=0.9, power_supply_status=2, power_supply_health=1,
                power_supply_technology=2, present=True, cell_voltage=np.zeros(0),
                cell_temperature=np.zeros(0), location="", serial_number=""),
                "sensor_msgs/msg/BatteryState"))
    events.sort(key=lambda e: e[0])
    with Writer(path) as w:
        conns = {}
        for t, topic, msg, mt in events:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, mt, typestore=ts)
            w.write(conns[topic], t, ts.serialize_ros1(msg, mt))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--imu-gap", type=float, default=None)
    ap.add_argument("--battery", action="store_true")
    a = ap.parse_args()
    write_bag(a.out, a.seconds, a.imu_gap, a.battery)
    print(f"wrote {a.out}")

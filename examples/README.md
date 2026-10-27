# Examples

Start from the one closest to your system; every example runs on a laptop
(synthetic data or a stand-in model) and unchanged on a Jetson.

| You have | Start with | What it shows |
| --- | --- | --- |
| nothing yet: just want to see it work | [quickstart/](quickstart/) | a NumPy sensor pipeline measured, diagnosed and validated in 2 minutes, with the expected output and a result checker |
| an ONNX image model and a folder of images | `edgelens benchmark --model m.onnx --inputs val/ --labels subfolders --image-prep imagenet` ([usage.md](../docs/usage.md#real-inputs-accuracy-and-threads)) | latency per stage, accuracy (top-1 / top-5) on the board, `sweep` across CPU / CUDA / TensorRT / FP16 |
| an ONNX model and prepared arrays (.npy / .bin) | `edgelens benchmark --model m.onnx --inputs x.npy --labels y.npy` | the same for time-series, audio or tabular models: RMSE or accuracy |
| a Python pipeline (several stages) | [timeseries_vibration.py](timeseries_vibration.py), [predictive_maintenance.py](predictive_maintenance.py) | harness mode: EdgeLens drives your stages and finds the bottleneck |
| your own loop you don't want to change | [observer_vision_loop.py](observer_vision_loop.py) | observer mode: three lines around your existing code |
| a C/C++ or separate Python application | [markers/](markers/) | `edgelens record -- ./app`: spans from any process via a header / `edgelens.mark` |
| a ROS bag from a robot | [rosbag_to_windows.py](rosbag_to_windows.py) + [robot_bag_replay.py](robot_bag_replay.py) | merge topics as an online model sees them, replay at the real rate |
| a recording with known events | [stream/](stream/) | `replay` / `detect` / `perturb`: time to detect, false alarms, sensor faults |
| a long-running deployment | [soak_leak_demo.py](soak_leak_demo.py) | `soak`: drift, throttling, memory growth (a known leak to check against) |
| requirements to gate in CI | [ci-jetson/](ci-jetson/) | an assumption contract and a GitHub Actions job on a self-hosted Jetson |

What was measured on real boards, with the numbers:
[docs/jetson.md](../docs/jetson.md).

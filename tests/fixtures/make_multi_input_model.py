"""Regenerates multi_input_signal.onnx: a sensor-style model with TWO inputs —
a float vibration window [1, 8, T] with a dynamic length T, and an int64
RPM scalar [1, 1]. Needs the `fixtures` extra (pip install onnx)."""
import os

import onnx
from onnx import TensorProto, helper

vib = helper.make_tensor_value_info("vib", TensorProto.FLOAT, [1, 8, "T"])
rpm = helper.make_tensor_value_info("rpm", TensorProto.INT64, [1, 1])
score = helper.make_tensor_value_info("score", TensorProto.FLOAT, [])
rpm_out = helper.make_tensor_value_info("rpm_out", TensorProto.INT64, [1, 1])
graph = helper.make_graph(
    [helper.make_node("ReduceMean", ["vib"], ["score"], keepdims=0),
     helper.make_node("Identity", ["rpm"], ["rpm_out"])],
    "multi_input_signal", [vib, rpm], [score, rpm_out])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(os.path.dirname(__file__), "multi_input_signal.onnx"))

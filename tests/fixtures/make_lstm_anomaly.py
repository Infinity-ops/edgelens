"""Regenerates lstm_anomaly.onnx — a small, untrained but architecturally
real anomaly model: a window of 50 time steps
x 16 sensor features -> LSTM(32) -> last hidden state -> Gemm -> Sigmoid ->
anomaly score in [0, 1]. Used by `edgelens sweep` tests and docs.

    pip install onnx        # dev-only, see the `fixtures` extra
    cd tests/fixtures && python make_lstm_anomaly.py
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

STEPS, FEATURES, HIDDEN = 50, 16, 32
rng = np.random.default_rng(0)

X = helper.make_tensor_value_info("window", TensorProto.FLOAT, [1, STEPS, FEATURES])
Y = helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])

W = (rng.standard_normal((1, 4 * HIDDEN, FEATURES)) * 0.2).astype(np.float32)
R = (rng.standard_normal((1, 4 * HIDDEN, HIDDEN)) * 0.2).astype(np.float32)
B = np.zeros((1, 8 * HIDDEN), np.float32)
Wout = (rng.standard_normal((HIDDEN, 1)) * 0.5).astype(np.float32)
Bout = np.zeros((1,), np.float32)

inits = [numpy_helper.from_array(W, "W"), numpy_helper.from_array(R, "R"),
         numpy_helper.from_array(B, "B"), numpy_helper.from_array(Wout, "Wout"),
         numpy_helper.from_array(Bout, "Bout"),
         numpy_helper.from_array(np.array([0], np.int64), "axes0")]
nodes = [
    helper.make_node("Transpose", ["window"], ["x_tbf"], perm=[1, 0, 2]),     # [T, 1, F]
    helper.make_node("LSTM", ["x_tbf", "W", "R", "B"], ["Y", "Y_h"], hidden_size=HIDDEN),
    helper.make_node("Squeeze", ["Y_h", "axes0"], ["h"]),                     # [1, H]
    helper.make_node("Gemm", ["h", "Wout", "Bout"], ["logit"]),
    helper.make_node("Sigmoid", ["logit"], ["score"]),
]
graph = helper.make_graph(nodes, "lstm_anomaly", [X], [Y], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                          producer_name="edgelens-fixtures")
model.ir_version = 7            # loadable by older onnxruntime (Jetson wheels)
onnx.checker.check_model(model)
onnx.save(model, "lstm_anomaly.onnx")
print("wrote lstm_anomaly.onnx")

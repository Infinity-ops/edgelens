"""Regenerates tiny_model.onnx. Requires `pip install onnx` (a dev-only
dependency for building this fixture, not an edgelens runtime dependency).
Run from the tests/fixtures/ directory."""
import onnx
from onnx import helper, TensorProto
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

flat = helper.make_node('Flatten', ['input'], ['flat'], axis=1)
W = np.random.randn(3072, 10).astype(np.float32) * 0.01
B = np.random.randn(10).astype(np.float32) * 0.01
W_init = helper.make_tensor('W', TensorProto.FLOAT, W.shape, W.flatten().tolist())
B_init = helper.make_tensor('B', TensorProto.FLOAT, B.shape, B.flatten().tolist())
matmul = helper.make_node('MatMul', ['flat', 'W'], ['matmul_out'])
add = helper.make_node('Add', ['matmul_out', 'B'], ['add_out'])
relu = helper.make_node('Relu', ['add_out'], ['output'])

graph = helper.make_graph([flat, matmul, add, relu], 'tiny-test-model', [X], [Y],
                           initializer=[W_init, B_init])
model = helper.make_model(graph, producer_name='edgelens-test',
                           opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, 'tiny_model.onnx')
print('saved tiny_model.onnx')

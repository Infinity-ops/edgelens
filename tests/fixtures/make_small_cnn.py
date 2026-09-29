"""Regenerates small_cnn.onnx — a small, untrained but architecturally
real CNN (Conv/ReLU/MaxPool x3 + GlobalAveragePool + FC), used for the
README's example output and as a bundled "try it now" model. Requires
`pip install onnx` (dev-only, see the `fixtures` extra in pyproject.toml).
Run from the tests/fixtures/ directory."""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

np.random.seed(0)

def conv_block(name, in_ch, out_ch):
    w = np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32) * 0.05
    b = np.zeros(out_ch, dtype=np.float32)
    w_init = numpy_helper.from_array(w, name=f'{name}_w')
    b_init = numpy_helper.from_array(b, name=f'{name}_b')
    conv = helper.make_node('Conv', [f'{name}_in', f'{name}_w', f'{name}_b'],
                             [f'{name}_conv'], pads=[1,1,1,1], strides=[1,1])
    relu = helper.make_node('Relu', [f'{name}_conv'], [f'{name}_relu'])
    pool = helper.make_node('MaxPool', [f'{name}_relu'], [f'{name}_out'],
                             kernel_shape=[2,2], strides=[2,2])
    return [conv, relu, pool], [w_init, b_init]

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

nodes, inits = [], []
n1, i1 = conv_block('b1', 3, 16); nodes += n1; inits += i1
n2, i2 = conv_block('b2', 16, 32); nodes += n2; inits += i2
n3, i3 = conv_block('b3', 32, 64); nodes += n3; inits += i3
nodes[0].input[0] = 'input'
nodes[3].input[0] = 'b1_out'
nodes[6].input[0] = 'b2_out'

# Global average pool (as real efficient CNNs do) before the final FC
# layer — avoids a ~50M-parameter FC from a naive flatten.
gap = helper.make_node('GlobalAveragePool', ['b3_out'], ['gap_out'])
flat = helper.make_node('Flatten', ['gap_out'], ['flat'], axis=1)

W = np.random.randn(64, 10).astype(np.float32) * 0.01
B = np.random.randn(10).astype(np.float32) * 0.01
w_fc = numpy_helper.from_array(W, name='fc_w')
b_fc = numpy_helper.from_array(B, name='fc_b')
matmul = helper.make_node('MatMul', ['flat', 'fc_w'], ['matmul_out'])
add = helper.make_node('Add', ['matmul_out', 'fc_b'], ['output'])

graph = helper.make_graph(
    nodes + [gap, flat, matmul, add],
    'small-cnn-for-edgelens-demo',
    [X], [Y],
    initializer=inits + [w_fc, b_fc],
)
model = helper.make_model(graph, producer_name='edgelens-demo',
                           opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, 'small_cnn.onnx')
print('saved small_cnn.onnx')

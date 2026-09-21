"""Executable reusable blocks shared by Tinygrad examples.

YOLO blocks match examples/yolov8.py's inference semantics. BatchNorm state is
explicit and frozen; these are not training-mode BN implementations.
"""
from ..ir import Graph, Node, TensorType
from ..compose import inline
from .vision import activation_graph, conv2d_graph
from .nn import linear_graph


def residual_graph():
    x = TensorType(('...', 'D'))
    g = Graph('RESIDUAL', {'x': x, 'branch': x}, ['out'], [Node('out', 'ADD', ['x', 'branch'], output_type=x)])
    g.validate()
    return g


def batchnorm_inference_graph(eps=1e-5):
    g = Graph('BATCHNORM_INFERENCE', {'x': TensorType(('N', 'C', 'H', 'W')),
              **{k: TensorType(('C',)) for k in ('mean', 'variance', 'weight', 'bias')}}, ['out'],
              metadata={'semantics': 'NCHW frozen running statistics; no running-state update'})
    for key in ('mean', 'variance', 'weight', 'bias'):
        g.add(Node(key+'_4d', 'RESHAPE', [key], {'shape': [1, -1, 1, 1]}))
    g.add(Node('eps', 'CONST', attrs={'value': eps}))
    g.add(Node('den', 'ADD', ['variance_4d', 'eps']))
    g.add(Node('inv', 'RSQRT', ['den']))
    g.add(Node('centered', 'SUB', ['x', 'mean_4d']))
    g.add(Node('normalized', 'MUL', ['centered', 'inv']))
    g.add(Node('scaled', 'MUL', ['normalized', 'weight_4d']))
    g.add(Node('out', 'ADD', ['scaled', 'bias_4d']))
    g.validate()
    return g


def pool2d_graph(kind='max', kernel_size=2, stride=None, padding=0):
    if kind not in ('max', 'avg'):
        raise ValueError('pool kind must be max or avg')
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    name = kind.upper() + '_POOL2D'
    g = Graph(name, {'x': TensorType(('N', 'C', 'H', 'W'))}, ['out'])
    g.add(Node('out', name, ['x'], {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}))
    g.validate()
    return g


def _module(g, child, prefix, x):
    """Lift every child parameter/state to an explicit parent input."""
    bindings = {'x': x}
    for key, typ in child.inputs.items():
        if key != 'x':
            name = prefix + '.' + key
            if name in g.inputs:
                raise ValueError(f'duplicate parameter: {name}')
            g.inputs[name] = typ
            bindings[key] = name
    return inline(g, child, prefix, bindings)[0]


def mlp_graph(din=8, hidden=16, dout=8, activation='gelu', bias=True):
    g = Graph('DENSE_MLP', {'x': TensorType(('B', 'T', din))}, ['out'])
    h = _module(g, linear_graph(din, hidden, bias), 'fc1', 'x')
    h = inline(g, activation_graph(activation), 'activation', {'x': h})[0]
    out = _module(g, linear_graph(hidden, dout, bias), 'fc2', h)
    g.add(Node('out', 'IDENTITY', [out]))
    g.validate()
    return g


def swiglu_graph(din=8, hidden=16, dout=8):
    g = Graph('SWIGLU', {'x': TensorType(('B', 'T', din))}, ['out'],
              metadata={'semantics': 'bias-free down(silu(gate(x)) * up(x))'})
    gate = _module(g, linear_graph(din, hidden, False), 'gate', 'x')
    gate = inline(g, activation_graph('silu'), 'activation', {'x': gate})[0]
    up = _module(g, linear_graph(din, hidden, False), 'up', 'x')
    mixed = g.add(Node('mixed', 'MUL', [gate, up]))
    out = _module(g, linear_graph(hidden, dout, False), 'down', mixed)
    g.add(Node('out', 'IDENTITY', [out]))
    g.validate()
    return g


def yolov8_conv_graph(c1=8, c2=8, kernel_size=3, stride=1, groups=1, dilation=1, padding=None):
    if c1 <= 0 or c2 <= 0 or groups <= 0 or c1 % groups or c2 % groups:
        raise ValueError('channels must be positive multiples of groups')
    if padding is None:
        padding = (dilation * (kernel_size - 1) + 1) // 2
    g = Graph('YOLOV8_CONV', {'x': TensorType(('N', c1, 'H', 'W'))}, ['out'],
              metadata={'source': 'examples/yolov8.py::Conv_Block', 'mode': 'inference'})
    conv = conv2d_graph(stride, padding, dilation, groups)
    conv.inputs['weight'] = TensorType((c2, c1//groups, kernel_size, kernel_size))
    x = _module(g, conv, 'conv', 'x')
    bn = batchnorm_inference_graph(0.001)
    for key in ('mean', 'variance', 'weight', 'bias'):
        bn.inputs[key] = TensorType((c2,))
    x = _module(g, bn, 'bn', x)
    x = inline(g, activation_graph('silu'), 'activation', {'x': x})[0]
    g.add(Node('out', 'IDENTITY', [x]))
    g.validate()
    return g


def yolov8_bottleneck_graph(c1=8, c2=8, shortcut=True, groups=1, expansion=0.5):
    hidden = int(c2 * expansion)
    if hidden < 1:
        raise ValueError('hidden channels must be positive')
    g = Graph('YOLOV8_BOTTLENECK', {'x': TensorType(('N', c1, 'H', 'W'))}, ['out'],
              metadata={'source': 'examples/yolov8.py::Bottleneck', 'mode': 'inference', 'kernels': [3, 3]})
    x = _module(g, yolov8_conv_graph(c1, hidden, 3), 'cv1', 'x')
    x = _module(g, yolov8_conv_graph(hidden, c2, 3, groups=groups), 'cv2', x)
    g.add(Node('out', 'ADD' if shortcut and c1 == c2 else 'IDENTITY', ['x', x] if shortcut and c1 == c2 else [x]))
    g.validate()
    return g


def yolov8_c2f_graph(c1=8, c2=8, repeats=1, shortcut=False, groups=1, expansion=0.5):
    hidden = int(c2 * expansion)
    if not isinstance(repeats, int) or repeats < 0 or hidden < 1:
        raise ValueError('nonnegative integer repeats and positive hidden channels required')
    g = Graph('YOLOV8_C2F', {'x': TensorType(('N', c1, 'H', 'W'))}, ['out'],
              metadata={'source': 'examples/yolov8.py::C2f', 'mode': 'inference', 'repeats': repeats})
    x = _module(g, yolov8_conv_graph(c1, 2*hidden, 1), 'cv1', 'x')
    left = g.add(Node('left', 'SLICE', [x], {'axis': 1, 'start': 0, 'stop': hidden}))
    right = g.add(Node('right', 'SLICE', [x], {'axis': 1, 'start': hidden, 'stop': 2*hidden}))
    parts = [left, right]
    for i in range(repeats):
        parts.append(_module(g, yolov8_bottleneck_graph(hidden, hidden, shortcut, groups, 1.0), f'block{i}', parts[-1]))
    x = g.add(Node('concat', 'CONCAT', parts, {'axis': 1}))
    x = _module(g, yolov8_conv_graph((2+repeats)*hidden, c2, 1), 'cv2', x)
    g.add(Node('out', 'IDENTITY', [x]))
    g.validate()
    return g


def yolov8_sppf_graph(c1=8, c2=8, kernel_size=5):
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError('SPPF requires a positive odd pooling kernel')
    hidden = c1 // 2
    g = Graph('YOLOV8_SPPF', {'x': TensorType(('N', c1, 'H', 'W'))}, ['out'],
              metadata={'source': 'examples/yolov8.py::SPPF', 'mode': 'inference'})
    x = _module(g, yolov8_conv_graph(c1, hidden, 1), 'cv1', 'x')
    parts = [x]
    for i in range(3):
        parts.append(inline(g, pool2d_graph('max', kernel_size, 1, kernel_size//2), f'pool{i}', {'x': parts[-1]})[0])
    x = g.add(Node('concat', 'CONCAT', parts, {'axis': 1}))
    x = _module(g, yolov8_conv_graph(4*hidden, c2, 1), 'cv2', x)
    g.add(Node('out', 'IDENTITY', [x]))
    g.validate()
    return g

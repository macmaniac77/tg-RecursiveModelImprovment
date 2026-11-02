"""Small JEPA-style latent prediction graphs, not an I-JEPA checkpoint port."""
import math

from ..ir import Graph, Node, TensorType
from .reusable import _module, mlp_graph


def jepa_graph(input_dim=8, hidden_dim=16, latent_dim=8):
    """Paired vectors -> context/target MLPs -> predictor -> latent MSE.

    Views must already be aligned by the caller. No masking, image tokenizer,
    positional conditioning, optimizer or hidden parameter state is implied.
    """
    if any(type(d) is not int or d <= 0 for d in (input_dim, hidden_dim, latent_dim)):
        raise ValueError('positive integer dimensions required')
    view = TensorType(('B', input_dim))
    latent = TensorType(('B', latent_dim))
    g = Graph('JEPA', {'context_view': view, 'target_view': view},
              ['prediction', 'target', 'loss'], metadata={
                  'version': 1, 'family': 'JEPA-style vector example',
                  'semantics': 'latent MSE; stopped target branch; external EMA after optimizer',
                  'reference': 'https://github.com/facebookresearch/ijepa',
                  'limitations': 'MLPs and prepared vector views; not I-JEPA/V-JEPA equivalence',
              })
    encoder = mlp_graph(input_dim, hidden_dim, latent_dim, 'relu')
    # Reuse the primitive MLP definition with a precise vector input contract.
    encoder.inputs['x'] = view
    context = _module(g, encoder, 'context_encoder', 'context_view')
    target = _module(g, encoder, 'target_encoder', 'target_view')
    g.add(Node('target', 'STOP_GRADIENT', [target], output_type=latent))
    predictor = mlp_graph(latent_dim, hidden_dim, latent_dim, 'relu')
    predictor.inputs['x'] = latent
    prediction = _module(g, predictor, 'predictor', context)
    g.add(Node('prediction', 'IDENTITY', [prediction], output_type=latent))
    g.add(Node('error', 'SUB', ['prediction', 'target']))
    g.add(Node('squared_error', 'MUL', ['error', 'error']))
    g.add(Node('loss', 'REDUCE_MEAN', ['squared_error'], {'axis': [0, 1]},
               output_type=TensorType(())))
    g.validate()
    return g


def ema_update_graph(decay=0.99):
    """Return detached target_next = decay * target + (1-decay) * online.

    This is a pure value graph for one parameter tensor; callers explicitly
    assign its output to target state after the online optimizer step.
    """
    if not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError('EMA decay must be finite and in [0, 1]')
    typ = TensorType(('...',))
    g = Graph('EMA_UPDATE', {'target': typ, 'online': typ}, ['target_next'],
              metadata={'version': 1, 'state': 'explicit; assignment performed by caller', 'decay': decay})
    g.add(Node('decay', 'CONST', attrs={'value': float(decay)}))
    g.add(Node('complement', 'CONST', attrs={'value': 1.0-decay}))
    g.add(Node('old', 'MUL', ['target', 'decay']))
    g.add(Node('new', 'MUL', ['online', 'complement']))
    g.add(Node('blend', 'ADD', ['old', 'new']))
    g.add(Node('target_next', 'STOP_GRADIENT', ['blend'], output_type=typ))
    g.validate()
    return g

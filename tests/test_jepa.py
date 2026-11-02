"""JEPA forward, gradient-boundary and explicit EMA-state equivalence."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('tinygrad')
from tinygrad import Tensor

from tgaoi.aoi import ema_update_graph, jepa_graph
from tgaoi.audit import audit_graph, catalog
from tgaoi.backends import TinygradBackend
from tgaoi.ir import Graph


def fixture():
    graph = jepa_graph(4, 5, 2)
    rng = np.random.default_rng(29)
    feeds = {key: rng.normal(0, 0.3, size=(3, 4) if key.endswith('_view') else typ.shape).astype('float32')
             for key, typ in graph.inputs.items()}
    return graph, feeds


def reference(feeds):
    def mlp(x, prefix):
        return np.maximum(x @ feeds[prefix+'.fc1.weight'].T + feeds[prefix+'.fc1.bias'], 0) @ feeds[prefix+'.fc2.weight'].T + feeds[prefix+'.fc2.bias']
    context = mlp(feeds['context_view'], 'context_encoder')
    target = mlp(feeds['target_view'], 'target_encoder')
    prediction = mlp(context, 'predictor')
    return {'prediction': prediction, 'target': target, 'loss': ((prediction-target)**2).mean()}


def test_jepa_roundtrip_and_numpy_equivalence():
    graph, feeds = fixture()
    graph = Graph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert audit_graph(graph)['status'] == 'lowerable'
    assert 'JEPA' in catalog() and 'EMA_UPDATE' in catalog()
    assert len(graph.metadata['aoi_instances']) == 3
    assert not any(n.op.startswith('AOI::') for n in graph.nodes)
    actual = TinygradBackend().run(graph, feeds)
    for key, expected in reference(feeds).items():
        np.testing.assert_allclose(actual[key].numpy(), expected, atol=2e-6, rtol=2e-5)


def test_jepa_gradients_match_finite_differences_and_stop_target():
    graph, values = fixture()
    feeds = {key: Tensor(value) for key, value in values.items()}
    TinygradBackend().run(graph, feeds)['loss'].backward()
    # Even deliberately trainable target feeds cannot receive loss gradients.
    for key, tensor in feeds.items():
        if key.startswith('target_'):
            assert tensor.grad is None or not np.any(tensor.grad.numpy())
    for key in ('context_view', 'context_encoder.fc2.bias', 'predictor.fc2.weight'):
        expected = np.zeros_like(values[key])
        for index in np.ndindex(expected.shape):
            delta = np.zeros_like(expected)
            delta[index] = 0.001
            plus = reference({**values, key: values[key]+delta})['loss']
            minus = reference({**values, key: values[key]-delta})['loss']
            expected[index] = (plus-minus)/0.002
        assert np.abs(expected).max() > 1e-5
        np.testing.assert_allclose(feeds[key].grad.numpy(), expected, atol=1e-5, rtol=0.005)


@pytest.mark.parametrize('decay', [0., 0.8, 1.])
def test_ema_detaches_and_returns_state_without_mutation(decay):
    graph = Graph.from_dict(json.loads(json.dumps(ema_update_graph(decay).to_dict())))
    old = np.array([[1., 2.], [-3., 4.]], dtype='float32')
    online = np.array([[5., -2.], [1., 0.]], dtype='float32')
    feeds = {'target': Tensor(old), 'online': Tensor(online)}
    result = TinygradBackend().run(graph, feeds)['target_next']
    np.testing.assert_allclose(result.numpy(), decay*old+(1-decay)*online, atol=1e-6)
    result.sum().backward()
    for tensor in feeds.values():
        assert tensor.grad is None or not np.any(tensor.grad.numpy())
    np.testing.assert_array_equal(feeds['target'].numpy(), old)
    np.testing.assert_array_equal(feeds['online'].numpy(), online)


@pytest.mark.parametrize('kwargs', [{'input_dim': 0}, {'hidden_dim': -1}, {'latent_dim': 1.5}])
def test_invalid_jepa_dimensions(kwargs):
    with pytest.raises(ValueError, match='positive integer'):
        jepa_graph(**kwargs)


@pytest.mark.parametrize('decay', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_ema_decay(decay):
    with pytest.raises(ValueError, match='EMA decay'):
        ema_update_graph(decay)


def test_example_trains_and_exports_graph(tmp_path):
    spec = importlib.util.spec_from_file_location('jepa_example', Path(__file__).parents[1]/'examples/jepa.py')
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    path = tmp_path/'jepa.arch.json'
    report = example.main(['--steps', '3', '--batch-size', '4', '--graph', str(path)])
    assert np.isfinite(report['last_pre_update_loss'])
    assert report['last_pre_update_loss'] < report['initial_loss']
    assert Graph.from_dict(json.loads(path.read_text())).name == 'JEPA'

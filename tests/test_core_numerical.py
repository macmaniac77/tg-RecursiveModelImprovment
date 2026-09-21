"""Numerical checks of the five directly lowerable default AOIs."""
import numpy as np
import pytest
pytest.importorskip('tinygrad')
from tgaoi.aoi import matmul_graph, softmax_graph, rmsnorm_graph, linear_graph, adam_graph
from tgaoi.backends import TinygradBackend


def run(g, feeds):
    return {k: v.numpy() for k, v in TinygradBackend().run(g, feeds).items()}


def test_matmul_and_linear():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3, 4)).astype('float32')
    w = rng.normal(size=(2, 4)).astype('float32')
    b = np.array([0.5, -0.2], dtype='float32')
    np.testing.assert_allclose(run(matmul_graph(3, 4, 2), {'a': x, 'b': w.T.copy()})['out'], x @ w.T, atol=1e-6)
    for bias in (True, False):
        feeds = {'x': x, 'weight': w, **({'bias': b} if bias else {})}
        np.testing.assert_allclose(run(linear_graph(4, 2, bias), feeds)['out'], x @ w.T + (b if bias else 0), atol=1e-6)


@pytest.mark.parametrize('axis', [-1, 0])
def test_stable_softmax(axis):
    x = np.array([[1000, 1001, -1000], [999, 1002, -999]], dtype='float32')
    expected = np.exp(x-x.max(axis=axis, keepdims=True))
    expected /= expected.sum(axis=axis, keepdims=True)
    np.testing.assert_allclose(run(softmax_graph(axis), {'x': x})['out'], expected, atol=1e-6)


def test_rmsnorm():
    x = np.array([[0, 0, 0], [1, -2, 3]], dtype='float32')
    w = np.array([1, 2, 3], dtype='float32')
    expected = x / np.sqrt((x*x).mean(-1, keepdims=True) + 1e-6) * w
    np.testing.assert_allclose(run(rmsnorm_graph(), {'x': x, 'weight': w})['out'], expected, atol=1e-6)


def test_uncorrected_adam_all_state_outputs():
    feeds = {k: np.array(v, dtype='float32') for k, v in {
        'param':[1., -2.], 'grad':[0.3, -0.7], 'm':[0.1, 0.2], 'v':[0.03, 0.04],
        'lr':0.001, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'one':1.}.items()}
    m = feeds['beta1']*feeds['m'] + (1-feeds['beta1'])*feeds['grad']
    v = feeds['beta2']*feeds['v'] + (1-feeds['beta2'])*feeds['grad']**2
    expected = feeds['param'] - feeds['lr']*m/(np.sqrt(v)+feeds['eps'])
    actual = run(adam_graph(), feeds)
    for key, ref in [('param_next',expected), ('m_next',m), ('v_next',v)]:
        np.testing.assert_allclose(actual[key], ref, atol=1e-6)

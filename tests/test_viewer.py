import json
from pathlib import Path
import pytest
from tgaoi.viewer import load_package, render, main
from tgaoi.audit import audit_graph, catalog
from tgaoi.ir import Graph, Node, TensorType

ROOT = Path(__file__).resolve().parents[1]


def test_package_preserves_mapping_and_decomposition():
    data = load_package(ROOT / 'reference_models/rfdetr')
    nodes = {n['id']: n for n in data['nodes']}
    assert nodes['backbone.encoder.block[*]']['parent'] == 'backbone'
    assert nodes['decoder.layer[*].cross_attention']['parent'] == 'decoder.layer[*]'
    assert 'VIT_ATTENTION' in data['blocks']['WINDOWED_DINOV2_BLOCK']['children']
    assert data['parameter_materialization']['status'] == 'pending_runtime_walk'
    assert data['catalog']['RF_DETR']['audit']['status'] == 'blocked'
    assert data['catalog']['SOFTMAX']['audit']['status'] == 'lowerable'
    assert any('bindings' in s for s in data['catalog']['VIT_SELF_ATTENTION']['audit']['issues'])


def test_render_does_not_allow_script_injection():
    html = render({'model': '</script><script>alert(1)</script>'})
    payload = html.split('<script id="data" type="application/json">')[1].split('</script>')[0]
    assert '<' not in payload
    assert json.loads(payload)['model'].startswith('</script>')


def test_export_canonical_graph(tmp_path):
    graph_path = tmp_path / 'model.arch.json'
    graph_path.write_text(json.dumps(catalog()['SOFTMAX'].to_dict()))
    out = tmp_path / 'viewer.html'
    main(['--package', str(ROOT / 'reference_models/rfdetr'), '--graph', str(graph_path), '--output', str(out)])
    assert 'REDUCE_MAX' in out.read_text()
    assert '__VIEWER_DATA__' not in out.read_text()


@pytest.mark.parametrize('nodes', [[Node('x', 'IDENTITY', ['x'])], [Node('y', 'IDENTITY', ['x']), Node('y', 'IDENTITY', ['x'])]])
def test_duplicate_ids_rejected(nodes):
    g = Graph('bad', {'x': TensorType((1,))}, ['x'], nodes)
    with pytest.raises(ValueError, match='duplicate'):
        Graph.from_dict(g.to_dict())


def test_unknown_ops_are_blocked():
    g = Graph('unknown', {'x': TensorType((1,))}, ['y'], [Node('y', 'OPAQUE::unknown', ['x'])])
    assert audit_graph(g)['status'] == 'blocked'

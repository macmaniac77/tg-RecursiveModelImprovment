from tgaoi.aoi import rmsnorm_graph
from tgaoi.serialize import dumps, loads


def test_json_roundtrip():
    g = rmsnorm_graph()
    rebuilt = loads(dumps(g))
    assert rebuilt.to_dict() == g.to_dict()

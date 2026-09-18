import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tinygrad")

from tgaoi.backends import TinygradBackend
from tgaoi.importers.torch_fx import extract_torch_state, import_fx


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 7)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(7, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_torch_fx_to_tinygrad_numerical_equivalence():
    torch.manual_seed(7)
    model = TinyMLP().eval()
    x = torch.randn(5, 4)

    with torch.no_grad():
        expected = model(x).cpu().numpy()

    graph = import_fx(model, x)

    # The supported model must be fully lowered: no opaque framework calls.
    assert not any(node.op.startswith("OPAQUE::") for node in graph.nodes)
    assert [node.op for node in graph.nodes].count("MATMUL") == 2
    assert "MAXIMUM" in [node.op for node in graph.nodes]

    feeds = {"x": x.detach().cpu().numpy(), **extract_torch_state(model)}
    actual = TinygradBackend().run(graph, feeds)[graph.outputs[0]].numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

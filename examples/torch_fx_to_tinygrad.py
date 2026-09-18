"""Milestone demo: execute identical PyTorch MLP weights through Tinygrad."""

import numpy as np
import torch

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


torch.manual_seed(7)
model = TinyMLP().eval()
x = torch.randn(5, 4)

graph = import_fx(model, x)
feeds = {"x": x.detach().cpu().numpy(), **extract_torch_state(model)}
tg_out = TinygradBackend().run(graph, feeds)[graph.outputs[0]].numpy()

with torch.no_grad():
    torch_out = model(x).cpu().numpy()

print("Canonical operations:")
print(" -> ".join(node.op for node in graph.nodes))
print(f"max absolute error: {np.max(np.abs(tg_out - torch_out)):.8g}")
np.testing.assert_allclose(tg_out, torch_out, rtol=1e-5, atol=1e-6)
print("PASS: PyTorch FX -> canonical IR -> Tinygrad matches numerically.")

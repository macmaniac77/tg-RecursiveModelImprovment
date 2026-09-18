"""PyTorch FX -> tg-aoi IR importer scaffold.

The first milestone is intentionally conservative: trace simple feed-forward
modules, preserve unsupported operations as OPAQUE::<target>, and canonicalize
recognized operations. This module is the next implementation target rather
than pretending arbitrary PyTorch conversion is already solved.
"""

from __future__ import annotations

from ..ir import Graph, Node, TensorType


def import_fx(module, *example_args) -> Graph:
    try:
        import torch.fx as fx
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[torch]'") from exc

    traced = fx.symbolic_trace(module)
    inputs: dict[str, TensorType] = {}
    nodes: list[Node] = []
    outputs: list[str] = []
    for n in traced.graph.nodes:
        if n.op == "placeholder":
            inputs[n.name] = TensorType(("?",))
        elif n.op == "output":
            raw = n.args[0]
            values = raw if isinstance(raw, (tuple, list)) else (raw,)
            outputs = [x.name for x in values]
        else:
            in_names = [x.name for x in n.all_input_nodes]
            target = str(n.target)
            nodes.append(Node(n.name, f"OPAQUE::{target}", in_names, {"fx_op": n.op}))
    g = Graph(module.__class__.__name__, inputs, outputs, nodes, {"source": "torch.fx"})
    g.validate()
    return g

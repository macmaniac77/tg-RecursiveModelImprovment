from __future__ import annotations

from .ir import Graph, Node

_ALIAS = {
    "torch.matmul": "MATMUL",
    "jax.numpy.matmul": "MATMUL",
    "dot_general": "MATMUL",
    "add": "ADD",
    "mul": "MUL",
    "sub": "SUB",
    "div": "DIV",
}


def canonical_op(name: str) -> str:
    return _ALIAS.get(name, name.upper())


def canonicalize(graph: Graph) -> Graph:
    for node in graph.nodes:
        node.op = canonical_op(node.op)
    graph.metadata["canonicalized"] = True
    graph.validate()
    return graph

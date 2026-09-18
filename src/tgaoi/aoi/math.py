from __future__ import annotations

from ..ir import Graph, Node, TensorType


def matmul_graph(m: str | int = "M", k: str | int = "K", n: str | int = "N") -> Graph:
    """Semantic MATMUL AOI. Future versions may lower further into indexing/reduction UOps."""
    g = Graph(
        name="MATMUL",
        inputs={
            "a": TensorType((m, k)),
            "b": TensorType((k, n)),
        },
        outputs=["out"],
    )
    g.add(Node("out", "MATMUL", ["a", "b"], output_type=TensorType((m, n)), semantic_tags=["linear-algebra"]))
    g.validate()
    return g


def softmax_graph(axis: int = -1) -> Graph:
    x = TensorType(("...", "D"))
    g = Graph(name="SOFTMAX", inputs={"x": x}, outputs=["out"])
    g.add(Node("mx", "REDUCE_MAX", ["x"], {"axis": axis, "keepdim": True}))
    g.add(Node("shifted", "SUB", ["x", "mx"]))
    g.add(Node("ex", "EXP", ["shifted"]))
    g.add(Node("den", "REDUCE_SUM", ["ex"], {"axis": axis, "keepdim": True}))
    g.add(Node("out", "DIV", ["ex", "den"], output_type=x, semantic_tags=["normalization", "probability"]))
    g.validate()
    return g

from __future__ import annotations

from ..ir import Graph, Node, TensorType


def adam_graph() -> Graph:
    """One conceptual Adam update step, intentionally explicit and reducible."""
    t = TensorType(("...",))
    scalar = TensorType(())
    g = Graph(
        name="ADAM_STEP",
        inputs={
            "param": t,
            "grad": t,
            "m": t,
            "v": t,
            "lr": scalar,
            "beta1": scalar,
            "beta2": scalar,
            "eps": scalar,
            "one": scalar,
        },
        outputs=["param_next", "m_next", "v_next"],
    )
    g.add(Node("b1m", "MUL", ["beta1", "m"]))
    g.add(Node("omb1", "SUB", ["one", "beta1"]))
    g.add(Node("g1", "MUL", ["omb1", "grad"]))
    g.add(Node("m_next", "ADD", ["b1m", "g1"]))
    g.add(Node("b2v", "MUL", ["beta2", "v"]))
    g.add(Node("g2", "MUL", ["grad", "grad"]))
    g.add(Node("omb2", "SUB", ["one", "beta2"]))
    g.add(Node("g2s", "MUL", ["omb2", "g2"]))
    g.add(Node("v_next", "ADD", ["b2v", "g2s"]))
    g.add(Node("rootv", "SQRT", ["v_next"]))
    g.add(Node("den", "ADD", ["rootv", "eps"]))
    g.add(Node("step0", "DIV", ["m_next", "den"]))
    g.add(Node("step", "MUL", ["lr", "step0"]))
    g.add(Node("param_next", "SUB", ["param", "step"]))
    g.validate()
    return g

from __future__ import annotations

from ..ir import Graph, Node, TensorType


def rmsnorm_graph(eps: float = 1e-6) -> Graph:
    x = TensorType(("...", "D"))
    g = Graph(
        name="RMSNORM",
        inputs={"x": x, "weight": TensorType(("D",))},
        outputs=["out"],
    )
    g.add(Node("sq", "MUL", ["x", "x"]))
    g.add(Node("mean_sq", "REDUCE_MEAN", ["sq"], {"axis": -1, "keepdim": True}))
    g.add(Node("eps", "CONST", [], {"value": eps}))
    g.add(Node("den", "ADD", ["mean_sq", "eps"]))
    g.add(Node("inv", "RSQRT", ["den"]))
    g.add(Node("norm", "MUL", ["x", "inv"]))
    g.add(Node("out", "MUL", ["norm", "weight"], output_type=x, semantic_tags=["normalization"]))
    g.validate()
    return g


def linear_graph(din: str | int = "DIN", dout: str | int = "DOUT", bias: bool = True) -> Graph:
    inputs = {
        "x": TensorType(("...", din)),
        "weight": TensorType((dout, din)),
    }
    if bias:
        inputs["bias"] = TensorType((dout,))
    g = Graph(name="LINEAR", inputs=inputs, outputs=["out"])
    g.add(Node("wt", "TRANSPOSE", ["weight"], {"axes": (-1, -2)}))
    g.add(Node("mm", "MATMUL", ["x", "wt"]))
    if bias:
        g.add(Node("out", "ADD", ["mm", "bias"], output_type=TensorType(("...", dout)), semantic_tags=["linear"]))
    else:
        g.add(Node("out", "IDENTITY", ["mm"], output_type=TensorType(("...", dout)), semantic_tags=["linear"]))
    g.validate()
    return g


def attention_graph() -> Graph:
    """Scaled dot-product attention as a composition of lower-level graph operations."""
    qkv = TensorType(("B", "H", "T", "D"))
    g = Graph(
        name="ATTENTION",
        inputs={"q": qkv, "k": qkv, "v": qkv, "scale": TensorType(())},
        outputs=["out"],
    )
    g.add(Node("kt", "TRANSPOSE", ["k"], {"axes": (-1, -2)}))
    g.add(Node("scores", "MATMUL", ["q", "kt"]))
    g.add(Node("scaled", "MUL", ["scores", "scale"]))
    g.add(Node("weights", "AOI::SOFTMAX", ["scaled"], {"axis": -1}))
    g.add(Node("out", "MATMUL", ["weights", "v"], output_type=qkv, semantic_tags=["attention"]))
    g.validate()
    return g

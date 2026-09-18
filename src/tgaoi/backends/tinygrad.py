from __future__ import annotations

from typing import Any
from ..ir import Graph


class TinygradBackend:
    """Minimal executable backend for the starter alphabet.

    Tinygrad is an optional dependency. The implementation deliberately maps
    the canonical IR into ordinary Tinygrad Tensor operations rather than
    binding to unstable compiler internals. Later lowering stages can target
    UOps directly when the IR semantics are mature.
    """

    def __init__(self):
        try:
            from tinygrad import Tensor  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[tinygrad]'") from exc

    def run(self, graph: Graph, feeds: dict[str, Any]):
        from tinygrad import Tensor

        graph.validate()
        env = dict(feeds)
        for n in graph.nodes:
            xs = [env[i] for i in n.inputs]
            op = n.op
            if op == "CONST": env[n.id] = Tensor(n.attrs["value"])
            elif op == "IDENTITY": env[n.id] = xs[0]
            elif op == "ADD": env[n.id] = xs[0] + xs[1]
            elif op == "SUB": env[n.id] = xs[0] - xs[1]
            elif op == "MUL": env[n.id] = xs[0] * xs[1]
            elif op == "DIV": env[n.id] = xs[0] / xs[1]
            elif op == "EXP": env[n.id] = xs[0].exp()
            elif op == "SQRT": env[n.id] = xs[0].sqrt()
            elif op == "RSQRT": env[n.id] = xs[0].rsqrt()
            elif op == "MATMUL": env[n.id] = xs[0].matmul(xs[1])
            elif op == "TRANSPOSE": env[n.id] = xs[0].transpose(*n.attrs["axes"])
            elif op == "REDUCE_SUM": env[n.id] = xs[0].sum(axis=n.attrs["axis"], keepdim=n.attrs.get("keepdim", False))
            elif op == "REDUCE_MEAN": env[n.id] = xs[0].mean(axis=n.attrs["axis"], keepdim=n.attrs.get("keepdim", False))
            elif op == "REDUCE_MAX": env[n.id] = xs[0].max(axis=n.attrs["axis"], keepdim=n.attrs.get("keepdim", False))
            else:
                raise NotImplementedError(f"Tinygrad lowering missing for {op}")
        return {name: env[name] for name in graph.outputs}

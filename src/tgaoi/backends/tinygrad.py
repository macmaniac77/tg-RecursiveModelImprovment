from __future__ import annotations

from typing import Any
from ..ir import Graph


SUPPORTED_OPS = frozenset({
    "COS", "SIN", "LOG",
    "SIGMOID", "TANH", "ERF", "CONV2D", "RESHAPE", "CONCAT", "SLICE", "MAX_POOL2D", "AVG_POOL2D",
    "CONST", "IDENTITY", "ADD", "SUB", "MUL", "DIV", "MAXIMUM", "EXP",
    "SQRT", "RSQRT", "MATMUL", "TRANSPOSE", "REDUCE_SUM", "REDUCE_MEAN", "REDUCE_MAX",
})


class TinygradBackend:
    """Executable Tinygrad backend for the starter canonical alphabet.

    Inputs may be Tinygrad tensors, NumPy arrays, Python scalars/lists, or
    detached framework tensors. The backend converts them at the boundary,
    then executes only the operations encoded in the canonical graph.
    """

    def __init__(self):
        try:
            from tinygrad import Tensor  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[tinygrad]'") from exc

    @staticmethod
    def _tensor(value):
        from tinygrad import Tensor

        if isinstance(value, Tensor):
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return Tensor(value)

    def run(self, graph: Graph, feeds: dict[str, Any]):
        from tinygrad import Tensor

        graph.validate()
        missing = [name for name in graph.inputs if name not in feeds]
        if missing:
            raise KeyError(f"missing graph feeds: {missing}")

        env = {name: self._tensor(feeds[name]) for name in graph.inputs}

        for n in graph.nodes:
            xs = [env[i] for i in n.inputs]
            op = n.op

            if op in ("COS", "SIN", "LOG"):
                env[n.id] = getattr(xs[0], op.lower())()
            elif op == "SIGMOID":
                env[n.id] = xs[0].sigmoid()
            elif op == "TANH":
                env[n.id] = xs[0].tanh()
            elif op == "ERF":
                env[n.id] = xs[0].erf()
            elif op == "CONV2D":
                env[n.id] = xs[0].conv2d(xs[1], xs[2] if len(xs) > 2 else None, **n.attrs)
            elif op == "RESHAPE":
                env[n.id] = xs[0].reshape(n.attrs["shape"])
            elif op == "CONCAT":
                env[n.id] = xs[0].cat(*xs[1:], dim=n.attrs["axis"])
            elif op == "SLICE":
                indices = [slice(None)] * len(xs[0].shape)
                indices[n.attrs["axis"]] = slice(n.attrs["start"], n.attrs["stop"])
                env[n.id] = xs[0][tuple(indices)]
            elif op in ("MAX_POOL2D", "AVG_POOL2D"):
                env[n.id] = getattr(xs[0], op.lower())(**n.attrs)
            elif op == "CONST":
                env[n.id] = Tensor(n.attrs["value"])
            elif op == "IDENTITY":
                env[n.id] = xs[0]
            elif op == "ADD":
                env[n.id] = xs[0] + xs[1]
            elif op == "SUB":
                env[n.id] = xs[0] - xs[1]
            elif op == "MUL":
                env[n.id] = xs[0] * xs[1]
            elif op == "DIV":
                env[n.id] = xs[0] / xs[1]
            elif op == "MAXIMUM":
                env[n.id] = xs[0].maximum(xs[1])
            elif op == "EXP":
                env[n.id] = xs[0].exp()
            elif op == "SQRT":
                env[n.id] = xs[0].sqrt()
            elif op == "RSQRT":
                env[n.id] = xs[0].rsqrt()
            elif op == "MATMUL":
                env[n.id] = xs[0].matmul(xs[1])
            elif op == "TRANSPOSE":
                env[n.id] = xs[0].transpose(*n.attrs["axes"])
            elif op == "REDUCE_SUM":
                env[n.id] = xs[0].sum(axis=n.attrs["axis"], keepdim=n.attrs.get("keepdim", False))
            elif op == "REDUCE_MEAN":
                env[n.id] = xs[0].mean(axis=n.attrs["axis"], keepdim=n.attrs.get("keepdim", False))
            elif op == "REDUCE_MAX":
                env[n.id] = xs[0].max(axis=n.attrs["axis"], keepdim=n.attrs.get("keepdim", False))
            else:
                raise NotImplementedError(f"Tinygrad lowering missing for {op}")

        return {name: env[name] for name in graph.outputs}

    def inspect_uops(self, graph: Graph, feeds: dict[str, Any]):
        """Inspect the actual lazy Tinygrad UOp DAG, before scheduling/kernel fusion.

        Omit buffer values and addresses. This is not optimized kernel assembly or
        an execution benchmark; graph construction may allocate input tensors.
        """
        outputs = self.run(graph, feeds)
        ordered = []
        indices = {}
        for tensor in outputs.values():
            for uop in tensor.uop.toposort():
                if uop not in indices:
                    indices[uop] = len(ordered)
                    ordered.append(uop)
        return {
            "schema_version": 1, "source_graph": graph.name,
            "stage": "lazy_tensor_uops_before_scheduling",
            "nodes": [{"id": indices[uop], "op": uop.op.name, "dtype": str(uop.dtype),
                       "inputs": [indices[src] for src in uop.src]} for uop in ordered],
            "outputs": {name: indices[tensor.uop] for name, tensor in outputs.items()},
        }

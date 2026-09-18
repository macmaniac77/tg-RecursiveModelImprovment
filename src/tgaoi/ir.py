from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


@dataclass(frozen=True)
class TensorType:
    """A lightweight type for graph validation and architecture comparison."""

    shape: tuple[str | int, ...]
    dtype: str = "float32"
    axes: tuple[str, ...] | None = None

    def compatible_with(self, other: "TensorType") -> bool:
        if self.dtype != other.dtype or len(self.shape) != len(other.shape):
            return False
        for a, b in zip(self.shape, other.shape):
            if isinstance(a, int) and isinstance(b, int) and a != b:
                return False
        return True


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    output_type: TensorType | None = None
    semantic_tags: list[str] = field(default_factory=list)


@dataclass
class Graph:
    name: str
    inputs: dict[str, TensorType]
    outputs: list[str]
    nodes: list[Node] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def add(self, node: Node) -> str:
        if node.id in self.node_map() or node.id in self.inputs:
            raise ValueError(f"duplicate graph value id: {node.id}")
        self.nodes.append(node)
        return node.id

    def validate(self) -> None:
        known = set(self.inputs)
        for node in self.nodes:
            missing = [x for x in node.inputs if x not in known]
            if missing:
                raise ValueError(f"{node.id}: unknown inputs {missing}")
            known.add(node.id)
        missing_out = [x for x in self.outputs if x not in known]
        if missing_out:
            raise ValueError(f"unknown graph outputs {missing_out}")

    def to_dict(self) -> dict[str, Any]:
        def encode_type(t: TensorType | None):
            return asdict(t) if t is not None else None

        return {
            "name": self.name,
            "inputs": {k: asdict(v) for k, v in self.inputs.items()},
            "outputs": self.outputs,
            "nodes": [
                {
                    "id": n.id,
                    "op": n.op,
                    "inputs": n.inputs,
                    "attrs": n.attrs,
                    "output_type": encode_type(n.output_type),
                    "semantic_tags": n.semantic_tags,
                }
                for n in self.nodes
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Graph":
        inputs = {
            k: TensorType(tuple(v["shape"]), v.get("dtype", "float32"), tuple(v["axes"]) if v.get("axes") else None)
            for k, v in data["inputs"].items()
        }
        nodes = []
        for n in data["nodes"]:
            ot = n.get("output_type")
            nodes.append(
                Node(
                    id=n["id"],
                    op=n["op"],
                    inputs=list(n.get("inputs", [])),
                    attrs=dict(n.get("attrs", {})),
                    output_type=TensorType(tuple(ot["shape"]), ot.get("dtype", "float32"), tuple(ot["axes"]) if ot.get("axes") else None) if ot else None,
                    semantic_tags=list(n.get("semantic_tags", [])),
                )
            )
        graph = cls(
            name=data["name"],
            inputs=inputs,
            outputs=list(data["outputs"]),
            nodes=nodes,
            metadata=dict(data.get("metadata", {})),
        )
        graph.validate()
        return graph


def walk_ops(graph: Graph) -> Iterable[str]:
    for node in graph.nodes:
        yield node.op

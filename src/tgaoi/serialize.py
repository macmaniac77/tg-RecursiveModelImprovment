from __future__ import annotations

import json
from pathlib import Path
from .ir import Graph


def dumps(graph: Graph, *, indent: int = 2) -> str:
    return json.dumps(graph.to_dict(), indent=indent, sort_keys=True)


def loads(text: str) -> Graph:
    return Graph.from_dict(json.loads(text))


def save(graph: Graph, path: str | Path) -> None:
    Path(path).write_text(dumps(graph) + "\n", encoding="utf-8")


def load(path: str | Path) -> Graph:
    return loads(Path(path).read_text(encoding="utf-8"))

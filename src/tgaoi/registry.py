from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .ir import Graph


@dataclass
class AOIDefinition:
    name: str
    version: str
    graph_factory: Callable[..., Graph]
    purpose: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def expand(self, **kwargs) -> Graph:
        graph = self.graph_factory(**kwargs)
        graph.metadata.setdefault("aoi", {})
        graph.metadata["aoi"].update({"name": self.name, "version": self.version})
        return graph


class Registry:
    def __init__(self):
        self._items: dict[tuple[str, str], AOIDefinition] = {}

    def register(self, definition: AOIDefinition) -> None:
        key = (definition.name, definition.version)
        if key in self._items:
            raise ValueError(f"AOI already registered: {key}")
        self._items[key] = definition

    def get(self, name: str, version: str = "1") -> AOIDefinition:
        return self._items[(name, version)]

    def list(self) -> list[AOIDefinition]:
        return list(self._items.values())


registry = Registry()

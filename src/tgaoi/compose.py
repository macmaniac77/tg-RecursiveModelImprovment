"""Explicit, namespaced AOI composition with preserved instance provenance."""
from copy import deepcopy
from .ir import Graph, Node


def inline(parent: Graph, child: Graph, prefix: str, bindings: dict[str, str]) -> list[str]:
    child.validate()
    if set(bindings) != set(child.inputs):
        raise ValueError(f'{child.name}: exact input bindings required')
    known = set(parent.inputs) | set(parent.node_map())
    if not set(bindings.values()) <= known:
        raise ValueError('binding references unknown parent values')
    mapping = {**bindings, **{n.id: f'{prefix}/{n.id}' for n in child.nodes}}
    if any(mapping[n.id] in known for n in child.nodes):
        raise ValueError(f'duplicate composition prefix: {prefix}')
    for node in child.nodes:
        parent.add(Node(mapping[node.id], node.op, [mapping[x] for x in node.inputs],
                        deepcopy(node.attrs), node.output_type, list(node.semantic_tags)))
    parent.metadata.setdefault('aoi_instances', []).append({
        'prefix': prefix, 'aoi': child.name, 'bindings': dict(bindings),
        'outputs': [mapping[x] for x in child.outputs], 'metadata': deepcopy(child.metadata)})
    return [mapping[x] for x in child.outputs]

"""Read-only capability inventory; structural validity is not equivalence."""
from . import aoi
from .backends.tinygrad import SUPPORTED_OPS


def catalog():
    """Default graph from every public builder, plus named activation variants."""
    graphs = {g.name: g for name in aoi.__all__ for g in [getattr(aoi, name)()]}
    for kind in ("gelu", "silu", "sigmoid", "tanh"):
        g = aoi.activation_graph(kind)
        graphs[g.name] = g
    avg = aoi.pool2d_graph("avg")
    graphs[avg.name] = avg
    return graphs


def audit_graph(graph, definitions=None):
    definitions = catalog() if definitions is None else definitions
    graph.validate()
    issues = []
    for node in graph.nodes:
        if node.op.startswith("AOI::"):
            name = node.op[5:]
            child = definitions.get(name)
            issues.append(f"{node.id}: nested AOI execution is not implemented ({name})")
            if child is None:
                issues.append(f"{node.id}: no graph definition for {name}")
            elif len(node.inputs) != len(child.inputs):
                issues.append(f"{node.id}: {len(node.inputs)} bindings for {len(child.inputs)} default child inputs; explicit binding/configuration required")
        elif node.op not in SUPPORTED_OPS:
            issues.append(f"{node.id}: unsupported primitive {node.op}")
    return {"status": "blocked" if issues else "lowerable", "issues": issues,
            "note": "Lowerable means primitive coverage only, not shape checking or numerical/source equivalence."}

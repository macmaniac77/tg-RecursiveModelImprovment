"""JAXPR -> tg-aoi IR importer scaffold."""

from __future__ import annotations

from ..ir import Graph, Node, TensorType


def import_jaxpr(fn, *example_args) -> Graph:
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[jax]'") from exc

    jaxpr = jax.make_jaxpr(fn)(*example_args).jaxpr
    inputs = {str(v): TensorType(("?",)) for v in jaxpr.invars}
    nodes: list[Node] = []
    name_of: dict[object, str] = {v: str(v) for v in jaxpr.invars}
    for idx, eqn in enumerate(jaxpr.eqns):
        out_id = f"v{idx}"
        in_names = [name_of.get(v, str(v)) for v in eqn.invars]
        op = str(eqn.primitive)
        nodes.append(Node(out_id, op, in_names, dict(eqn.params)))
        for v in eqn.outvars:
            name_of[v] = out_id
    outputs = [name_of.get(v, str(v)) for v in jaxpr.outvars]
    g = Graph(getattr(fn, "__name__", "jax_fn"), inputs, outputs, nodes, {"source": "jaxpr"})
    g.validate()
    return g

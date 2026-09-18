"""PyTorch FX -> tg-aoi IR importer.

Milestone 1 deliberately supports a small, auditable subset:
- placeholders
- torch.nn.Linear
- torch.nn.ReLU
- graph output

Linear is lowered immediately to TRANSPOSE -> MATMUL -> optional ADD.
ReLU is lowered to CONST(0) -> MAXIMUM. Model parameters become explicit
Graph inputs so the canonical IR contains no hidden framework state.

Unsupported operations are preserved as OPAQUE::<target> nodes rather than
silently translated incorrectly.
"""

from __future__ import annotations

from typing import Any

from ..ir import Graph, Node, TensorType


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def _tensor_type(value: Any) -> TensorType:
    shape = tuple(int(x) for x in value.shape)
    return TensorType(shape=shape, dtype=_dtype_name(value.dtype))


def extract_torch_state(module) -> dict[str, Any]:
    """Return model parameters as NumPy arrays keyed exactly as the IR expects."""
    state: dict[str, Any] = {}
    for name, value in module.state_dict().items():
        state[f"param::{name}"] = value.detach().cpu().numpy()
    return state


def import_fx(module, *example_args) -> Graph:
    try:
        import torch
        import torch.fx as fx
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[torch]'") from exc

    traced = fx.symbolic_trace(module)
    inputs: dict[str, TensorType] = {}
    nodes: list[Node] = []
    outputs: list[str] = []

    placeholder_index = 0

    for n in traced.graph.nodes:
        if n.op == "placeholder":
            if placeholder_index < len(example_args):
                inputs[n.name] = _tensor_type(example_args[placeholder_index])
            else:
                inputs[n.name] = TensorType(("?",))
            placeholder_index += 1
            continue

        if n.op == "output":
            raw = n.args[0]
            values = raw if isinstance(raw, (tuple, list)) else (raw,)
            outputs = [x.name for x in values]
            continue

        in_names = [x.name for x in n.all_input_nodes]

        if n.op == "call_module":
            submodule = traced.get_submodule(str(n.target))

            if isinstance(submodule, nn.Linear):
                if len(in_names) != 1:
                    raise ValueError(f"{n.name}: Linear expected one tensor input")

                weight_name = f"param::{n.target}.weight"
                inputs[weight_name] = _tensor_type(submodule.weight)

                wt = f"{n.name}__weight_t"
                mm = f"{n.name}__matmul"
                nodes.append(Node(wt, "TRANSPOSE", [weight_name], {"axes": (-1, -2)}))
                nodes.append(Node(mm, "MATMUL", [in_names[0], wt]))

                if submodule.bias is not None:
                    bias_name = f"param::{n.target}.bias"
                    inputs[bias_name] = _tensor_type(submodule.bias)
                    nodes.append(
                        Node(
                            n.name,
                            "ADD",
                            [mm, bias_name],
                            semantic_tags=["linear", "torch.nn.Linear"],
                        )
                    )
                else:
                    nodes.append(
                        Node(
                            n.name,
                            "IDENTITY",
                            [mm],
                            semantic_tags=["linear", "torch.nn.Linear"],
                        )
                    )
                continue

            if isinstance(submodule, nn.ReLU):
                zero = f"{n.name}__zero"
                nodes.append(Node(zero, "CONST", [], {"value": 0.0}))
                nodes.append(
                    Node(
                        n.name,
                        "MAXIMUM",
                        [in_names[0], zero],
                        semantic_tags=["activation", "torch.nn.ReLU"],
                    )
                )
                continue

        target = str(n.target)
        nodes.append(
            Node(
                n.name,
                f"OPAQUE::{target}",
                in_names,
                {"fx_op": n.op},
            )
        )

    g = Graph(
        name=module.__class__.__name__,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        metadata={
            "source": "torch.fx",
            "supported_subset": ["torch.nn.Linear", "torch.nn.ReLU"],
        },
    )
    g.validate()
    return g

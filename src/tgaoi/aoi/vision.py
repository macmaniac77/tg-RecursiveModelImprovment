from __future__ import annotations

from ..ir import Graph, Node, TensorType


def activation_graph(kind: str = "relu") -> Graph:
    if kind not in ("relu", "silu", "gelu", "sigmoid", "tanh"):
        raise ValueError(f"unsupported activation: {kind}")
    x = TensorType(("...", "D"))
    g = Graph(f"ACTIVATION::{kind.upper()}", {"x": x}, ["out"])
    if kind == "relu":
        g.add(Node("zero", "CONST", attrs={"value": 0.0}))
        g.add(Node("out", "MAXIMUM", ["x", "zero"], output_type=x))
    elif kind in ("sigmoid", "silu"):
        g.add(Node("sigmoid", "SIGMOID", ["x"]))
        g.add(Node("out", "MUL" if kind == "silu" else "IDENTITY",
                   ["x", "sigmoid"] if kind == "silu" else ["sigmoid"], output_type=x))
    elif kind == "tanh":
        g.add(Node("out", "TANH", ["x"], output_type=x))
    else:
        g.add(Node("scale", "CONST", attrs={"value": 2 ** -0.5}))
        g.add(Node("scaled", "MUL", ["x", "scale"]))
        g.add(Node("erf", "ERF", ["scaled"]))
        g.add(Node("one", "CONST", attrs={"value": 1.0}))
        g.add(Node("plus", "ADD", ["erf", "one"]))
        g.add(Node("half", "CONST", attrs={"value": 0.5}))
        g.add(Node("gate", "MUL", ["plus", "half"]))
        g.add(Node("out", "MUL", ["x", "gate"], output_type=x))
    g.metadata["semantics"] = "elementwise; GELU uses erf, not tanh approximation"
    g.validate()
    return g


def layernorm_graph() -> Graph:
    x = TensorType(("...", "D"))
    g = Graph("LAYERNORM", {"x": x, "weight": TensorType(("D",)),
              "bias": TensorType(("D",)), "eps": TensorType(())}, ["out"])
    g.add(Node("mean", "REDUCE_MEAN", ["x"], {"axis": -1, "keepdim": True}))
    g.add(Node("centered", "SUB", ["x", "mean"]))
    g.add(Node("squared", "MUL", ["centered", "centered"]))
    g.add(Node("variance", "REDUCE_MEAN", ["squared"], {"axis": -1, "keepdim": True}))
    g.add(Node("den", "ADD", ["variance", "eps"]))
    g.add(Node("inv", "RSQRT", ["den"]))
    g.add(Node("norm", "MUL", ["centered", "inv"]))
    g.add(Node("scaled", "MUL", ["norm", "weight"]))
    g.add(Node("out", "ADD", ["scaled", "bias"], output_type=x))
    g.metadata["semantics"] = "last-axis affine normalization, population variance"
    g.validate()
    return g


def conv2d_graph(stride=1, padding=0, dilation=1, groups=1, bias=False) -> Graph:
    inputs = {"x": TensorType(("N", "CIN", "H", "W")),
              "weight": TensorType(("COUT", "CIN_PER_GROUP", "KH", "KW"))}
    if bias:
        inputs["bias"] = TensorType(("COUT",))
    g = Graph("CONV2D", inputs, ["out"], metadata={"semantics": "NCHW cross-correlation; Tinygrad tensor-level lowering"})
    g.add(Node("out", "CONV2D", list(inputs),
               {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups},
               output_type=TensorType(("N", "COUT", "OH", "OW"))))
    g.validate()
    return g


def patch_embedding_graph() -> Graph:
    g = Graph(
        name="PATCH_EMBEDDING",
        inputs={"image": TensorType(("B", "C", "H", "W")), "weight": TensorType(("D", "C", "P", "P"))},
        outputs=["tokens"],
        metadata={
            "level": "sentence",
            "source_examples": ["Dinov2WithRegistersPatchEmbeddings"],
        },
    )
    g.add(Node("patches", "AOI::CONV2D", ["image", "weight"], {"stride": "P", "kernel": "P"}))
    g.add(Node("flat", "FLATTEN", ["patches"], {"start_dim": 2}))
    g.add(Node("tokens", "TRANSPOSE", ["flat"], {"axes": (1, 2)}, output_type=TensorType(("B", "T", "D"))))
    g.validate()
    return g


def vit_self_attention_graph() -> Graph:
    x = TensorType(("B", "T", "D"))
    g = Graph(
        name="VIT_SELF_ATTENTION",
        inputs={
            "x": x,
            "q_weight": TensorType(("D", "D")),
            "k_weight": TensorType(("D", "D")),
            "v_weight": TensorType(("D", "D")),
            "out_weight": TensorType(("D", "D")),
            "scale": TensorType(()),
        },
        outputs=["out"],
        metadata={
            "level": "sentence",
            "source_examples": ["Dinov2WithRegistersSelfAttention", "Dinov2WithRegistersAttention"],
        },
    )
    g.add(Node("q", "AOI::LINEAR", ["x", "q_weight"]))
    g.add(Node("k", "AOI::LINEAR", ["x", "k_weight"]))
    g.add(Node("v", "AOI::LINEAR", ["x", "v_weight"]))
    g.add(Node("context", "AOI::ATTENTION", ["q", "k", "v", "scale"]))
    g.add(Node("out", "AOI::LINEAR", ["context", "out_weight"], output_type=x))
    g.validate()
    return g


def vit_mlp_graph() -> Graph:
    x = TensorType(("B", "T", "D"))
    g = Graph(
        name="VIT_MLP",
        inputs={
            "x": x,
            "fc1_weight": TensorType(("H", "D")),
            "fc2_weight": TensorType(("D", "H")),
        },
        outputs=["out"],
        metadata={"level": "sentence", "source_examples": ["Dinov2WithRegistersMLP"]},
    )
    g.add(Node("h", "AOI::LINEAR", ["x", "fc1_weight"]))
    g.add(Node("a", "AOI::ACTIVATION::GELU", ["h"]))
    g.add(Node("out", "AOI::LINEAR", ["a", "fc2_weight"], output_type=x))
    g.validate()
    return g


def vit_block_graph() -> Graph:
    x = TensorType(("B", "T", "D"))
    g = Graph(
        name="WINDOWED_DINOV2_BLOCK",
        inputs={"x": x},
        outputs=["out"],
        metadata={
            "level": "paragraph",
            "source_examples": ["WindowedDinov2WithRegistersLayer"],
            "notes": "Window/full-attention reshape and LayerScale/DropPath are explicit child AOIs.",
        },
    )
    g.add(Node("n1", "AOI::LAYERNORM", ["x"]))
    g.add(Node("attn", "AOI::VIT_SELF_ATTENTION", ["n1"]))
    g.add(Node("ls1", "AOI::LAYERSCALE", ["attn"]))
    g.add(Node("dp1", "AOI::DROPPATH", ["ls1"]))
    g.add(Node("r1", "ADD", ["x", "dp1"]))
    g.add(Node("n2", "AOI::LAYERNORM", ["r1"]))
    g.add(Node("mlp", "AOI::VIT_MLP", ["n2"]))
    g.add(Node("ls2", "AOI::LAYERSCALE", ["mlp"]))
    g.add(Node("dp2", "AOI::DROPPATH", ["ls2"]))
    g.add(Node("out", "ADD", ["r1", "dp2"], output_type=x))
    g.validate()
    return g


def convx_graph() -> Graph:
    x = TensorType(("B", "CIN", "H", "W"))
    g = Graph(
        name="CONVX",
        inputs={"x": x},
        outputs=["out"],
        metadata={"level": "sentence", "source_examples": ["ConvX"]},
    )
    g.add(Node("conv", "AOI::CONV2D", ["x"]))
    g.add(Node("norm", "AOI::NORM2D", ["conv"]))
    g.add(Node("out", "AOI::ACTIVATION", ["norm"], output_type=TensorType(("B", "COUT", "OH", "OW"))))
    g.validate()
    return g


def bottleneck_graph() -> Graph:
    x = TensorType(("B", "C", "H", "W"))
    g = Graph(
        name="BOTTLENECK",
        inputs={"x": x},
        outputs=["out"],
        metadata={"level": "sentence", "source_examples": ["Bottleneck"]},
    )
    g.add(Node("a", "AOI::CONVX", ["x"]))
    g.add(Node("b", "AOI::CONVX", ["a"]))
    g.add(Node("out", "AOI::OPTIONAL_RESIDUAL", ["x", "b"], output_type=x))
    g.validate()
    return g


def c2f_graph() -> Graph:
    g = Graph(
        name="C2F",
        inputs={"x": TensorType(("B", "CIN", "H", "W"))},
        outputs=["out"],
        metadata={"level": "paragraph", "source_examples": ["C2f"]},
    )
    g.add(Node("split", "AOI::CONVX_AND_SPLIT", ["x"]))
    g.add(Node("chain", "AOI::REPEATED_BOTTLENECK", ["split"]))
    g.add(Node("cat", "CONCAT", ["split", "chain"], {"axis": 1}))
    g.add(Node("out", "AOI::CONVX", ["cat"], output_type=TensorType(("B", "COUT", "H", "W"))))
    g.validate()
    return g


def multiscale_projector_graph() -> Graph:
    g = Graph(
        name="MULTISCALE_PROJECTOR",
        inputs={"features": TensorType(("LEVELS", "B", "C", "H", "W"))},
        outputs=["projected"],
        metadata={"level": "paragraph", "source_examples": ["MultiScaleProjector"]},
    )
    g.add(Node("resampled", "AOI::MULTISCALE_RESAMPLE", ["features"]))
    g.add(Node("fused", "CONCAT", ["resampled"], {"axis": 1}))
    g.add(Node("mixed", "AOI::C2F", ["fused"]))
    g.add(Node("projected", "AOI::NORM2D", ["mixed"]))
    g.validate()
    return g


def dino_backbone_graph() -> Graph:
    g = Graph(
        name="DINO_V2_BACKBONE",
        inputs={"image": TensorType(("B", "3", "H", "W"))},
        outputs=["features"],
        metadata={
            "level": "chapter",
            "source_examples": ["TinyDinoV2", "DINOv2Backbone", "WindowedDinov2WithRegistersBackbone"],
        },
    )
    g.add(Node("embed", "AOI::PATCH_EMBEDDING_WITH_POSITION_AND_REGISTERS", ["image"]))
    g.add(Node("encoder", "AOI::REPEATED_WINDOWED_DINOV2_BLOCK", ["embed"]))
    g.add(Node("selected", "AOI::FEATURE_TAP_SELECT", ["encoder"]))
    g.add(Node("features", "AOI::MULTISCALE_PROJECTOR", ["selected"]))
    g.validate()
    return g

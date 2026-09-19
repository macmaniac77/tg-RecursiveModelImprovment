from __future__ import annotations

from ..ir import Graph, Node, TensorType


def activation_graph(kind: str = "relu") -> Graph:
    """Canonical activation AOI.

    The RF-DETR Tinygrad port currently uses relu, silu and exact erf-based gelu.
    These remain named AOIs until their primitive/UOp lowering is implemented.
    """
    x = TensorType(("...", "D"))
    g = Graph(
        name=f"ACTIVATION::{kind.upper()}",
        inputs={"x": x},
        outputs=["out"],
        metadata={
            "level": "word",
            "source_examples": ["reference_models/rfdetr/detect.py::Activation"],
            "lowering_status": "semantic_aoi_pending_primitive_expansion",
        },
    )
    g.add(Node("out", f"ACTIVATION_PRIMITIVE::{kind.upper()}", ["x"], output_type=x))
    g.validate()
    return g


def layernorm_graph() -> Graph:
    x = TensorType(("...", "D"))
    g = Graph(
        name="LAYERNORM",
        inputs={"x": x, "weight": TensorType(("D",)), "bias": TensorType(("D",)), "eps": TensorType(())},
        outputs=["out"],
        metadata={"level": "word", "lowering_status": "pending_mean_variance_primitive_expansion"},
    )
    g.add(Node("norm", "LAYERNORM_PRIMITIVE", ["x", "eps"]))
    g.add(Node("scaled", "MUL", ["norm", "weight"]))
    g.add(Node("out", "ADD", ["scaled", "bias"], output_type=x))
    g.validate()
    return g


def conv2d_graph() -> Graph:
    """Semantic convolution AOI.

    Convolution is intentionally represented as a reusable AOI, not as an
    irreducible alphabet letter. A later lowering pass should expand it into the
    chosen Tinygrad/UOp-level substrate while preserving this semantic wrapper.
    """
    g = Graph(
        name="CONV2D",
        inputs={
            "x": TensorType(("N", "CIN", "H", "W")),
            "weight": TensorType(("COUT", "CIN_PER_GROUP", "KH", "KW")),
        },
        outputs=["out"],
        metadata={
            "level": "word",
            "lowering_status": "pending_uop_expansion",
            "attributes": ["stride", "padding", "dilation", "groups", "bias"],
        },
    )
    g.add(Node("out", "CONV2D_PRIMITIVE", ["x", "weight"], output_type=TensorType(("N", "COUT", "OH", "OW"))))
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

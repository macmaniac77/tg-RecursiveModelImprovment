from __future__ import annotations

from ..ir import Graph, Node, TensorType


def bilinear_sampler_graph() -> Graph:
    g = Graph(
        name="BILINEAR_SAMPLER",
        inputs={
            "features": TensorType(("N", "C", "H", "W")),
            "grid": TensorType(("N", "P", "2")),
        },
        outputs=["sampled"],
        metadata={"level": "word", "source_examples": ["bilinear_sampler_fast"]},
    )
    g.add(Node("coords", "AOI::BILINEAR_COORDINATES", ["grid"]))
    g.add(Node("neighbors", "GATHER", ["features", "coords"]))
    g.add(Node("weights", "AOI::BILINEAR_WEIGHTS", ["grid", "coords"]))
    g.add(Node("sampled", "AOI::WEIGHTED_SUM", ["neighbors", "weights"], output_type=TensorType(("N", "C", "P"))))
    g.validate()
    return g


def ms_deform_attention_graph() -> Graph:
    g = Graph(
        name="MS_DEFORM_ATTN",
        inputs={
            "query": TensorType(("B", "Q", "D")),
            "reference_points": TensorType(("B", "Q", "L", "R")),
            "memory": TensorType(("B", "S", "D")),
        },
        outputs=["out"],
        metadata={"level": "sentence", "source_examples": ["MSDeformAttnFast"]},
    )
    g.add(Node("value", "AOI::LINEAR", ["memory"]))
    g.add(Node("offsets", "AOI::LINEAR", ["query"]))
    g.add(Node("weight_logits", "AOI::LINEAR", ["query"]))
    g.add(Node("weights", "AOI::SOFTMAX", ["weight_logits"]))
    g.add(Node("locations", "AOI::DEFORMABLE_SAMPLE_LOCATIONS", ["reference_points", "offsets"]))
    g.add(Node("sampled", "AOI::BILINEAR_SAMPLER", ["value", "locations"]))
    g.add(Node("mixed", "AOI::WEIGHTED_SUM", ["sampled", "weights"]))
    g.add(Node("out", "AOI::LINEAR", ["mixed"], output_type=TensorType(("B", "Q", "D"))))
    g.validate()
    return g


def detection_mlp_graph() -> Graph:
    g = Graph(
        name="DETECTION_MLP",
        inputs={"x": TensorType(("...", "DIN"))},
        outputs=["out"],
        metadata={"level": "sentence", "source_examples": ["DetectionMLP"]},
    )
    g.add(Node("stack", "AOI::REPEATED_LINEAR_RELU", ["x"], {"last_activation": False}))
    g.add(Node("out", "IDENTITY", ["stack"], output_type=TensorType(("...", "DOUT"))))
    g.validate()
    return g


def detr_decoder_layer_graph() -> Graph:
    x = TensorType(("B", "Q", "D"))
    g = Graph(
        name="DETR_DECODER_LAYER",
        inputs={"target": x, "memory": TensorType(("B", "S", "D")), "reference_points": TensorType(("B", "Q", "L", "4"))},
        outputs=["out"],
        metadata={"level": "paragraph", "source_examples": ["TransformerDecoderLayer"]},
    )
    g.add(Node("self_attn", "AOI::MULTIHEAD_SELF_ATTENTION", ["target"]))
    g.add(Node("r1", "ADD", ["target", "self_attn"]))
    g.add(Node("n1", "AOI::LAYERNORM", ["r1"]))
    g.add(Node("cross", "AOI::MS_DEFORM_ATTN", ["n1", "reference_points", "memory"]))
    g.add(Node("r2", "ADD", ["n1", "cross"]))
    g.add(Node("n2", "AOI::LAYERNORM", ["r2"]))
    g.add(Node("ffn", "AOI::RELU_FFN", ["n2"]))
    g.add(Node("r3", "ADD", ["n2", "ffn"]))
    g.add(Node("out", "AOI::LAYERNORM", ["r3"], output_type=x))
    g.validate()
    return g


def detr_decoder_graph() -> Graph:
    g = Graph(
        name="DETR_DECODER",
        inputs={
            "target": TensorType(("B", "Q", "D")),
            "memory": TensorType(("B", "S", "D")),
            "reference_points": TensorType(("B", "Q", "4")),
        },
        outputs=["hidden", "references"],
        metadata={"level": "chapter", "source_examples": ["TransformerDecoder"]},
    )
    g.add(Node("query_pos", "AOI::SINE_POSITION_EMBED_AND_MLP", ["reference_points"]))
    g.add(Node("hidden", "AOI::REPEATED_DETR_DECODER_LAYER", ["target", "memory", "reference_points", "query_pos"]))
    g.add(Node("references", "AOI::REFERENCE_POINT_REFINEMENT", ["reference_points", "hidden"]))
    g.validate()
    return g


def hybrid_encoder_graph() -> Graph:
    g = Graph(
        name="HYBRID_ENCODER",
        inputs={"features": TensorType(("LEVELS", "B", "C", "H", "W"))},
        outputs=["memory", "two_stage_boxes", "two_stage_memory"],
        metadata={"level": "paragraph", "source_examples": ["HybridEncoder"]},
    )
    g.add(Node("memory", "AOI::FLATTEN_AND_CONCAT_FEATURE_LEVELS", ["features"]))
    g.add(Node("proposals", "AOI::ENCODER_PROPOSALS", ["memory"]))
    g.add(Node("two_stage_memory", "AOI::TOPK_PROPOSAL_MEMORY", ["memory", "proposals"]))
    g.add(Node("two_stage_boxes", "AOI::TOPK_PROPOSAL_BOXES", ["proposals"]))
    g.validate()
    return g


def rfdetr_graph() -> Graph:
    g = Graph(
        name="RF_DETR",
        inputs={"image": TensorType(("B", "3", "H", "W"))},
        outputs=["pred_logits", "pred_boxes"],
        metadata={
            "level": "novel",
            "source_examples": ["RFDetr"],
            "reference_model": "reference_models/rfdetr",
        },
    )
    g.add(Node("features", "AOI::DINO_V2_BACKBONE", ["image"]))
    g.add(Node("projected", "AOI::INPUT_PROJECTIONS", ["features"]))
    g.add(Node("encoded", "AOI::HYBRID_ENCODER", ["projected"]))
    g.add(Node("queries", "AOI::QUERY_AND_REFERENCE_EMBEDDINGS", ["encoded"]))
    g.add(Node("decoded", "AOI::DETR_DECODER", ["queries", "encoded"]))
    g.add(Node("pred_logits", "AOI::CLASSIFICATION_HEAD", ["decoded"]))
    g.add(Node("pred_boxes", "AOI::BOX_REGRESSION_HEAD", ["decoded"]))
    g.validate()
    return g

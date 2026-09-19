from __future__ import annotations

from ..ir import Graph, Node, TensorType


def box_iou_graph() -> Graph:
    g = Graph(
        name="BOX_IOU",
        inputs={"boxes1": TensorType(("N", "4")), "boxes2": TensorType(("M", "4"))},
        outputs=["iou"],
        metadata={"level": "word", "source_examples": ["box_iou", "generalized_box_iou"]},
    )
    g.add(Node("intersection", "AOI::BOX_INTERSECTION", ["boxes1", "boxes2"]))
    g.add(Node("union", "AOI::BOX_UNION", ["boxes1", "boxes2", "intersection"]))
    g.add(Node("iou", "DIV", ["intersection", "union"], output_type=TensorType(("N", "M"))))
    g.validate()
    return g


def hungarian_matcher_graph() -> Graph:
    g = Graph(
        name="HUNGARIAN_MATCHER",
        inputs={
            "pred_logits": TensorType(("B", "Q", "C")),
            "pred_boxes": TensorType(("B", "Q", "4")),
            "target_labels": TensorType(("T",)),
            "target_boxes": TensorType(("T", "4")),
        },
        outputs=["assignment"],
        metadata={
            "level": "sentence",
            "source_examples": ["HungarianMatcher"],
            "execution_domain": "CPU/no-grad in current RF-DETR training code",
        },
    )
    g.add(Node("class_cost", "AOI::FOCAL_CLASS_COST", ["pred_logits", "target_labels"]))
    g.add(Node("bbox_cost", "AOI::L1_BOX_COST", ["pred_boxes", "target_boxes"]))
    g.add(Node("giou_cost", "AOI::GIOU_COST", ["pred_boxes", "target_boxes"]))
    g.add(Node("cost", "AOI::WEIGHTED_SUM", ["class_cost", "bbox_cost", "giou_cost"]))
    g.add(Node("assignment", "HUNGARIAN_SOLVE", ["cost"]))
    g.validate()
    return g


def detr_set_criterion_graph() -> Graph:
    g = Graph(
        name="DETR_SET_CRITERION",
        inputs={
            "pred_logits": TensorType(("B", "Q", "C")),
            "pred_boxes": TensorType(("B", "Q", "4")),
            "targets": TensorType(("TARGET_SET",)),
        },
        outputs=["loss_total"],
        metadata={"level": "paragraph", "source_examples": ["SetCriterion"]},
    )
    g.add(Node("match", "AOI::HUNGARIAN_MATCHER", ["pred_logits", "pred_boxes", "targets"]))
    g.add(Node("cls", "AOI::IA_BCE_OR_FOCAL_CLASSIFICATION_LOSS", ["pred_logits", "targets", "match"]))
    g.add(Node("l1", "AOI::L1_BOX_LOSS", ["pred_boxes", "targets", "match"]))
    g.add(Node("giou", "AOI::GIOU_LOSS", ["pred_boxes", "targets", "match"]))
    g.add(Node("loss_total", "AOI::WEIGHTED_SUM", ["cls", "l1", "giou"], output_type=TensorType(())))
    g.validate()
    return g

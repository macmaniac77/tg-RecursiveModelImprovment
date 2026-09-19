"""RF-DETR Nano on TinyGrad — single-file trainer.

Pairs with detect.py + rfdetr_nano_tinygrad.safetensors.

  python train.py --mode smoke --steps 5
  python train.py --mode finetune --split val --limit 16 --steps 20
  python train.py --mode pretrain --split train --steps 5000 --shuffle

--freeze-encoder is a weak-GPU memory knob, not full training.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import detect  # model + official preprocess; also puts vendored tinygrad on sys.path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tinygrad import Device, Tensor, nn
from tinygrad.helpers import getenv, trange
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save

# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    w = w.clamp(min_=0.0)
    h = h.clamp(min_=0.0)
    return Tensor.stack(x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h, dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min_=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min_=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = boxes1[..., None, :2].maximum(boxes2[..., :2])
    rb = boxes1[..., None, 2:].minimum(boxes2[..., 2:])
    wh = (rb - lt).clamp(min_=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, union = box_iou(boxes1, boxes2)
    lt = boxes1[..., None, :2].minimum(boxes2[..., :2])
    rb = boxes1[..., None, 2:].maximum(boxes2[..., 2:])
    wh = (rb - lt).clamp(min_=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area


# ---------------------------------------------------------------------------
# Hungarian matcher (CPU, no grad)
# ---------------------------------------------------------------------------


class HungarianMatcher:
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    def __call__(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        *,
        group_detr: int = 1,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        bs, num_queries = pred_logits.shape[0], pred_logits.shape[1]

        flat_logits = pred_logits.reshape(bs * num_queries, -1)
        out_prob = flat_logits.sigmoid()
        out_bbox = pred_boxes.reshape(bs * num_queries, 4)

        tgt_ids = Tensor.cat(*[t["labels"] for t in targets], dim=0)
        tgt_bbox = Tensor.cat(*[t["boxes"] for t in targets], dim=0)

        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        alpha, gamma = 0.25, 2.0
        neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(-flat_logits).logsigmoid())
        pos_cost = alpha * ((1 - out_prob) ** gamma) * (-flat_logits.logsigmoid())
        tgt_ids_np = tgt_ids.numpy().astype(np.int64)
        cost_class = Tensor(pos_cost.numpy()[:, tgt_ids_np] - neg_cost.numpy()[:, tgt_ids_np])
        cost_bbox = (out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C_np = C.numpy().astype(np.float64).reshape(bs, num_queries, -1)
        max_cost = C_np.max() if C_np.size > 0 else 0.0
        C_np[np.isinf(C_np) | np.isnan(C_np)] = max_cost * 2

        sizes = [int(t["boxes"].shape[0]) for t in targets]
        g_num_queries = num_queries // group_detr
        C_list = [C_np[:, i * g_num_queries : (i + 1) * g_num_queries, :] for i in range(group_detr)]

        indices: list[tuple[np.ndarray, np.ndarray]] = []
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            chunks = np.split(C_g, np.cumsum(sizes)[:-1], axis=-1) if len(sizes) > 1 else [C_g]
            indices_g = [linear_sum_assignment(chunks[b][b]) for b in range(bs)]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([a[0], b[0] + g_num_queries * g_i]), np.concatenate([a[1], b[1]]))
                    for a, b in zip(indices, indices_g)
                ]
        return indices


# ---------------------------------------------------------------------------
# SetCriterion
# ---------------------------------------------------------------------------


def build_weight_dict(*, num_decoder_layers: int = 2, two_stage: bool = True) -> dict[str, float]:
    w = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    for i in range(num_decoder_layers - 1):
        for k, v in list(w.items()):
            w[f"{k}_{i}"] = v
    if two_stage:
        for k, v in list({"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}.items()):
            w[f"{k}_enc"] = v
    return w


def _indices_to_idx(indices: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    batch_idx = np.concatenate([np.full_like(src, i, dtype=np.int64) for i, (src, _) in enumerate(indices)])
    src_idx = np.concatenate([src.astype(np.int64) for (src, _) in indices])
    return batch_idx, src_idx


def _gather_matched(tensor: Tensor, b_idx: np.ndarray, s_idx: np.ndarray) -> Tensor:
    if b_idx.size == 0:
        return tensor.reshape(0, *tensor.shape[2:])
    rows = [tensor[int(b), int(s)] for b, s in zip(b_idx, s_idx)]
    return rows[0] if len(rows) == 1 else Tensor.stack(rows, dim=0)


class SetCriterion:
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        *,
        focal_alpha: float = 0.25,
        losses: tuple[str, ...] = ("labels", "boxes", "cardinality"),
        group_detr: int = 1,
        ia_bce_loss: bool = True,
        sum_group_losses: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.group_detr = group_detr
        self.ia_bce_loss = ia_bce_loss
        self.sum_group_losses = sum_group_losses

    def _num_boxes(self, targets: list[dict[str, Tensor]]) -> float:
        n = float(sum(int(t["labels"].shape[0]) for t in targets))
        if not self.sum_group_losses:
            n *= self.group_detr
        return max(n, 1.0)

    def _loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[np.ndarray, np.ndarray]],
        num_boxes: float,
        *,
        log: bool = True,
    ) -> dict[str, Tensor]:
        src_logits = outputs["pred_logits"]
        b_idx, s_idx = _indices_to_idx(indices)
        target_classes_o = np.concatenate(
            [t["labels"].numpy()[j].astype(np.int64) for t, (_, j) in zip(targets, indices)]
        )

        alpha, gamma = self.focal_alpha, 2.0
        pred_boxes = outputs["pred_boxes"]
        src_boxes_t = _gather_matched(pred_boxes, b_idx, s_idx)
        tgt_boxes = Tensor.cat(
            *[t["boxes"][Tensor(j.astype(np.int32))] for t, (_, j) in zip(targets, indices)], dim=0
        )
        iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes_t.detach()), box_cxcywh_to_xyxy(tgt_boxes))
        pos_ious = Tensor(np.diag(iou.numpy())).detach()
        prob = src_logits.sigmoid()
        neg_weights = prob ** gamma
        pw_np = np.zeros(src_logits.shape, dtype=np.float32)
        nw_np = neg_weights.numpy().copy()
        prob_np = prob.numpy()
        for bi, si, cls, piou in zip(b_idx, s_idx, target_classes_o, pos_ious.numpy()):
            t = max((prob_np[bi, si, cls] ** alpha) * (piou ** (1 - alpha)), 0.01)
            pw_np[bi, si, cls] = t
            nw_np[bi, si, cls] = 1.0 - t
        pos_weights = Tensor(pw_np)
        neg_weights = Tensor(nw_np)
        loss_ce = (neg_weights * src_logits - src_logits.logsigmoid() * (pos_weights + neg_weights)).sum() / num_boxes

        out: dict[str, Tensor | float] = {"loss_ce": loss_ce}
        if log and len(target_classes_o) > 0:
            pred_cls = src_logits.numpy()[b_idx, s_idx].argmax(-1)
            acc = float((pred_cls == target_classes_o).mean() * 100.0)
            out["class_error"] = 100.0 - acc
        return out

    def _loss_boxes(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[np.ndarray, np.ndarray]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        b_idx, s_idx = _indices_to_idx(indices)
        src_boxes = _gather_matched(outputs["pred_boxes"], b_idx, s_idx)
        tgt_boxes = Tensor.cat(
            *[t["boxes"][Tensor(j.astype(np.int32))] for t, (_, j) in zip(targets, indices)], dim=0
        )
        loss_bbox = (src_boxes - tgt_boxes).abs().sum() / num_boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        loss_giou = (1.0 - Tensor(np.diag(giou.numpy()))).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _loss_cardinality(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, Tensor]:
        pred_logits = outputs["pred_logits"]
        tgt_lengths = np.array([int(t["labels"].shape[0]) for t in targets], dtype=np.float32)
        card_pred = (pred_logits.argmax(-1).numpy() != pred_logits.shape[-1] - 1).sum(1).astype(np.float32)
        return {"cardinality_error": float(np.abs(card_pred - tgt_lengths).mean())}

    def _forward_one(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        *,
        suffix: str = "",
        log_labels: bool = True,
    ) -> dict[str, Tensor]:
        indices = self.matcher(outputs, targets, group_detr=self.group_detr)
        num_boxes = self._num_boxes(targets)
        losses: dict[str, Tensor] = {}
        for loss in self.losses:
            if loss == "labels":
                d = self._loss_labels(outputs, targets, indices, num_boxes, log=log_labels)
            elif loss == "boxes":
                d = self._loss_boxes(outputs, targets, indices, num_boxes)
            elif loss == "cardinality":
                d = self._loss_cardinality(outputs, targets, indices)
            else:
                continue
            for k, v in d.items():
                losses[k + suffix] = v
        return losses

    def __call__(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        out_wo_aux = {k: v for k, v in outputs.items() if k not in ("aux_outputs", "enc_outputs")}
        losses = self._forward_one(out_wo_aux, targets, log_labels=True)
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                losses.update(self._forward_one(aux, targets, suffix=f"_{i}", log_labels=False))
        if "enc_outputs" in outputs:
            losses.update(self._forward_one(outputs["enc_outputs"], targets, suffix="_enc", log_labels=False))
        return losses


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def _load_coco_json(ann_path: Path) -> dict[str, Any]:
    with open(ann_path) as f:
        return json.load(f)


def load_coco_sample(
    image_path: Path,
    ann_path: Path,
    image_id: int | None = None,
    *,
    coco: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict]:
    img = Image.open(image_path).convert("RGB")
    pre, _meta = detect.preprocess_official_exact(img)
    pre = pre[0] if pre.ndim == 4 else pre
    data = coco if coco is not None else _load_coco_json(ann_path)
    img_id = image_id or int(image_path.stem)
    anns = [a for a in data["annotations"] if a["image_id"] == img_id and not a.get("iscrowd", 0)]
    info = next(i for i in data["images"] if i["id"] == img_id)
    w, h = info["width"], info["height"]
    boxes, labels = [], []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        boxes.append([(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])
        labels.append(a["category_id"])
    if not boxes:
        boxes = [[0.5, 0.5, 0.1, 0.1]]
        labels = [1]
    target = {"labels": np.array(labels, dtype=np.int64), "boxes": np.array(boxes, dtype=np.float32)}
    return pre, target


def iter_coco_split(
    img_dir: Path,
    ann_path: Path,
    *,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    skip_empty: bool = False,
) -> Iterator[tuple[Path, np.ndarray, dict]]:
    coco = _load_coco_json(ann_path)
    id_to_file = {i["id"]: i["file_name"] for i in coco["images"]}
    ids = sorted(id_to_file.keys())
    if shuffle:
        rng = np.random.default_rng(seed)
        ids = list(rng.permutation(ids))
    if limit is not None:
        ids = ids[: int(limit)]

    anns_by_id: dict[int, list] = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns_by_id.setdefault(a["image_id"], []).append(a)

    for img_id in ids:
        fname = id_to_file[img_id]
        p = img_dir / fname
        if not p.exists():
            alt = img_dir / f"{img_id:012d}.jpg"
            if not alt.exists():
                continue
            p = alt
        if skip_empty and not anns_by_id.get(img_id):
            continue
        pre, tgt = load_coco_sample(p, ann_path, image_id=img_id, coco=coco)
        yield p, pre, tgt


# ---------------------------------------------------------------------------
# Train helpers
# ---------------------------------------------------------------------------

_ENCODER_PREFIX = "backbone.0.encoder.encoder."


def compute_multi_scale_scales(
    resolution: int,
    *,
    expanded_scales: bool = True,
    patch_size: int = 16,
    num_windows: int = 2,
) -> list[int]:
    base = resolution // (patch_size * num_windows)
    offsets = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] if expanded_scales else [-3, -2, -1, 0, 1, 2, 3, 4]
    min_size = patch_size * num_windows * 2
    scales = [s * patch_size * num_windows for s in (base + o for o in offsets)]
    return [s for s in scales if s >= min_size]


def apply_multi_scale(pre: np.ndarray, step: int, *, resolution: int = 384) -> np.ndarray:
    scales = compute_multi_scale_scales(resolution)
    rng = np.random.default_rng(step)
    scale = int(rng.choice(scales))
    if pre.ndim == 4:
        return np.stack([detect.resize_bilinear2d_aa(pre[i], scale, scale) for i in range(pre.shape[0])], axis=0)
    return detect.resize_bilinear2d_aa(pre, scale, scale)


def drop_scheduler(drop_rate: float, epochs: int, niter_per_ep: int) -> np.ndarray:
    return np.full(epochs * niter_per_ep, drop_rate)


def optimizer_schedule_step(opt: Any) -> None:
    for p in opt.params:
        if p.grad is None:
            p.grad = Tensor.zeros_like(p)
    opt.schedule_step()


def get_trainable_parameters(model: Any, *, freeze_encoder: bool = False) -> list[Tensor]:
    sd = get_state_dict(model)
    if not freeze_encoder:
        return list(sd.values())
    return [t for k, t in sd.items() if not k.startswith(_ENCODER_PREFIX)]


class ModelEma:
    def __init__(self, model: Any, decay: float = 0.993, tau: int = 0) -> None:
        self.decay = decay
        self.tau = tau
        self.updates = 1
        self._shadow: dict[str, np.ndarray] = {k: v.numpy().copy() for k, v in get_state_dict(model).items()}

    def _current_decay(self) -> float:
        if self.tau == 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    def update(self, model: Any) -> None:
        decay = self._current_decay()
        for k, t in get_state_dict(model).items():
            self._shadow[k] = decay * self._shadow[k] + (1.0 - decay) * t.numpy()
        self.updates += 1

    def state_dict_tensors(self) -> dict[str, Tensor]:
        return {k: Tensor(v) for k, v in self._shadow.items()}


def _targets_from_numpy(tgt: dict) -> list[dict]:
    return [{"labels": Tensor(tgt["labels"]), "boxes": Tensor(tgt["boxes"])}]


def _count_params(params: list) -> int:
    n = 0
    for p in params:
        try:
            n += int(np.prod(p.shape))
        except Exception:
            pass
    return n


def _resolve_coco_paths(coco_dir: Path, split: str) -> tuple[Path, Path]:
    split = split.lower()
    if split not in ("train", "val"):
        raise ValueError(f"split must be train|val, got {split!r}")
    return coco_dir / f"{split}2017", coco_dir / "annotations" / f"instances_{split}2017.json"


def _build_model(*, phase_b: bool, group_detr: int) -> detect.RFDetr:
    if phase_b:
        return detect.RFDetr(num_queries=3900, num_classes=91, group_detr=13, num_layers=2)
    return detect.RFDetr(num_queries=300, num_classes=91, group_detr=group_detr, num_layers=2)


def train_steps(
    model: detect.RFDetr,
    criterion: SetCriterion,
    weight_dict: dict[str, float],
    opt: nn.optim.Optimizer,
    sample_iter,
    *,
    steps: int,
    num_groups: int,
    multi_scale: bool,
    grad_accum: int,
    drop_path_schedule: np.ndarray | None,
    ema: ModelEma | None,
    log_every: int,
) -> list[float]:
    losses: list[float] = []
    accum = max(1, grad_accum)
    t0 = time.perf_counter()
    cycle = itertools.cycle(list(sample_iter))

    for step in trange(steps):
        if drop_path_schedule is not None and step < len(drop_path_schedule):
            model.update_drop_path(float(drop_path_schedule[step]))

        path, pre, tgt_np = next(cycle)
        x_base = pre if pre.ndim == 4 else pre[np.newaxis, ...]
        batch = apply_multi_scale(x_base, step) if multi_scale else x_base
        targets = _targets_from_numpy(tgt_np)

        with Tensor.train():
            if step % accum == 0:
                opt.zero_grad()
            out = model.forward_train(Tensor(batch), num_groups=num_groups)
            loss_dict = criterion(out, targets)
            total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            (total / accum).backward()
            if (step + 1) % accum == 0:
                optimizer_schedule_step(opt)
                if ema is not None:
                    ema.update(model)

        loss_val = float(total.numpy().item())
        losses.append(loss_val)
        if step % max(1, log_every) == 0:
            name = path.name if hasattr(path, "name") else str(path)
            print(f"step {step:5d} loss={loss_val:.4f} img={name}")

    elapsed = time.perf_counter() - t0
    print(
        f"device={Device.DEFAULT} steps={steps} step_s_mean={elapsed / max(steps, 1):.2f}s "
        f"loss={losses[-1]:.4f}"
    )
    return losses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_COCO = HERE / "data" / "coco"
DEFAULT_IMG = DEFAULT_COCO / "val2017" / "000000000139.jpg"
DEFAULT_ANN = DEFAULT_COCO / "annotations" / "instances_val2017.json"


def main() -> None:
    os.environ.setdefault("TG_SPEED_MODE", "0")
    ap = argparse.ArgumentParser(description="RF-DETR Nano TinyGrad trainer (uses detect.py).")
    ap.add_argument("--mode", choices=("smoke", "finetune", "pretrain"), default=getenv("RFDETR_TRAIN_MODE", "smoke"))
    ap.add_argument("image", nargs="?", type=Path, default=None, help="Smoke-mode image")
    ap.add_argument("--ann", type=Path, default=None)
    ap.add_argument("--coco-dir", type=Path, default=DEFAULT_COCO)
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=float(getenv("RFDETR_LR", 1e-4)))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--group-detr", type=int, default=1)
    ap.add_argument("--multi-scale", action="store_true")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.993)
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--freeze-encoder", action="store_true", help="Weak-GPU knob: skip DINO in AdamW")
    ap.add_argument("--full", action="store_true", help="Force full-param training")
    ap.add_argument("--grad-accum-steps", type=int, default=int(getenv("RFDETR_GRAD_ACCUM", 1)))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=0)
    args = ap.parse_args()

    phase_b = args.mode == "pretrain"
    if phase_b:
        args.group_detr = 13
        args.use_ema = True
        args.multi_scale = True
        if args.steps is None:
            args.steps = int(getenv("RFDETR_TRAIN_STEPS", 500))
        default_out = HERE / "rfdetr_nano_pretrain.safetensors"
    elif args.mode == "finetune":
        if args.steps is None:
            args.steps = int(getenv("RFDETR_TRAIN_STEPS", 100))
        default_out = HERE / "rfdetr_nano_finetuned.safetensors"
    else:
        if args.steps is None:
            args.steps = int(getenv("RFDETR_TRAIN_STEPS", getenv("STEPS", 20)))
        default_out = HERE / "rfdetr_nano_finetuned.safetensors"

    if args.full:
        args.freeze_encoder = False
    freeze = bool(args.freeze_encoder)

    samples: list[tuple[Path, np.ndarray, dict]] = []
    if args.mode == "smoke":
        image = args.image or DEFAULT_IMG
        ann = args.ann or DEFAULT_ANN
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        if not ann.exists():
            raise FileNotFoundError(f"Annotations not found: {ann}")
        pre, tgt = load_coco_sample(image, ann)
        samples = [(image, pre, tgt)]
        print(f"mode=smoke image={image.name} objects={len(tgt['labels'])}")
    else:
        img_dir, ann = _resolve_coco_paths(args.coco_dir, args.split)
        if args.ann is not None:
            ann = args.ann
        if not img_dir.exists():
            raise FileNotFoundError(f"COCO images missing: {img_dir}")
        if not ann.exists():
            raise FileNotFoundError(f"Annotations missing: {ann}")
        for item in iter_coco_split(
            img_dir, ann, limit=args.limit, shuffle=args.shuffle, seed=args.seed, skip_empty=True
        ):
            samples.append(item)
        if not samples:
            raise RuntimeError(f"No images found under {img_dir} for {ann}")
        print(f"mode={args.mode} split={args.split} images={len(samples)} coco={args.coco_dir}")

    weights = detect.find_weights(args.weights)
    model = _build_model(phase_b=phase_b, group_detr=args.group_detr)
    model.load_weights(weights, strict=False, verbose=False)

    matcher = HungarianMatcher()
    weight_dict = build_weight_dict(num_decoder_layers=2)
    criterion = SetCriterion(91, matcher, weight_dict, group_detr=13, ia_bce_loss=True)
    params = get_trainable_parameters(model, freeze_encoder=freeze)
    print(
        f"device={Device.DEFAULT} freeze_encoder={freeze} trainable_tensors={len(params)} "
        f"approx_params={_count_params(params):,} lr={args.lr} steps={args.steps} weights={weights.name}"
    )
    if freeze:
        print("NOTE: --freeze-encoder active — DINO backbone not updated.")
    else:
        print("FULL training: all loaded parameters in AdamW.")

    opt = nn.optim.AdamW(params, lr=args.lr)
    ema = ModelEma(model, decay=args.ema_decay) if args.use_ema else None
    dp_schedule = drop_scheduler(args.drop_path, args.epochs, max(1, args.steps // max(args.epochs, 1))) if args.drop_path > 0 else None
    log_every = args.log_every if args.log_every > 0 else max(1, args.steps // 5)
    fwd_groups = 13 if phase_b else args.group_detr

    train_steps(
        model,
        criterion,
        weight_dict,
        opt,
        samples,
        steps=args.steps,
        num_groups=fwd_groups,
        multi_scale=args.multi_scale,
        grad_accum=args.grad_accum_steps,
        drop_path_schedule=dp_schedule,
        ema=ema,
        log_every=log_every,
    )

    out_path = args.out or default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_save(nn.state.get_state_dict(model), str(out_path))
    print(f"saved {out_path}")
    if ema is not None:
        ema_path = out_path.with_name(out_path.stem + "_ema.safetensors")
        safe_save(ema.state_dict_tensors(), str(ema_path))
        print(f"saved {ema_path}")


if __name__ == "__main__":
    main()

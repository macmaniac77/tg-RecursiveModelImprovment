"""RF-DETR Nano on TinyGrad — single-file detector.

Usage:
  python detect.py image.jpg
  python detect.py image.jpg -o overlay.jpg
  python detect.py image.jpg --weights rfdetr_nano_tinygrad.safetensors

Pair this file with rfdetr_nano_tinygrad.safetensors. TinyGrad is the only
runtime (pip install tinygrad, or a vendored tinygrad/ next to this file).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

HERE = Path(__file__).resolve().parent
for _tg in (HERE / "tinygrad", HERE.parent / "tinygrad"):
    if (_tg / "tinygrad").is_dir():
        sys.path.insert(0, str(_tg))
        break

import cv2
import numpy as np
from PIL import Image
from tinygrad import Device, Tensor, TinyJit, nn, dtypes
from tinygrad.helpers import getenv, polyN
from tinygrad.nn.state import load_state_dict as tiny_load_state_dict, safe_load

COCO_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
COCO_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ImageInput = Union[Path, str, Image.Image, np.ndarray]


def _aa_filter(x: float) -> float:
    x = abs(x)
    return 1.0 - x if x < 1.0 else 0.0


def _compute_1d_aa_weights(
    input_size: int, output_size: int, *, interp_size: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    scale = float(input_size) / float(output_size)
    support = (interp_size * 0.5 * scale) if scale >= 1.0 else (interp_size * 0.5)
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    max_interp = int(math.ceil(support)) * 2 + 1

    indices: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    for i in range(output_size):
        center = scale * (i + 0.5)
        xmin = max(int(math.floor(center - support + 0.5)), 0)
        xmax = min(int(math.floor(center + support + 0.5)), input_size)
        xsize = min(max(xmax - xmin, 0), max_interp)
        w = np.zeros(max_interp, dtype=np.float64)
        total = 0.0
        for j in range(xsize):
            wt = _aa_filter((j + xmin - center + 0.5) * invscale)
            w[j] = wt
            total += wt
        if total > 0.0:
            w[:xsize] /= total
        idx = np.arange(xmin, xmin + xsize, dtype=np.int64)
        indices.append(idx)
        weights.append(w[:xsize])
    return indices, weights


def _resize_1d_aa(signal: np.ndarray, output_size: int) -> np.ndarray:
    input_size = int(signal.shape[0])
    if input_size == output_size:
        return signal.astype(np.float32, copy=True)
    indices, weights = _compute_1d_aa_weights(input_size, output_size)
    out = np.empty(output_size, dtype=np.float64)
    sig = signal.astype(np.float64, copy=False)
    for i, (idx, w) in enumerate(zip(indices, weights)):
        out[i] = np.dot(sig[idx], w.astype(np.float64))
    return out.astype(np.float32)


def resize_bilinear2d_aa(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize (C, H, W) float32 with separable antialiased bilinear."""
    chw = np.asarray(chw, dtype=np.float32)
    if chw.ndim != 3:
        raise ValueError(f"expected CHW, got shape {chw.shape}")
    c, h, w = chw.shape
    if h == out_h and w == out_w:
        return chw.copy()

    mid = np.empty((c, h, out_w), dtype=np.float32)
    for ci in range(c):
        for yi in range(h):
            mid[ci, yi, :] = _resize_1d_aa(chw[ci, yi, :], out_w)

    out = np.empty((c, out_h, out_w), dtype=np.float32)
    for ci in range(c):
        for xi in range(out_w):
            out[ci, :, xi] = _resize_1d_aa(mid[ci, :, xi], out_h)
    return out


def _load_rgb_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def preprocess_official_exact(
    image: ImageInput,
    size: int = 384,
) -> Tuple[np.ndarray, dict]:
    """Official RFDETR.predict() preproc: direct square resize, ImageNet normalize."""
    pil = _load_rgb_pil(image)
    rgb = np.array(pil, dtype=np.uint8)
    orig_h, orig_w = int(rgb.shape[0]), int(rgb.shape[1])

    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    chw = resize_bilinear2d_aa(chw, size, size)

    means = COCO_MEANS.reshape(3, 1, 1)
    stds = COCO_STDS.reshape(3, 1, 1)
    chw = (chw - means) / stds
    batch = chw[np.newaxis, ...].astype(np.float32)

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "resolution": int(size),
        "input_size": (int(size), int(size)),
        "pad": (0, 0),
        "scale": float(size) / float(max(orig_w, 1)),
        "preproc": "official_exact_direct_resize_no_letterbox_numpy",
    }
    return batch, meta


def tensor_from_batch(batch: np.ndarray):
    return Tensor(np.asarray(batch, dtype=np.float32))


def speed_mode() -> bool:
  """TG_SPEED_MODE=1 strips SCHED mid-graph realizes on eval (two-stage + decoder intermediates)."""
  return os.environ.get("TG_SPEED_MODE", "0") == "1"


def _eval_speed_lazy() -> bool:
  """Lazy SCHED graph only for inference speed mode — not under Tensor.training.

  Training keeps layer/MSDA realizes so OpenCL kernels stay under bracket-depth limits
  while autograd still flows through realized buffers.
  """
  return speed_mode() and not Tensor.training


def _sync(t: Tensor) -> Tensor:
  """SCHED sync — skip under TG_SPEED_MODE eval; always realize in train / default eval."""
  if isinstance(t, Tensor) and not _eval_speed_lazy():
    return t.realize()
  return t


def _sync_two_stage(t: Tensor) -> Tensor:
  """Encoder two-stage mid-loop — strip under TG_SPEED_MODE (parity-validated) or train."""
  if isinstance(t, Tensor) and not (speed_mode() or Tensor.training):
    return t.realize()
  return t


def _sync_msda(t: Tensor) -> Tensor:
  """MSDA gather/split materialization — always realize (CL rangeify / kernel sizing)."""
  return t.realize() if isinstance(t, Tensor) else t


def _sync_parity(t: Tensor) -> Tensor:
  """Always realize — historical numerical parity (e.g. sine frequency table)."""
  return t.realize() if isinstance(t, Tensor) else t


def topk_pytorch_compat(scores: Tensor, k: int, dim: int = -1) -> Tensor:
  """Top-k with PyTorch tie-breaking: higher flat index wins on equal scores."""
  n = scores.shape[dim]
  idx_range = Tensor.arange(n, dtype=scores.dtype)
  while idx_range.ndim < scores.ndim:
    idx_range = idx_range.unsqueeze(0)
  _, topk_idx = (scores + idx_range * 1e-9).topk(k, dim=dim)
  return topk_idx.cast(dtypes.int32)


_DECODER_MAP = {
  "self_attn.in_proj_weight": "sa.weight",
  "self_attn.in_proj_bias": "sa.bias",
  "self_attn.out_proj.weight": "sa_out.weight",
  "self_attn.out_proj.bias": "sa_out.bias",
  "norm1.weight": "sa_norm.weight",
  "norm1.bias": "sa_norm.bias",
  "norm2.weight": "ca_norm.weight",
  "norm2.bias": "ca_norm.bias",
  "norm3.weight": "ffn_norm.weight",
  "norm3.bias": "ffn_norm.bias",
  "linear1.weight": "linear1.weight",
  "linear1.bias": "linear1.bias",
  "linear2.weight": "linear2.weight",
  "linear2.bias": "linear2.bias",
}


def _remap_key(key: str) -> Optional[str]:
  dec_prefix = "transformer.decoder.layers."
  if key.startswith(dec_prefix):
    rest = key[len(dec_prefix):]
    layer_idx, remainder = rest.split(".", 1)
    if remainder.startswith("cross_attn."):
      suffix = remainder.split(".", 1)[1]
      return f"decoder.layers.{layer_idx}.cross_attn.{suffix}"
    if remainder in _DECODER_MAP:
      return f"decoder.layers.{layer_idx}.{_DECODER_MAP[remainder]}"
    return None
  if key.startswith("transformer.decoder.norm."):
    return "decoder.norm." + key.split("transformer.decoder.norm.", 1)[1]
  if key.startswith("transformer.decoder.ref_point_head.layers."):
    return "decoder.ref_point_head.layers." + key.split("transformer.decoder.ref_point_head.layers.", 1)[1]
  if key.startswith("transformer.decoder.bbox_embed.layers."):
    return "decoder.bbox_embed.layers." + key.split("transformer.decoder.bbox_embed.layers.", 1)[1]
  for prefix in (
    "transformer.enc_output.",
    "transformer.enc_output_norm.",
    "transformer.enc_out_bbox_embed.",
    "transformer.enc_out_class_embed.",
  ):
    if key.startswith(prefix):
      return "encoder." + key.split("transformer.", 1)[1]
  if key.startswith("bbox_embed.layers."):
    return f"box_head.layers.{key.split('.', 2)[2]}"
  if key.startswith("class_embed."):
    return f"class_head.{key.split('.', 1)[1]}"
  return key

class IdentityDropout:
  def __init__(self, _: float) -> None:
    pass

  def __call__(self, x: Tensor) -> Tensor:
    return x


def _to_2tuple(val: int | Tuple[int, int]) -> Tuple[int, int]:
  if isinstance(val, tuple):
    return val
  return (val, val)


def _gelu_pytorch(x: Tensor) -> Tensor:
  """Exact GELU (erf), matching PyTorch / HuggingFace transformers — not tanh approx."""
  return 0.5 * x * (1 + (x * (2**-0.5)).erf())


def _half_pixel_indices(input_size: int, output_size: int, offset: float = 0.0) -> Tensor:
  if output_size == 1:
    return Tensor.zeros(output_size, dtype=dtypes.float32)
  scale = input_size / output_size
  base = Tensor.arange(output_size, dtype=dtypes.float32)
  return (base + 0.5) * scale - 0.5 + offset


def _resize_bicubic(x: Tensor, size: Tuple[int, int], coeff_a: float = -0.75, offset: float = 0.0) -> Tensor:
  assert x.ndim == 4, "Expected (N, C, H, W)"
  out_h, out_w = size
  if out_h == x.shape[-2] and out_w == x.shape[-1]:
    return x

  def _interpolate_axis(inp: Tensor, axis: int, target: int) -> Tensor:
    idx = _half_pixel_indices(inp.shape[axis], target, offset=offset)
    floor = idx.floor()
    ratio = idx - floor
    base_idx = floor.int()
    resh = [1] * inp.ndim
    resh[axis] = target
    expand = list(inp.shape)
    expand[axis] = target

    idxs = [
      (base_idx - 1).clip(0, inp.shape[axis] - 1),
      base_idx.clip(0, inp.shape[axis] - 1),
      (base_idx + 1).clip(0, inp.shape[axis] - 1),
      (base_idx + 2).clip(0, inp.shape[axis] - 1),
    ]

    def _weight(shift: Tensor, coeffs: List[float]) -> Tensor:
      return polyN(shift, coeffs).reshape(resh).expand(expand)

    ratio_view = ratio.reshape(resh)
    c0 = _weight(ratio_view + 1.0, [coeff_a, -5 * coeff_a, 8 * coeff_a, -4 * coeff_a])
    c1 = _weight(ratio_view, [coeff_a + 2, -(coeff_a + 3), 0.0, 1.0])
    c2 = _weight(1.0 - ratio_view, [coeff_a + 2, -(coeff_a + 3), 0.0, 1.0])
    c3 = _weight(2.0 - ratio_view, [coeff_a, -5 * coeff_a, 8 * coeff_a, -4 * coeff_a])

    gathered = [inp.gather(axis, idx.reshape(resh).expand(expand).int()) for idx in idxs]
    return gathered[0] * c0 + gathered[1] * c1 + gathered[2] * c2 + gathered[3] * c3

  resized = _interpolate_axis(x.cast(dtypes.float32), -2, out_h)
  resized = _interpolate_axis(resized, -1, out_w)
  return resized.cast(x.dtype)


def _window_partition(tokens: Tensor, num_windows: int, grid_hw: Tuple[int, int]) -> Tuple[Tensor, int, int]:
  if num_windows <= 1:
    return tokens, grid_hw[0], grid_hw[1]
  bsz, num_tokens, dim = tokens.shape
  h, w = grid_hw
  tokens_only = tokens[:, 1:]
  tokens_only = tokens_only.reshape(bsz, h, w, dim)
  h_win = h // num_windows
  w_win = w // num_windows
  tokens_only = (
    tokens_only.reshape(bsz, num_windows, h_win, num_windows, w_win, dim)
    .permute(0, 1, 3, 2, 4, 5)
    .reshape(bsz * (num_windows**2), h_win * w_win, dim)
  )
  cls = tokens[:, :1]
  cls = Tensor.cat(*[cls for _ in range(num_windows**2)], dim=0)
  return Tensor.cat(cls, tokens_only, dim=1), h_win, w_win


def _merge_window_patches(tokens: Tensor, batch_size: int, num_windows: int, grid_hw: Tuple[int, int]) -> Tensor:
  if num_windows <= 1:
    return tokens
  h, w = grid_hw
  h_win = h // num_windows
  w_win = w // num_windows
  num_windows_sq = num_windows**2
  # tokens shape: (batch_size * num_windows_sq, h_win * w_win, C)
  B, HW, C = tokens.shape
  # First flatten all window tokens together per original batch item
  tokens = tokens.reshape(B // num_windows_sq, num_windows_sq * HW, C)
  # Then reshape to (batch_size * num_windows, num_windows, h_win, w_win, C)
  tokens = tokens.reshape(batch_size * num_windows, num_windows, h_win, w_win, C)
  # Permute to interleave windows: (batch_size * num_windows, h_win, num_windows, w_win, C)
  tokens = tokens.permute(0, 2, 1, 3, 4)
  # Final reshape to (batch_size, h, w, C) happens in the caller
  return tokens


def _make_stage_names(num_layers: int) -> Tuple[str, ...]:
  return tuple(["stem"] + [f"stage{i}" for i in range(1, num_layers + 1)])


@dataclass
class WindowedDinov2WithRegistersConfig:
  hidden_size: int = 384
  num_hidden_layers: int = 12
  num_attention_heads: int = 6
  mlp_ratio: float = 4.0
  hidden_act: str = "gelu"
  hidden_dropout_prob: float = 0.0
  attention_probs_dropout_prob: float = 0.0
  initializer_range: float = 0.02
  layer_norm_eps: float = 1e-6
  layerscale_value: float = 1.0
  drop_path_rate: float = 0.0
  use_swiglu_ffn: bool = False
  num_register_tokens: int = 0
  num_channels: int = 3
  image_size: int = 384
  patch_size: int = 16
  qkv_bias: bool = True
  num_windows: int = 2
  window_block_indexes: Sequence[int] = field(default_factory=tuple)
  stage_names: Tuple[str, ...] = field(default_factory=tuple)
  out_features: Tuple[str, ...] = field(default_factory=tuple)
  out_feature_indexes: Tuple[int, ...] = field(default_factory=tuple)
  out_feature_channels: Tuple[int, ...] = field(default_factory=tuple)
  reshape_hidden_states: bool = True
  apply_layernorm: bool = True
  interpolate_antialias: bool = True
  interpolate_offset: float = 0.0

  def __post_init__(self) -> None:
    if not self.stage_names:
      self.stage_names = _make_stage_names(self.num_hidden_layers)
    if not self.out_feature_indexes:
      self.out_feature_indexes = tuple(range(3, self.num_hidden_layers + 1, 3))
    if not self.out_features:
      self.out_features = tuple(self.stage_names[idx] for idx in self.out_feature_indexes)
    if not self.out_feature_channels:
      self.out_feature_channels = tuple([self.hidden_size] * len(self.out_feature_indexes))


class Dinov2WithRegistersPatchEmbeddings:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    patch_size = _to_2tuple(config.patch_size)
    image_size = _to_2tuple(config.image_size)
    self.image_size = image_size
    self.patch_size = patch_size
    self.num_channels = config.num_channels
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    self.num_patches = num_patches
    self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

  def __call__(self, pixel_values: Tensor) -> Tensor:
    if pixel_values.shape[1] != self.num_channels:
      raise ValueError(f"Expected {self.num_channels} channels but received {pixel_values.shape[1]}")
    embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
    return embeddings


class WindowedDinov2WithRegistersEmbeddings:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    self.config = config
    self.cls_token = Tensor.randn(1, 1, config.hidden_size)
    self.mask_token = Tensor.zeros(1, config.hidden_size)
    self.register_tokens = (
      Tensor.zeros(1, config.num_register_tokens, config.hidden_size) if config.num_register_tokens > 0 else None
    )
    self.patch_embeddings = Dinov2WithRegistersPatchEmbeddings(config)
    num_patches = self.patch_embeddings.num_patches
    self.position_embeddings = Tensor.randn(1, num_patches + 1, config.hidden_size)
    self.dropout = IdentityDropout(config.hidden_dropout_prob)

  def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
    num_patches = embeddings.shape[1] - 1
    num_positions = self.position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
      return self.position_embeddings
    class_pos = self.position_embeddings[:, :1]
    patch_pos = self.position_embeddings[:, 1:]
    grid_h = height // self.config.patch_size
    grid_w = width // self.config.patch_size
    sqrt = int(math.sqrt(num_positions))
    patch_pos = patch_pos.reshape(1, sqrt, sqrt, -1).permute(0, 3, 1, 2)
    offset = getattr(self.config, "interpolate_offset", 0.0)
    resized = _resize_bicubic(patch_pos, (grid_h, grid_w), coeff_a=-0.75, offset=offset)
    resized = resized.permute(0, 2, 3, 1).reshape(1, -1, self.config.hidden_size)
    return Tensor.cat(class_pos, resized, dim=1)

  def __call__(self, pixel_values: Tensor, bool_masked_pos: Optional[Tensor] = None) -> Tensor:
    batch_size, _, height, width = pixel_values.shape
    embeddings = self.patch_embeddings(pixel_values)
    if bool_masked_pos is not None:
      mask = bool_masked_pos.unsqueeze(-1)
      replacement = self.mask_token.reshape(1, 1, -1).expand_as(embeddings)
      embeddings = mask.where(replacement, embeddings)
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = Tensor.cat(cls_tokens, embeddings, dim=1)
    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
    # Option B minimal fix: insert registers BEFORE window partitioning so they are part of the
    # correct token sequence. Windowing then operates on (CLS + registers + patches).
    if self.register_tokens is not None:
      regs = self.register_tokens.expand(embeddings.shape[0], -1, -1)
      embeddings = Tensor.cat(embeddings[:, :1], regs, embeddings[:, 1:], dim=1)
    if self.config.num_windows > 1:
      num_h = height // self.config.patch_size
      num_w = width // self.config.patch_size
      embeddings, _, _ = _window_partition(embeddings, self.config.num_windows, (num_h, num_w))
    return self.dropout(embeddings)


class Dinov2WithRegistersSelfAttention:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    all_head_size = config.hidden_size
    if all_head_size % config.num_attention_heads != 0:
      raise ValueError("Hidden size must be divisible by number of heads")
    self.num_attention_heads = config.num_attention_heads
    self.head_dim = all_head_size // config.num_attention_heads
    self.scale = self.head_dim ** -0.5
    self.query = nn.Linear(config.hidden_size, all_head_size, bias=config.qkv_bias)
    self.key = nn.Linear(config.hidden_size, all_head_size, bias=config.qkv_bias)
    self.value = nn.Linear(config.hidden_size, all_head_size, bias=config.qkv_bias)

  def __call__(self, hidden_states: Tensor) -> Tensor:
    bsz, seq_len, _ = hidden_states.shape
    q = self.query(hidden_states).reshape(bsz, seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
    k = self.key(hidden_states).reshape(bsz, seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
    v = self.value(hidden_states).reshape(bsz, seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(-1)
    context = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, -1)
    return context


class Dinov2WithRegistersSelfOutput:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.dropout = IdentityDropout(config.hidden_dropout_prob)

  def __call__(self, hidden_states: Tensor) -> Tensor:
    return self.dropout(self.dense(hidden_states))


class Dinov2WithRegistersAttention:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    self.attention = Dinov2WithRegistersSelfAttention(config)
    self.output = Dinov2WithRegistersSelfOutput(config)

  def __call__(self, hidden_states: Tensor) -> Tensor:
    return self.output(self.attention(hidden_states))


class Dinov2WithRegistersLayerScale:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    self.lambda1 = Tensor.ones(config.hidden_size) * config.layerscale_value

  def __call__(self, hidden_state: Tensor) -> Tensor:
    return hidden_state * self.lambda1


class Dinov2WithRegistersDropPath:
  def __init__(self, drop_prob: float) -> None:
    self.drop_prob = drop_prob

  def __call__(self, hidden_states: Tensor) -> Tensor:
    if self.drop_prob <= 0.0 or not self.training:
      return hidden_states
    keep_prob = 1.0 - self.drop_prob
    shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
    random = (Tensor.rand(shape, dtype=hidden_states.dtype) + keep_prob).floor()
    return hidden_states / keep_prob * random


class Dinov2WithRegistersMLP:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    hidden = int(config.hidden_size * config.mlp_ratio)
    self.fc1 = nn.Linear(config.hidden_size, hidden)
    self.fc2 = nn.Linear(hidden, config.hidden_size)

  def __call__(self, hidden_state: Tensor) -> Tensor:
    return self.fc2(_gelu_pytorch(self.fc1(hidden_state)))


class WindowedDinov2WithRegistersLayer:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    self.num_windows = config.num_windows
    self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention = Dinov2WithRegistersAttention(config)
    self.layer_scale1 = Dinov2WithRegistersLayerScale(config)
    self.drop_path = Dinov2WithRegistersDropPath(config.drop_path_rate)
    self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.mlp = Dinov2WithRegistersMLP(config)
    self.layer_scale2 = Dinov2WithRegistersLayerScale(config)

  def __call__(self, hidden_states: Tensor, run_full_attention: bool) -> Tensor:
    shortcut = hidden_states
    if run_full_attention and self.num_windows > 1:
      num_windows_sq = self.num_windows**2
      bsz, length, dim = hidden_states.shape
      hidden_states = hidden_states.reshape(bsz // num_windows_sq, num_windows_sq * length, dim)
    attn = self.attention(self.norm1(hidden_states))
    if run_full_attention and self.num_windows > 1:
      num_windows_sq = self.num_windows**2
      bsz, length, dim = hidden_states.shape
      attn = attn.reshape(bsz * num_windows_sq, length // num_windows_sq, dim)
    hidden_states = shortcut + self.drop_path(self.layer_scale1(attn))
    mlp = self.layer_scale2(self.mlp(self.norm2(hidden_states)))
    return hidden_states + self.drop_path(mlp)


class WindowedDinov2WithRegistersEncoder:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    self.layer = [WindowedDinov2WithRegistersLayer(config) for _ in range(config.num_hidden_layers)]
    self.window_block_indexes = set(config.window_block_indexes)
    self.num_windows = config.num_windows

  def __call__(self, hidden_states: Tensor) -> List[Tensor]:
    all_hidden_states = []
    for idx, layer_module in enumerate(self.layer):
      all_hidden_states.append(hidden_states)
      run_full_attention = idx not in self.window_block_indexes
      hidden_states = layer_module(hidden_states, run_full_attention=run_full_attention)
    all_hidden_states.append(hidden_states)
    return all_hidden_states


class WindowedDinov2WithRegistersBackbone:
  def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
    self.config = config
    self.embeddings = WindowedDinov2WithRegistersEmbeddings(config)
    self.encoder = WindowedDinov2WithRegistersEncoder(config)
    self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.num_register_tokens = config.num_register_tokens

  def __call__(self, pixel_values: Tensor) -> List[Tensor]:
    embeddings = self.embeddings(pixel_values)
    hidden_states = self.encoder(embeddings)
    features: List[Tensor] = []
    batch_size, _, height, width = pixel_values.shape
    patch_h = height // self.config.patch_size
    patch_w = width // self.config.patch_size
    for stage_name, stage_hidden in zip(self.config.stage_names, hidden_states):
      if stage_name not in self.config.out_features:
        continue
      if self.config.apply_layernorm:
        stage_hidden = self.layernorm(stage_hidden)
      if self.config.reshape_hidden_states:
        tokens = stage_hidden[:, self.num_register_tokens + 1 :]
        if self.config.num_windows > 1:
          tokens = _merge_window_patches(tokens, batch_size, self.config.num_windows, (patch_h, patch_w))
        tokens = tokens.reshape(batch_size, patch_h, patch_w, -1)
        stage_hidden = tokens.permute(0, 3, 1, 2)
      features.append(stage_hidden)
    return features





class LayerNorm2d:
  def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
    self.weight = Tensor.ones(num_channels)
    self.bias = Tensor.zeros(num_channels)
    self.eps = eps

  def __call__(self, x: Tensor) -> Tensor:
    y = x.permute(0, 2, 3, 1).layernorm(axis=-1, eps=self.eps)
    y = y * self.weight.reshape(1, 1, 1, -1) + self.bias.reshape(1, 1, 1, -1)
    return y.permute(0, 3, 1, 2)


class Identity:
  def __call__(self, x: Tensor) -> Tensor:
    return x


class Activation:
  def __init__(self, name: str) -> None:
    self.name = name

  def __call__(self, x: Tensor) -> Tensor:
    if self.name == "silu":
      return x * x.sigmoid()
    if self.name == "relu":
      return x.relu()
    if self.name == "gelu":
      return _gelu_pytorch(x)
    raise ValueError(f"Unsupported activation {self.name}")


class TinySequential:
  def __init__(self, *layers) -> None:
    self._layer_names: List[str] = []
    for idx, layer in enumerate(layers):
      name = str(idx)
      setattr(self, name, layer)
      self._layer_names.append(name)

  def __call__(self, x: Tensor) -> Tensor:
    out = x
    for name in self._layer_names:
      out = getattr(self, name)(out)
    return out


def _get_norm(norm: Optional[str], out_channels: int):
  if norm is None:
    return Identity()
  if norm == "LN":
    return LayerNorm2d(out_channels)
  raise ValueError(f"Unsupported norm {norm}")


class ConvX:
  def __init__(
    self,
    in_planes: int,
    out_planes: int,
    kernel: int | Tuple[int, int] = 3,
    stride: int = 1,
    act: str = "relu",
    layer_norm: bool = False,
  ) -> None:
    kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)
    padding = (kernel[0] // 2, kernel[1] // 2)
    self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
    self.bn = LayerNorm2d(out_planes) if layer_norm else nn.BatchNorm(out_planes)
    self.act = Activation(act)

  def __call__(self, x: Tensor) -> Tensor:
    return self.act(self.bn(self.conv(x)))


class Bottleneck:
  def __init__(
    self,
    c1: int,
    c2: int,
    shortcut: bool = True,
    g: int = 1,
    k: Tuple[int, int] = (3, 3),
    e: float = 0.5,
    act: str = "silu",
    layer_norm: bool = False,
  ) -> None:
    hidden = int(c2 * e)
    self.cv1 = ConvX(c1, hidden, k[0], 1, act=act, layer_norm=layer_norm)
    self.cv2 = ConvX(hidden, c2, k[1], 1, act=act, layer_norm=layer_norm)
    self.add = shortcut and c1 == c2

  def __call__(self, x: Tensor) -> Tensor:
    out = self.cv2(self.cv1(x))
    return x + out if self.add else out


class C2f:
  def __init__(
    self,
    c1: int,
    c2: int,
    n: int = 1,
    shortcut: bool = False,
    g: int = 1,
    e: float = 0.5,
    act: str = "silu",
    layer_norm: bool = False,
  ) -> None:
    self.c = int(c2 * e)
    self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm)
    self.cv2 = ConvX((2 + n) * self.c, c2, 1, 1, act=act, layer_norm=layer_norm)
    self.m = [Bottleneck(self.c, self.c, shortcut, g=g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm) for _ in range(n)]

  def __call__(self, x: Tensor) -> Tensor:
    y = list(self.cv1(x).split((self.c, self.c), dim=1))
    for block in self.m:
      y.append(block(y[-1]))
    return self.cv2(Tensor.cat(*y, dim=1))


class MultiScaleProjector:
  def __init__(
    self,
    in_channels: Sequence[int],
    out_channels: int,
    scale_factors: Sequence[float],
    num_blocks: int = 3,
    layer_norm: bool = False,
    survival_prob: float = 1.0,
    force_drop_last_n_features: int = 0,
  ) -> None:
    self.scale_factors = scale_factors
    self.survival_prob = survival_prob
    self.force_drop_last_n_features = force_drop_last_n_features
    self.use_extra_pool = False
    self.stages_sampling: List[List[TinySequential]] = []
    self.stages: List[TinySequential] = []
    for scale in scale_factors:
      sampling_modules: List[TinySequential] = []
      for in_dim in in_channels:
        layers = []
        if scale == 4.0:
          layers.append(nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2))
          layers.append(LayerNorm2d(in_dim // 2) if layer_norm else nn.BatchNorm(in_dim // 2))
          layers.append(Activation("gelu"))
          layers.append(nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2))
        elif scale == 2.0:
          layers.append(nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2))
        elif scale == 1.0:
          pass
        elif scale == 0.5:
          layers.append(ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm))
        elif scale == 0.25:
          self.use_extra_pool = True
          continue
        else:
          raise NotImplementedError(f"Unsupported scale factor {scale}")
        sampling_modules.append(TinySequential(*layers))
      self.stages_sampling.append(sampling_modules)

      fused_dim = int(sum(in_channel // max(1, int(scale)) if scale > 1 else in_channel for in_channel in in_channels))
      stage_layers = [
        C2f(fused_dim, out_channels, num_blocks, layer_norm=layer_norm),
        _get_norm("LN", out_channels),
      ]
      self.stages.append(TinySequential(*stage_layers))


  def __call__(self, x: List[Tensor]) -> List[Tensor]:
    num_features = len(x)
    features = list(x)
    if self.force_drop_last_n_features > 0:
      for i in range(self.force_drop_last_n_features):
        features[-(i + 1)] = Tensor.zeros_like(features[-(i + 1)])

    results: List[Tensor] = []
    for stage_idx, stage in enumerate(self.stages):
      fused: List[Tensor] = []
      for feat_idx, sampler in enumerate(self.stages_sampling[stage_idx]):
        fused.append(sampler(features[feat_idx]))
      fused_feature = fused[0] if len(fused) == 1 else Tensor.cat(*fused, dim=1)
      results.append(stage(fused_feature))
    if self.use_extra_pool:
      results.append(results[-1].max_pool2d(kernel_size=(1, 1), stride=(2, 2), padding=(0, 0)))
    return results


class TinyDinoV2:
  def __init__(
    self,
    *,
    hidden_size: int,
    num_hidden_layers: int,
    out_feature_indexes: Sequence[int],
    patch_size: int,
    num_windows: int,
    positional_encoding_size: int,
  ) -> None:
    config = WindowedDinov2WithRegistersConfig(
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=hidden_size // 64,
      image_size=patch_size * positional_encoding_size,
      patch_size=patch_size,
      num_windows=num_windows,
      out_feature_indexes=tuple(out_feature_indexes),
      interpolate_offset=-0.5,   # Tuned for better match to official bicubic grid (align_corners=False style)
    )
    window_block = set(range(out_feature_indexes[-1] + 1))
    window_block.difference_update(out_feature_indexes)
    config.window_block_indexes = tuple(sorted(window_block))
    self.encoder = WindowedDinov2WithRegistersBackbone(config)
    self._out_feature_channels = list(config.out_feature_channels)

  def __call__(self, x: Tensor) -> List[Tensor]:
    return self.encoder(x)





class DINOv2Backbone:
  def __init__(
    self,
    *,
    size: str = "small",
    out_feature_indexes: Sequence[int] = (3, 6, 9, 12),
    patch_size: int = 16,
    num_windows: int = 2,
    positional_encoding_size: int = 24,
    projector_scale: Sequence[str] = ("P4",),
    projector_dim: int = 256,
    layer_norm: bool = True,
  ) -> None:
    size_to_hidden = {"tiny": 192, "small": 384, "base": 768, "large": 1024}
    hidden = size_to_hidden[size]
    self.encoder = TinyDinoV2(
      hidden_size=hidden,
      num_hidden_layers=12,
      out_feature_indexes=out_feature_indexes,
      patch_size=patch_size,
      num_windows=num_windows,
      positional_encoding_size=positional_encoding_size,
    )
    level_map = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
    scale_factors = [level_map[lvl] for lvl in projector_scale]
    self.projector = MultiScaleProjector(
      in_channels=self.encoder._out_feature_channels,
      out_channels=projector_dim,
      scale_factors=scale_factors,
      layer_norm=layer_norm,
    )

  def __call__(self, x: Tensor) -> List[Tensor]:
    return self.projector(self.encoder(x))

# RF-DETR Nano @ 384 uses single P4 level (24x24).
_NANO_P4_SHAPE: Tuple[int, int] = (24, 24)


def _spatial_shapes(data, n_levels: int) -> List[Tuple[int, int]]:
  """Resolve spatial shapes without forcing .numpy() on the JIT hot path.

  Nano (n_levels==1) is always P4 24x24 @ 384 — return the constant so TinyJit
  capture never host-syncs spatial_shapes (required under TG_SPEED_MODE).
  """
  if n_levels == 1:
    if isinstance(data, Tensor):
      # (1, 2) int tensor from HybridEncoder — values are always 24,24 for Nano.
      if len(data.shape) >= 1 and int(data.shape[0]) == 1:
        return [_NANO_P4_SHAPE]
    elif len(data) == 1:
      h, w = int(data[0][0]), int(data[0][1])
      return [(h, w)]
  if isinstance(data, Tensor):
    return [tuple(int(x) for x in row) for row in data.numpy().tolist()]
  return [tuple(int(x) for x in row) for row in data]


def bilinear_sampler_fast(features: Tensor, grid: Tensor) -> Tensor:
  """Bilinear gather with one feature flatten realize (parity-compatible with msda.py)."""
  n, c, h, w = features.shape
  _, total_pts, _ = grid.shape

  gx = grid[..., 0]
  gy = grid[..., 1]

  x0 = gx.floor().cast(dtypes.int32)
  y0 = gy.floor().cast(dtypes.int32)
  x1 = x0 + 1
  y1 = y0 + 1

  x0c, y0c = x0.clip(0, w - 1), y0.clip(0, h - 1)
  x1c, y1c = x1.clip(0, w - 1), y1.clip(0, h - 1)

  fx0, fy0 = x0.cast(dtypes.float32), y0.cast(dtypes.float32)
  wa = (x1.cast(dtypes.float32) - gx) * (y1.cast(dtypes.float32) - gy)
  wb = (x1.cast(dtypes.float32) - gx) * (gy - fy0)
  wc = (gx - fx0) * (y1.cast(dtypes.float32) - gy)
  wd = (gx - fx0) * (gy - fy0)

  feat_flat = _sync_msda(features.permute(0, 2, 3, 1).reshape(n * h * w, c))
  batch_offsets = Tensor.arange(n, dtype=dtypes.int32).reshape(n, 1).mul(h * w).expand(n, total_pts)

  def gather(ix: Tensor, iy: Tensor) -> Tensor:
    idx = (batch_offsets + iy * w + ix).reshape(-1)
    return feat_flat[idx].reshape(n, total_pts, c)

  v00 = gather(x0c, y0c)
  v01 = gather(x0c, y1c)
  v10 = gather(x1c, y0c)
  v11 = gather(x1c, y1c)

  mask00 = ((x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)).cast(dtypes.float32)
  mask01 = ((x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h)).cast(dtypes.float32)
  mask10 = ((x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h)).cast(dtypes.float32)
  mask11 = ((x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)).cast(dtypes.float32)

  summed = (
    v00 * wa.unsqueeze(-1) * mask00.unsqueeze(-1)
    + v01 * wb.unsqueeze(-1) * mask01.unsqueeze(-1)
    + v10 * wc.unsqueeze(-1) * mask10.unsqueeze(-1)
    + v11 * wd.unsqueeze(-1) * mask11.unsqueeze(-1)
  )
  return summed.permute(0, 2, 1)


class MSDeformAttnFast:
  """Vectorized MSDA — same modules/weights as MSDeformAttn, no query chunk loop."""

  def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
    if d_model % n_heads != 0:
      raise ValueError("d_model must be divisible by n_heads")
    self.d_model = d_model
    self.n_levels = n_levels
    self.n_heads = n_heads
    self.n_points = n_points
    self.head_dim = d_model // n_heads

    self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
    self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
    self.value_proj = nn.Linear(d_model, d_model)
    self.output_proj = nn.Linear(d_model, d_model)

  def __call__(
    self,
    query: Tensor,
    reference_points: Tensor,
    input_flatten: Tensor,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask: Tensor | None = None,
  ) -> Tensor:
    query = _sync_msda(query)
    reference_points = _sync_msda(reference_points)
    input_flatten = _sync_msda(input_flatten)

    n, len_q, _ = query.shape
    _, len_in, _ = input_flatten.shape

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
      value = value.masked_fill(input_padding_mask[..., None], 0.0)

    value = _sync_msda(
      value.reshape(n, len_in, self.n_heads, self.head_dim)
      .transpose(1, 2)
      .permute(0, 1, 3, 2)
    )

    spatial_shapes_list = _spatial_shapes(input_spatial_shapes, self.n_levels)
    offset_normalizer = Tensor(
      [[float(w), float(h)] for h, w in spatial_shapes_list],
      dtype=dtypes.float32,
    )
    scales = [Tensor([float(w), float(h)], dtype=dtypes.float32) for h, w in spatial_shapes_list]

    split_values: List[Tensor] = []
    start = 0
    for h_l, w_l in spatial_shapes_list:
      end = start + h_l * w_l
      lvl = value[..., start:end]
      split_values.append(_sync_msda(lvl.reshape(n * self.n_heads, self.head_dim, h_l, w_l)))
      start = end

    ref_points = reference_points[:, :len_q]

    offsets = self.sampling_offsets(query).reshape(
      n, len_q, self.n_heads, self.n_levels, self.n_points, 2
    )
    weights = (
      self.attention_weights(query)
      .reshape(n, len_q, self.n_heads, self.n_levels * self.n_points)
      .softmax(-1)
      .reshape(n, len_q, self.n_heads, self.n_levels, self.n_points)
    )

    offsets = offsets.isfinite().where(offsets, offsets * 0)
    ref_points = ref_points.isfinite().where(ref_points, ref_points * 0)

    level_outputs: List[Tensor] = []
    for lvl, (h_l, w_l) in enumerate(spatial_shapes_list):
      locs = offsets[:, :, :, lvl, :, :]

      if ref_points.ndim == 4:
        ref_lvl = ref_points[:, :, lvl, :]
      else:
        ref_lvl = ref_points

      if ref_lvl.shape[-1] == 2:
        base = ref_lvl.reshape(n, len_q, 1, 1, 2)
        norm = offset_normalizer[lvl].reshape(1, 1, 1, 1, 2)
        sampling_locations = base + locs / norm
      else:
        base = ref_lvl[..., :2].reshape(n, len_q, 1, 1, 2)
        size = ref_lvl[..., 2:].reshape(n, len_q, 1, 1, 2)
        sampling_locations = base + locs / self.n_points * size * 0.5

      locs_lvl = sampling_locations.permute(0, 2, 1, 3, 4)
      locs_lvl = locs_lvl.reshape(n * self.n_heads, len_q * self.n_points, 2)
      locs_lvl = locs_lvl.isfinite().where(locs_lvl, locs_lvl * 0)
      locs_pixel = locs_lvl * scales[lvl].reshape(1, 1, 2) - 0.5

      sampled_flat = bilinear_sampler_fast(split_values[lvl], locs_pixel)
      sampled = sampled_flat.reshape(
        n, self.n_heads, self.head_dim, len_q, self.n_points
      ).permute(0, 3, 1, 4, 2)

      w_lvl = weights[:, :, :, lvl, :].unsqueeze(-1)
      level_outputs.append(sampled * w_lvl)

    full_output = Tensor.stack(level_outputs, dim=0).sum(axis=0).sum(axis=3)
    full_output = full_output.reshape(n, len_q, self.n_heads * self.head_dim)


    return self.output_proj(full_output)

def inverse_sigmoid(x: Tensor, eps: float = 1e-6) -> Tensor:
  x = x.clip(eps, 1 - eps)
  return (x / (1 - x)).log()


def gen_sineembed_for_position(pos_tensor: Tensor, dim: int = 128) -> Tensor:
  """Exact numeric port of official rfdetr.models.transformer.gen_sineembed_for_position."""
  scale = 2 * math.pi
  dim_t = Tensor.arange(dim, dtype=pos_tensor.dtype)
  dim_t = _sync_parity(10000 ** (2 * (dim_t // 2) / dim))

  def _embed(coord: Tensor) -> Tensor:
    val = coord * scale
    pos = val / dim_t.reshape(1, 1, dim)
    sin_part = pos[..., 0::2].sin()
    cos_part = pos[..., 1::2].cos()
    stacked = Tensor.stack(sin_part, cos_part, dim=-1)
    return stacked.flatten(-2)

  if pos_tensor.shape[-1] == 4:
    cx = pos_tensor[..., 0:1]
    cy = pos_tensor[..., 1:2]
    w = pos_tensor[..., 2:3]
    h = pos_tensor[..., 3:4]
    return Tensor.cat(_embed(cy), _embed(cx), _embed(w), _embed(h), dim=-1)
  cx = pos_tensor[..., 0:1]
  cy = pos_tensor[..., 1:2]
  return Tensor.cat(_embed(cy), _embed(cx), dim=-1)


def gen_encoder_output_proposals(
  memory: Tensor,
  memory_padding_mask: Tensor | None,
  spatial_shapes: Sequence[Tuple[int, int]],
  unsigmoid: bool = True,
) -> Tuple[Tensor, Tensor]:
  bs, _, dim = memory.shape
  proposals: List[Tensor] = []
  cur = 0
  for lvl, (h, w) in enumerate(spatial_shapes):
    if memory_padding_mask is not None:
      mask = memory_padding_mask[:, cur : cur + h * w].reshape(bs, h, w)
      inv = (~mask).cast(memory.dtype)
      valid_h = inv[:, :, 0].sum(axis=1).reshape(bs, 1, 1, 1)
      valid_w = inv[:, 0, :].sum(axis=1).reshape(bs, 1, 1, 1)
    else:
      valid_h = Tensor.full((bs, 1, 1, 1), float(h), dtype=memory.dtype)
      valid_w = Tensor.full((bs, 1, 1, 1), float(w), dtype=memory.dtype)

    grid_y = Tensor.arange(h, dtype=memory.dtype).reshape(1, h, 1).expand(bs, h, w)
    grid_x = Tensor.arange(w, dtype=memory.dtype).reshape(1, 1, w).expand(bs, h, w)
    grid = Tensor.stack(grid_x, grid_y, dim=-1)
    grid = (grid + 0.5) / Tensor.cat(valid_w, valid_h, dim=-1)
    wh = Tensor.ones_like(grid) * (0.05 * (2.0 ** lvl))
    proposals.append(Tensor.cat(grid, wh, dim=-1).reshape(bs, -1, 4))
    cur += h * w

  output_proposals = proposals[0] if len(proposals) == 1 else Tensor.cat(*proposals, dim=1)
  valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(axis=-1, keepdim=True)

  if unsigmoid:
    eps = 1e-6
    clipped = output_proposals.clip(eps, 1 - eps)
    output_proposals = (clipped / (1 - clipped)).log()
    if memory_padding_mask is not None:
      output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
    output_proposals = output_proposals.masked_fill(~valid, 0.0)
  else:
    if memory_padding_mask is not None:
      output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
    output_proposals = output_proposals.masked_fill(~valid, 0.0)

  output_memory = memory
  if memory_padding_mask is not None:
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
  output_memory = output_memory.masked_fill(~valid, 0.0)
  return output_memory, output_proposals


class HybridEncoder:
  def __init__(
    self,
    *,
    d_model: int = 256,
    num_feature_levels: int = 1,
    base_num_queries: int = 300,
    group_detr: int = 1,
    two_stage: bool = True,
    bbox_reparam: bool = True,
  ) -> None:
    self.d_model = d_model
    self.num_feature_levels = num_feature_levels
    self.base_num_queries = base_num_queries
    self.group_detr = max(1, group_detr)
    self.two_stage = two_stage
    self.bbox_reparam = bbox_reparam
    self.enc_output = [nn.Linear(d_model, d_model) for _ in range(self.group_detr)]
    self.enc_output_norm = [nn.LayerNorm(d_model) for _ in range(self.group_detr)]
    self.enc_out_class_embed: List[nn.Linear] | None = None
    self.enc_out_bbox_embed: List[DetectionMLP] | None = None

  def __call__(self, feats: List[Tensor], *, num_groups: int = 1) -> dict[str, Tensor | None]:
    bs = feats[0].shape[0]
    spatial_shapes: List[Tuple[int, int]] = []
    flattened: List[Tensor] = []
    for feat in feats:
      b, c, h, w = feat.shape
      assert b == bs, "All features must share the same batch size"
      spatial_shapes.append((h, w))
      flattened.append(feat.reshape(b, c, h * w).transpose(1, 2))
    memory = flattened[0] if len(flattened) == 1 else Tensor.cat(*flattened, dim=1)

    valid_ratios = Tensor.ones(bs, len(spatial_shapes), 2, dtype=memory.dtype)
    level_start_index: List[int] = []
    offset = 0
    for h, w in spatial_shapes:
      level_start_index.append(offset)
      offset += h * w

    output: dict[str, Tensor | None] = {
      "memory": memory,
      "mask": None,
      "spatial_shapes": Tensor(spatial_shapes, dtype=dtypes.int32),
      "level_start_index": Tensor(level_start_index, dtype=dtypes.int32),
      "valid_ratios": valid_ratios,
      "two_stage_refpoints": None,
      "two_stage_memory": None,
      "two_stage_boxes": None,
    }

    if self.two_stage and self.enc_out_bbox_embed is not None and self.enc_out_class_embed is not None:
      refpoints, mems, boxes = self._two_stage_select(memory, spatial_shapes, num_groups=num_groups)
      output["two_stage_refpoints"] = refpoints
      output["two_stage_memory"] = mems
      output["two_stage_boxes"] = boxes
    return output

  def _two_stage_select(
    self, memory: Tensor, spatial_shapes: Sequence[Tuple[int, int]], *, num_groups: int = 1,
  ) -> Tuple[Tensor, Tensor, Tensor]:
    memory_unrolled, proposals = gen_encoder_output_proposals(
      _sync(memory), None, spatial_shapes, unsigmoid=not self.bbox_reparam,
    )
    memory_unrolled, proposals = _sync(memory_unrolled), _sync(proposals)

    ref_list: List[Tensor] = []
    mem_list: List[Tensor] = []
    boxes_list: List[Tensor] = []
    for idx in range(max(1, num_groups)):
      proj = _sync_two_stage(self.enc_output_norm[idx](self.enc_output[idx](memory_unrolled)))
      class_logits = _sync_two_stage(self.enc_out_class_embed[idx](proj))
      raw_boxes = _sync_two_stage(self.enc_out_bbox_embed[idx](proj))
      if self.bbox_reparam:
        cxcy = _sync_two_stage(raw_boxes[..., :2] * proposals[..., 2:] + proposals[..., :2])
        wh = _sync_two_stage(raw_boxes[..., 2:].exp() * proposals[..., 2:])
        box_preds = Tensor.cat(cxcy, wh, dim=-1)
      else:
        box_preds = _sync_two_stage(raw_boxes + proposals)
      topk = min(self.base_num_queries, class_logits.shape[1])
      scores = _sync_two_stage(class_logits.max(axis=-1))
      topk_idx = topk_pytorch_compat(scores, topk, dim=-1)
      topk_idx = _sync(topk_idx)
      mem_idx = _sync_two_stage(topk_idx.unsqueeze(-1).expand(topk_idx.shape[0], topk, proj.shape[-1]))
      box_idx = _sync_two_stage(topk_idx.unsqueeze(-1).expand(topk_idx.shape[0], topk, 4))
      gathered_boxes = box_preds.gather(1, box_idx)
      ref_list.append(gathered_boxes.detach())
      boxes_list.append(gathered_boxes)
      mem_list.append(proj.gather(1, mem_idx))

    refpoints = ref_list[0] if len(ref_list) == 1 else Tensor.cat(*ref_list, dim=1)
    mems = mem_list[0] if len(mem_list) == 1 else Tensor.cat(*mem_list, dim=1)
    boxes = boxes_list[0] if len(boxes_list) == 1 else Tensor.cat(*boxes_list, dim=1)
    return refpoints, mems, boxes


class TransformerDecoderLayer:
  def __init__(
    self,
    *,
    d_model: int = 256,
    sa_heads: int = 8,
    ca_heads: int = 4,
    n_levels: int = 1,
    n_points: int = 4,
    ffn_dim: int = 2048,
    group_detr: int = 1,
  ) -> None:
    self.d_model = d_model
    self.sa_heads = sa_heads
    self.ca_heads = ca_heads
    self.n_levels = n_levels
    self.n_points = n_points
    self.group_detr = max(1, group_detr)

    self.sa = nn.Linear(d_model, d_model * 3)
    self.sa_out = nn.Linear(d_model, d_model)
    self.sa_norm = nn.LayerNorm(d_model)
    self.cross_attn = MSDeformAttnFast(d_model, n_levels=n_levels, n_heads=ca_heads, n_points=n_points)
    self.ca_norm = nn.LayerNorm(d_model)
    self.linear1 = nn.Linear(d_model, ffn_dim)
    self.linear2 = nn.Linear(ffn_dim, d_model)
    self.ffn_norm = nn.LayerNorm(d_model)

  def _self_attention(self, tgt: Tensor, query_pos: Tensor | None) -> Tensor:
    bsz, num_queries, dim = tgt.shape
    qk_input = tgt if query_pos is None else tgt + query_pos

    def _proj(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
      out = x @ weight.transpose()
      return out if bias is None else out + bias

    qkv = _proj(qk_input, self.sa.weight, self.sa.bias)
    q, k, _ = qkv.chunk(3, dim=-1)
    v = _proj(tgt, self.sa.weight[2 * dim :], self.sa.bias[2 * dim :])

    head_dim = dim // self.sa_heads
    q = q.reshape(bsz, num_queries, self.sa_heads, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(bsz, num_queries, self.sa_heads, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(bsz, num_queries, self.sa_heads, head_dim).permute(0, 2, 1, 3)
    context = q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(bsz, num_queries, dim)
    return self.sa_out(context)

  def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
    return tensor if pos is None else tensor + pos

  def __call__(
    self,
    tgt: Tensor,
    reference_points: Tensor,
    input_flatten: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    *,
    query_pos: Tensor | None = None,
    padding_mask: Tensor | None = None,
  ) -> Tensor:
    sa_out = self._self_attention(tgt, query_pos)
    tgt = self.sa_norm(tgt + sa_out)
    tgt = _sync(tgt)
    cross = self.cross_attn(
      self.with_pos_embed(tgt, query_pos),
      reference_points,
      input_flatten,
      spatial_shapes,
      level_start_index,
      padding_mask,
    )
    cross = _sync(cross)
    tgt = self.ca_norm(tgt + cross)
    tgt = _sync(tgt)
    ffn = self.linear2(self.linear1(tgt).relu())
    return _sync(self.ffn_norm(tgt + ffn))


class TransformerDecoder:
  def __init__(
    self,
    layers: List[TransformerDecoderLayer],
    *,
    d_model: int = 256,
    return_intermediate: bool = True,
    lite_refpoint_refine: bool = True,
    bbox_reparam: bool = True,
  ) -> None:
    self.layers = layers
    self.num_layers = len(layers)
    self.norm = nn.LayerNorm(d_model)
    self.return_intermediate = return_intermediate
    self.lite_refpoint_refine = lite_refpoint_refine
    self.bbox_reparam = bbox_reparam
    self.ref_point_head = DetectionMLP(2 * d_model, d_model, d_model, num_layers=2)
    self.bbox_embed: DetectionMLP | None = None

  def refpoints_refine(self, refpoints_unsigmoid: Tensor, deltas: Tensor) -> Tensor:
    if self.bbox_reparam:
      cxcy = deltas[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
      wh = deltas[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
      return Tensor.cat(cxcy, wh, dim=-1)
    return refpoints_unsigmoid + deltas

  def __call__(
    self,
    tgt: Tensor,
    memory: Tensor,
    *,
    memory_key_padding_mask: Tensor | None = None,
    refpoints_unsigmoid: Tensor | None = None,
    spatial_shapes: Tensor | None = None,
    level_start_index: Tensor | None = None,
    valid_ratios: Tensor | None = None,
  ) -> Tuple[Tensor, Tensor]:
    memory = _sync(memory)
    spatial_shapes = _sync(spatial_shapes) if spatial_shapes is not None else None
    level_start_index = _sync(level_start_index) if level_start_index is not None else None
    memory_key_padding_mask = _sync(memory_key_padding_mask) if memory_key_padding_mask is not None else None
    valid_ratios = _sync(valid_ratios) if valid_ratios is not None else None

    output = _sync(tgt)
    intermediate: List[Tensor] = []
    refpoints = _sync(refpoints_unsigmoid) if refpoints_unsigmoid is not None else Tensor.zeros_like(output[..., :4])
    ratios = valid_ratios if valid_ratios is not None else Tensor.ones(
      output.shape[0], spatial_shapes.shape[0] if spatial_shapes is not None else 1, 2, dtype=output.dtype
    )
    ratio_cat = _sync(Tensor.cat(ratios, ratios, dim=-1))

    def get_reference(current: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
      current = _sync(current)
      obj_center = current if self.bbox_reparam else current.sigmoid()
      obj_center = _sync(obj_center)
      ratio = ratio_cat[:, None, :, :]
      ref_input = _sync(obj_center[:, :, None, :] * ratio)
      sine_input = _sync(ref_input[:, :, 0, :])
      sine_embed = gen_sineembed_for_position(sine_input, self.layers[0].d_model // 2)
      query_pos = self.ref_point_head(sine_embed)
      return obj_center, ref_input, query_pos

    for idx, layer in enumerate(self.layers):
      if idx > 0:
        output = _sync(output)
        refpoints = _sync(refpoints)
      _, ref_input, query_pos = get_reference(refpoints)
      ref_input = _sync(ref_input)
      query_pos = _sync(query_pos)
      output = layer(
        output,
        ref_input,
        memory,
        spatial_shapes if spatial_shapes is not None else Tensor([[0, 0]], dtype=dtypes.int32),
        level_start_index if level_start_index is not None else Tensor([0], dtype=dtypes.int32),
        query_pos=query_pos,
        padding_mask=memory_key_padding_mask,
      )
      intermediate.append(self.norm(output))
      if not self.lite_refpoint_refine and self.bbox_embed is not None:
        delta = self.bbox_embed(output)
        refpoints = self.refpoints_refine(refpoints, delta)
        refpoints = _sync(refpoints)

    refs = refpoints.unsqueeze(0)
    if self.return_intermediate:
      return Tensor.stack(*intermediate), refs
    return self.norm(output).unsqueeze(0), refs


class DetectionMLP:
  def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 3) -> None:
    dims = [in_dim]
    dims.extend(hidden_dim for _ in range(max(0, num_layers - 1)))
    dims.append(out_dim)
    self.layers = [nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]
    self.num_layers = len(self.layers)

  def __call__(self, x: Tensor) -> Tensor:
    out = x
    for idx, layer in enumerate(self.layers):
      out = layer(out)
      if idx < self.num_layers - 1:
        out = out.relu()
    return out


class RFDetr:
  def __init__(
    self,
    *,
    num_classes: int = 91,
    num_queries: int = 3900,
    num_layers: int = 2,
    projector_levels: Sequence[str] = ("P4",),
    group_detr: int = 13,
    sa_heads: int = 8,
    ca_heads: int = 16,
    dec_n_points: int = 2,
    lite_refpoint_refine: bool = True,
    bbox_reparam: bool = True,
    two_stage: bool = True,
  ) -> None:
    self.backbone = [DINOv2Backbone(projector_scale=projector_levels)]
    self.num_feature_levels = len(projector_levels)
    self.group_detr = max(1, group_detr)
    self.total_queries = num_queries
    self.base_queries = max(1, num_queries // self.group_detr)

    self.query_feat = nn.Embedding(num_queries, 256)
    self.refpoint_embed = nn.Embedding(num_queries, 4)
    self.input_proj = [nn.Conv2d(256, 256, 1) for _ in range(self.num_feature_levels)]
    for p in self.input_proj:
      p.weight.assign(Tensor.eye(256, dtype=p.weight.dtype).reshape(256, 256, 1, 1))
      if p.bias is not None:
        p.bias.assign(Tensor.zeros(256, dtype=p.bias.dtype))

    self.encoder = HybridEncoder(
      d_model=256,
      num_feature_levels=self.num_feature_levels,
      base_num_queries=self.base_queries,
      group_detr=self.group_detr,
      two_stage=two_stage,
      bbox_reparam=bbox_reparam,
    )

    decoder_layers = [
      TransformerDecoderLayer(
        d_model=256,
        sa_heads=sa_heads,
        ca_heads=ca_heads,
        n_levels=self.num_feature_levels,
        n_points=dec_n_points,
        ffn_dim=2048,
        group_detr=self.group_detr,
      )
      for _ in range(num_layers)
    ]
    self.decoder = TransformerDecoder(
      decoder_layers,
      d_model=256,
      return_intermediate=True,
      lite_refpoint_refine=lite_refpoint_refine,
      bbox_reparam=bbox_reparam,
    )

    self.class_head = nn.Linear(256, num_classes)
    self.box_head = DetectionMLP(256, 256, 4, num_layers=3)

    if not lite_refpoint_refine:
      self.decoder.bbox_embed = self.box_head
    self.encoder.enc_out_class_embed = [nn.Linear(256, num_classes) for _ in range(self.group_detr)]
    self.encoder.enc_out_bbox_embed = [DetectionMLP(256, 256, 4, num_layers=3) for _ in range(self.group_detr)]

  def _select_query_embeddings(self) -> Tuple[Tensor, Tensor]:
    return (
      self.query_feat.weight[: self.base_queries],
      self.refpoint_embed.weight[: self.base_queries],
    )

  def _apply_two_stage_refs(self, refpoints: Tensor, proposals: Tensor, boxes: Tensor | None) -> Tensor:
    refpoints, proposals = _sync(refpoints), _sync(proposals)
    boxes = _sync(boxes) if boxes is not None else None
    ts_len = proposals.shape[1]
    ref_subset = refpoints[:, :ts_len, :]
    rest = refpoints[:, ts_len:, :]
    if ref_subset.shape[1] == 0:
      return refpoints
    if self.decoder.bbox_reparam:
      cxcy = _sync(proposals[..., :2] + ref_subset[..., :2] * proposals[..., 2:])
      wh = _sync(proposals[..., 2:] * ref_subset[..., 2:].exp())
      ref_subset = Tensor.cat(cxcy, wh, dim=-1)
    else:
      ref_subset = _sync(ref_subset + proposals)
    if rest.shape[1] == 0:
      return ref_subset
    return _sync(Tensor.cat(ref_subset, rest, dim=1))

  def __call__(self, x: Tensor) -> dict[str, Tensor]:
    return self._forward_impl(x)

  def _forward_impl(self, x: Tensor) -> dict[str, Tensor]:
    bs = x.shape[0]
    features = self.backbone[0](x)
    if len(features) < self.num_feature_levels:
      features = features + [features[-1] for _ in range(self.num_feature_levels - len(features))]

    feats = [proj(feat) for proj, feat in zip(self.input_proj, features[: self.num_feature_levels])]
    enc = self.encoder(feats)
    enc_memory = _sync(enc["memory"])
    enc_mask = _sync(enc["mask"]) if enc["mask"] is not None else None
    enc_shapes = _sync(enc["spatial_shapes"])
    enc_level_start = _sync(enc["level_start_index"])
    enc_valid_ratios = _sync(enc["valid_ratios"])
    enc_two_stage_refs = _sync(enc["two_stage_refpoints"]) if enc["two_stage_refpoints"] is not None else None
    enc_two_stage_boxes = _sync(enc["two_stage_boxes"]) if enc["two_stage_boxes"] is not None else None

    query_embed, ref_embed = self._select_query_embeddings()
    query = _sync(query_embed.unsqueeze(0).expand(bs, -1, -1))
    refpoints = _sync(ref_embed.unsqueeze(0).expand(bs, -1, -1))

    if enc_two_stage_refs is not None:
      refpoints = self._apply_two_stage_refs(refpoints, enc_two_stage_boxes, enc_two_stage_boxes)
      refpoints = _sync(refpoints)

    hs, references = self.decoder(
      query,
      enc_memory,
      memory_key_padding_mask=enc_mask,
      refpoints_unsigmoid=refpoints,
      spatial_shapes=enc_shapes,
      level_start_index=enc_level_start,
      valid_ratios=enc_valid_ratios,
    )

    logits = self.class_head(hs)
    if self.decoder.bbox_reparam:
      deltas = self.box_head(hs)
      cxcy = deltas[..., :2] * references[..., 2:] + references[..., :2]
      wh = deltas[..., 2:].exp() * references[..., 2:]
      boxes = Tensor.cat(cxcy, wh, dim=-1)
    else:
      boxes = (self.box_head(hs) + references).sigmoid()

    out: dict[str, Tensor] = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}
    if logits.shape[0] > 1:
      out["aux_outputs"] = [
        {"pred_logits": logits[idx], "pred_boxes": boxes[idx]}
        for idx in range(logits.shape[0] - 1)
      ]
    return out

  def forward_train(self, x: Tensor, *, num_groups: int | None = None) -> dict[str, Tensor]:
    """Training forward: full query bank, no realize(), returns official loss dict keys."""
    ng = self.group_detr if num_groups is None else num_groups
    bs = x.shape[0]
    features = self.backbone[0](x)
    if len(features) < self.num_feature_levels:
      features = features + [features[-1] for _ in range(self.num_feature_levels - len(features))]

    feats = [proj(feat) for proj, feat in zip(self.input_proj, features[: self.num_feature_levels])]
    enc = self.encoder(feats, num_groups=ng)
    enc_memory = enc["memory"]
    enc_mask = enc["mask"]
    enc_shapes = enc["spatial_shapes"]
    enc_level_start = enc["level_start_index"]
    enc_valid_ratios = enc["valid_ratios"]
    enc_two_stage_boxes = enc["two_stage_boxes"]
    enc_two_stage_memory = enc["two_stage_memory"]

    query = self.query_feat.weight.unsqueeze(0).expand(bs, -1, -1)
    refpoints = self.refpoint_embed.weight.unsqueeze(0).expand(bs, -1, -1)

    if enc_two_stage_boxes is not None:
      ts_len = enc_two_stage_boxes.shape[1]
      ref_subset = refpoints[:, :ts_len, :]
      rest = refpoints[:, ts_len:, :]
      if self.decoder.bbox_reparam:
        cxcy = ref_subset[..., :2] * enc_two_stage_boxes[..., 2:] + enc_two_stage_boxes[..., :2]
        wh = ref_subset[..., 2:].exp() * enc_two_stage_boxes[..., 2:]
        ref_subset = Tensor.cat(cxcy, wh, dim=-1)
      else:
        ref_subset = ref_subset + enc_two_stage_boxes
      refpoints = ref_subset if rest.shape[1] == 0 else Tensor.cat(ref_subset, rest, dim=1)

    hs, references = self.decoder(
      query,
      enc_memory,
      memory_key_padding_mask=enc_mask,
      refpoints_unsigmoid=refpoints,
      spatial_shapes=enc_shapes,
      level_start_index=enc_level_start,
      valid_ratios=enc_valid_ratios,
    )

    logits = self.class_head(hs)
    if self.decoder.bbox_reparam:
      deltas = self.box_head(hs)
      cxcy = deltas[..., :2] * references[..., 2:] + references[..., :2]
      wh = deltas[..., 2:].exp() * references[..., 2:]
      boxes = Tensor.cat(cxcy, wh, dim=-1)
    else:
      boxes = (self.box_head(hs) + references).sigmoid()

    out: dict[str, Tensor] = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}
    if logits.shape[0] > 1:
      out["aux_outputs"] = [
        {"pred_logits": logits[idx], "pred_boxes": boxes[idx]}
        for idx in range(logits.shape[0] - 1)
      ]
    if enc_two_stage_memory is not None and enc_two_stage_boxes is not None:
      qpg = enc_two_stage_memory.shape[1] // max(1, ng)
      cls_parts = [
        self.encoder.enc_out_class_embed[g_idx](enc_two_stage_memory[:, g_idx * qpg : (g_idx + 1) * qpg, :])
        for g_idx in range(ng)
      ]
      out["enc_outputs"] = {
        "pred_logits": Tensor.cat(*cls_parts, dim=1) if len(cls_parts) > 1 else cls_parts[0],
        "pred_boxes": enc_two_stage_boxes,
      }
    return out

  def update_drop_path(self, drop_path_rate: float, vit_encoder_num_layers: int = 12) -> None:
    """Linear drop-path schedule across ViT blocks (matches official lwdetr)."""
    dp_rates = np.linspace(0, drop_path_rate, vit_encoder_num_layers)
    for i, layer in enumerate(self.backbone[0].encoder.encoder.layer):
      layer.drop_path.drop_prob = float(dp_rates[i])

  @staticmethod
  def _normalize_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for key, value in state_dict.items():
      mapped = _remap_key(key) if key.startswith("transformer.") else key
      if mapped is not None:
        normalized[mapped] = value
      else:
        normalized[key] = value
    for key in list(normalized.keys()):
      if key.startswith("layers.") and not key.startswith("decoder."):
        alt = "decoder." + key
        if alt in normalized:
          del normalized[key]
    return normalized

  def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = False, verbose: bool = False) -> None:
    tiny_load_state_dict(self, state_dict, strict=strict, verbose=verbose)

  def load_weights(self, safetensor_path: str | Path, strict: bool = True, verbose: bool = True) -> None:
    state_dict = self._normalize_state_dict(safe_load(str(safetensor_path)))
    self.load_state_dict(state_dict, strict=strict, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# RF-DETR Nano uses 91 logits (COCO category id). Index 0 is unused; 1 is person.
COCO_CLASSES = (
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


def find_weights(explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        HERE / "rfdetr_nano_tinygrad.safetensors",
        HERE / "weights" / "rfdetr_nano_tinygrad.safetensors",
        HERE.parent / "weights" / "rfdetr_nano_tinygrad.safetensors",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing rfdetr_nano_tinygrad.safetensors (put it next to detect.py)"
    )


def decode(logits: np.ndarray, boxes: np.ndarray, *, score_thresh: float = 0.3, topk: int = 100):
    prob = 1.0 / (1.0 + np.exp(-logits))
    flat = prob.reshape(-1)
    idx = np.argsort(-flat)[:topk]
    scores = flat[idx]
    num_classes = logits.shape[1]
    q_idx = idx // num_classes
    c_idx = idx % num_classes
    dets = []
    for i in np.where(scores >= score_thresh)[0]:
        cx, cy, w, h = boxes[q_idx[i]]
        dets.append({
            "score": float(scores[i]),
            "label": int(c_idx[i]),
            "box": [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)],
        })
    return dets


def scale_dets(dets, meta: dict):
    size = meta["resolution"]
    orig_w, orig_h = meta["orig_w"], meta["orig_h"]
    sx = orig_w / size
    sy = orig_h / size
    out = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        out.append({
            **d,
            "box": [x0 * size * sx, y0 * size * sy, x1 * size * sx, y1 * size * sy],
        })
    return out


def draw(img: np.ndarray, dets, out_path: Path):
    colors = {c: ((i * 50) % 256, (i * 100) % 256, (i * 150) % 256) for i, c in enumerate(COCO_CLASSES)}
    h, w = img.shape[:2]
    thick = max(1, int((h + w) / 400))
    counts = defaultdict(int)
    for d in dets:
        x0, y0, x1, y1 = map(int, d["box"])
        label = COCO_CLASSES[d["label"]] if d["label"] < len(COCO_CLASSES) else str(d["label"])
        color = colors.get(label, (0, 255, 0))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick)
        txt = f"{label} {d['score']:.2f}"
        cv2.putText(img, txt, (x0, max(y0 - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        counts[label] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"saved {out_path} ({len(dets)} boxes)")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


def main() -> None:
    os.environ.setdefault("TG_SPEED_MODE", "1")
    parser = argparse.ArgumentParser(description="RF-DETR Nano (TinyGrad) detector")
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument("-o", "--output", type=Path, default=None, help="overlay jpg path")
    parser.add_argument("--weights", type=Path, default=None, help="safetensors path")
    parser.add_argument("--size", type=int, default=int(getenv("RFDETR_SIZE", 384)))
    parser.add_argument("--score", type=float, default=float(getenv("RFDETR_SCORE", 0.3)))
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    weights = find_weights(args.weights)
    batch, meta = preprocess_official_exact(image_path, size=args.size)
    x = tensor_from_batch(batch)

    model = RFDetr(num_queries=300, num_classes=91, group_detr=1)
    model.load_weights(weights, strict=True, verbose=False)

    @TinyJit
    def detect(inp: Tensor):
        out = model(inp)
        return out["pred_logits"].realize(), out["pred_boxes"].realize()

    for _ in range(2):
        detect(x)

    t0 = time.perf_counter()
    logits_t, boxes_t = detect(x)
    ms = (time.perf_counter() - t0) * 1000
    print(f"device={Device.DEFAULT} weights={weights.name} forward_e2e={ms:.1f}ms")

    logits = logits_t.numpy()[0]
    boxes = boxes_t.numpy()[0]
    dets = scale_dets(decode(logits, boxes, score_thresh=args.score), meta)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"failed to read image for overlay: {image_path}")
    out_path = args.output if args.output is not None else image_path.with_name(f"{image_path.stem}_rfdetr.jpg")
    draw(bgr, dets, out_path)


if __name__ == "__main__":
    main()

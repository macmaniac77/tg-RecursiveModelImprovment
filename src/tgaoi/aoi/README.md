# AOI library destination

AOIs are named reusable function blocks built from lower-level canonical operations or other AOIs.

## Current inventory

The package currently exports **28 AOI graph builders**:

- Core math: `MATMUL`, `SOFTMAX`
- Core NN: `RMSNORM`, `LINEAR`, `ATTENTION`
- Optimization: `ADAM_STEP`
- Vision: activation, LayerNorm, Conv2D, patch embedding, ViT self-attention, ViT MLP, windowed DINOv2/ViT block, ConvX, bottleneck, C2f, multiscale projector, DINOv2 backbone
- Detection: bilinear sampler, multi-scale deformable attention, detection MLP, DETR decoder layer, DETR decoder, hybrid encoder, RF-DETR
- Training: box IoU, Hungarian matcher, DETR set criterion

Every exported builder constructs a `Graph`, calls `Graph.validate()`, and is included in `tests/test_aoi.py`.

### What "validated" means today

The current test proves **structural graph validity**: the graph can be constructed, its references satisfy the present IR validator, and it declares outputs.

It does **not** yet mean every AOI is recursively closed or numerically equivalent to its source implementation.

The lower/core AOIs are substantially decomposed into canonical operations. Several higher-level RF-DETR-derived AOIs intentionally reference named child AOIs that do not yet have exported implementations. Examples include `LAYERSCALE`, `DROPPATH`, `NORM2D`, `MULTIHEAD_SELF_ATTENTION`, proposal-generation/refinement blocks, RF-DETR heads, and several loss/cost helpers.

Those names are design commitments, not claims of completed execution support. Per the repository contract, they must be implemented and equivalence-tested before the corresponding parent AOI can be called fully recursive/executable.

## Destination vocabulary

Examples include:

- MATMUL
- SOFTMAX
- RMSNORM
- CONV
- ROPE
- LINEAR
- ATTENTION
- MLP / SwiGLU
- optimizer steps such as Adam/AdamW
- TransformerBlock
- ViTBlock
- DiTBlock
- DETR encoder/decoder blocks
- recurrent-depth blocks
- latent-feedback blocks
- complete model families

Rules:

1. Every AOI has a versioned definition.
2. Every AOI has typed inputs/outputs and state requirements.
3. Every AOI can be recursively expanded until it reaches the primitive substrate.
4. AOIs may contain AOIs indefinitely.
5. AOIs carry provenance, aliases, purpose tags, tests, and benchmark history.
6. Framework spellings belong in the Rosetta dictionary, not in the semantic definition.
7. Discovered repeated subgraphs may be proposed as new AOIs, but should only be promoted after tests and measured reuse justify them.

Long-term objective: architecture reasoning can occur at whatever abstraction level is useful, while exact reducibility is always retained.

## Capability review (2026-09-21)

`audit.catalog()` exposes the 28 default graphs plus GELU/SILU activation variants.
Only five default graphs currently contain exclusively backend-supported ops:
MATMUL, SOFTMAX, RMSNORM, LINEAR, and ADAM_STEP. The new NumPy comparison tests
cover these, including stable softmax and all optimizer state outputs. ADAM_STEP
is explicitly **uncorrected Adam moments**, not standard Adam or AdamW.
ATTENTION still stops at AOI::SOFTMAX because nested execution is absent.

The viewer reports unsupported primitives, missing child definitions, and arity
mismatches against default child signatures. This is a capability audit, not a
complete shape/type validator or proof of source-model equivalence.

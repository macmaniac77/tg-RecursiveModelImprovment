# tg-RecursiveModelImprovment

**A reducible function-block language for AI architectures, with Tinygrad as the first execution backend.**

The premise is simple: treat neural architectures like a language.

- **letters** are small stable tensor operations,
- **words** are reusable AOIs such as softmax or RMSNorm,
- **sentences** are attention, optimizer steps, and other blocks,
- **paragraphs/chapters** are Transformer, ViT, DiT, DETR, recurrent-depth, and other model structures,
- and the **library** records the architectures, experiments, mutations, and measured outcomes.

A higher-level block is never a black box: it must remain expandable toward the primitive substrate.

## Why

The long-term target is an architecture research workbench that can:

1. import model graphs from multiple ecosystems,
2. canonicalize them into one typed IR,
3. recognize and declare reusable function blocks,
4. lower canonical architectures to Tinygrad,
5. benchmark them under controlled compute,
6. mutate architectures and training systems,
7. preserve exact experimental provenance,
8. learn from the accumulated research corpus.

The intended first laboratory is deliberately modest: **single-GPU research on an RTX 4070-class machine**, where many small controlled experiments are more valuable than attempting frontier-scale pretraining.

## Current v0.1

Implemented now:

- framework-neutral `Graph`, `Node`, and `TensorType`
- JSON round-trip serialization
- starter AOIs: `SOFTMAX`, `RMSNORM`, `LINEAR`, `ATTENTION`, `ADAM_STEP`
- a minimal Tinygrad executor for the starter alphabet
- PyTorch FX and JAXPR importer scaffolds that preserve unknown operations instead of lying about support
- Rosetta dictionary entries connecting math, Tinygrad, PyTorch, JAX, C++ reference ideas, and Lean notes
- research-loop and benchmark-design documents
- tests for IR validation and serialization

This is intentionally **not** yet an arbitrary-model converter. That claim should only be made after equivalence tests exist.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest
```

Tinygrad backend:

```bash
pip install -e '.[dev,tinygrad]'
```

## Inspect an AOI

```bash
python examples/build_attention.py
```

## Design rule

> A name is compression, not opacity.

`ATTENTION` may be convenient to reason about as one unit, but the system must be able to open it, inspect its children, mutate them, and ultimately lower them to the primitive execution substrate.

## Near-term target

```text
small PyTorch MLP
      ↓ torch.fx
canonical tg-aoi IR
      ↓
Tinygrad execution
      ↓
copy identical weights
      ↓
numerically equivalent output
```

Then repeat with JAX. After translation is trustworthy, build Transformer blocks, architecture mutation, repeated-subgraph AOI discovery, and the controlled research harness.

See `docs/ROADMAP.md`.

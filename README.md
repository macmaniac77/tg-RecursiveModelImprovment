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

## Repository map

Every major subsystem contains its own README describing what it is expected to become. Future agents should read `AGENTS.md` first, then the README nearest the code they are changing.

- `src/tgaoi/` — canonical architecture language, types, graph IR, registry, expansion/compression
- `src/tgaoi/aoi/` — reusable AOI/function-block library
- `src/tgaoi/importers/` — PyTorch/JAX/ONNX/Tinygrad → canonical IR
- `src/tgaoi/backends/` — canonical IR → Tinygrad execution first, other exporters later
- `specs/` — Rosetta Stone: math/LaTeX/TG/PyTorch/JAX/C++/Lean definitions
- `models/` — canonical complete model-family definitions
- `training/` — serializable training algorithms and curricula
- `data/` — datasets, provenance, splits, active/self-training data loops
- `benchmarks/` — measurable missions and validation protocols
- `objectives/` — multi-objective constraints / Pareto / reward hierarchy
- `experiment/` — reproducible training and measurement harness
- `mutate/` — typed architecture/training mutation operators
- `discovery/` — repeated-subgraph mining and new-AOI promotion
- `search/` — Dream-RSI-like meta-research/search policy
- `memory/` — symbolic, empirical, and learned research memory
- `viewer/` — graph/dashboard interface for human inspection and intervention

## Current state

Implemented now:

- framework-neutral `Graph`, `Node`, and `TensorType`
- JSON round-trip serialization
- **28 exported AOI graph builders** spanning core math/NN, optimization, vision, RF-DETR/detection, and training; all are covered by structural graph validation, while higher-level RF-DETR-derived AOIs still contain explicitly named child AOIs awaiting recursive implementation
- a minimal Tinygrad executor for the starter alphabet
- PyTorch FX `Linear → ReLU → Linear` lowering with explicit weights/state
- a committed PyTorch→IR→Tinygrad numerical equivalence test
- JAXPR importer scaffold
- Rosetta dictionary starter entries
- research-loop and benchmark-design documents

This is intentionally **not** yet an arbitrary-model converter. Support is earned one equivalence-tested pattern at a time.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest
```

Tinygrad + PyTorch milestone:

```bash
pip install -e '.[dev,tinygrad,torch]'
python examples/torch_fx_to_tinygrad.py
```

## Design rule

> A name is compression, not opacity.

`ATTENTION` may be convenient to reason about as one unit, but the system must be able to open it, inspect its children, mutate them, and ultimately lower them to the primitive execution substrate.

## Next hard milestone

Make AOIs genuinely recursive and executable:

```text
TRANSFORMER_BLOCK
    ↓ expand
ATTENTION + RMSNORM + MLP + residuals
    ↓ expand
SOFTMAX + LINEAR + ...
    ↓ expand
canonical primitive graph
    ↓
Tinygrad
```

Then prove equivalence against a tiny PyTorch Transformer block.

See `AGENTS.md`, `docs/ROADMAP.md`, and the README inside each subsystem.

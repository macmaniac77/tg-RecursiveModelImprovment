# Weights versus architecture

tg-RecursiveModelImprovment is an **architecture research system**, not a general model-file converter.

## The important separation

```text
architecture code / graph  ──→ canonical AOI graph
weight container           ──→ parameter tensors
parameter map              ──→ attaches tensors to AOI-owned parameters
```

These are related, but they are not the same job.

## Common formats

### .pt / .pth

PyTorch convention. A file saved by `torch.save` can contain a raw
`state_dict`, a checkpoint dictionary, or other Python objects. PyTorch
recommends saving/loading a `state_dict` for inference. The architecture must
normally be constructed separately before the state dict can be loaded.

### .bin

Not a single universal format. In Hugging Face/PyTorch model repositories,
`pytorch_model.bin` commonly contains pickled PyTorch weights/state. Treat the
surrounding framework/config as part of the model package.

### .safetensors

A deliberately simple tensor container: names, dtype, shape, offsets and raw
tensor data. It does not execute Python and is well suited to this project
because architecture stays in code/IR while the file carries parameter tensors.

### .gguf

A GGML/llama.cpp-oriented binary container with tensors plus typed key/value
metadata. It can carry substantially more model/runtime metadata than a bare
state_dict and commonly contains quantized tensors, but it is still not the
canonical architecture language for this repository.

## Project policy

Do not make the core tg-aoi library responsible for converting among
`.pt`, `.bin`, `.safetensors`, and `.gguf`.

Instead, each reference model package supplies a small **weight adapter** that
returns:

```python
dict[canonical_or_source_parameter_name, tensor]
```

The architecture importer operates on runnable code/graphs. The parameter map
joins those tensors to the canonical architecture.

If a model needs a one-time conversion (for example PyTorch state_dict →
safetensors for a Tinygrad port), keep that conversion script with the model
package or upstream tooling. Do not let checkpoint conversion semantics leak
into architecture search.

## RF-DETR policy

The supplied Tinygrad RF-DETR already loads safetensors through Tinygrad.
Use that as the baseline weight adapter. The source's `_remap_key` /
`_normalize_state_dict` logic is valuable provenance and should be represented
in the RF-DETR package, but the generic tg-aoi core should only see the resulting
parameter dictionary plus its explicit map.

## Architectural mapping

The canonical mapping should be taken from the executable code, not inferred
from the weight file. RF-DETR now has source-derived entries for ViT blocks,
convolutional projection blocks, deformable attention, DETR decoder layers,
heads, matcher and criterion in `reference_models/rfdetr/model_dictionary.yaml`.

Deep lowering happens progressively:

```text
RF_DETR
  ↓
DINO_V2_BACKBONE + HYBRID_ENCODER + DETR_DECODER + HEADS
  ↓
WINDOWED_DINOV2_BLOCK / MS_DEFORM_ATTN / DETR_DECODER_LAYER / ...
  ↓
ATTENTION / LINEAR / CONV2D / LAYERNORM / ...
  ↓
Tinygrad/UOp-level primitive substrate
```

The semantic names are retained even after lower-level expansion so architecture
search can reason at multiple abstraction depths.

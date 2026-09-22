# Tinygrad source catalog and reusable AOIs

Pinned upstream: `tinygrad/tinygrad` at
`b7ff6dedf7e385b603bf9442ee23c6adf6601fb4`.

The committed `catalog.json.gz` inventories **all 288 tracked files** under
`examples/`, `extra/models/`, and `tinygrad/llm/` at this revision: **63 Python example files and
15 shared Python model files plus 8 LLM implementation files**, with 1,463 module/class/function records. Other
files (shell scripts, browser code, assets, configuration, etc.) are inventoried
and linked, not translated. Files elsewhere in Tinygrad are outside this scope.

This is a static source projection, **not full-model canonical IR**. All 86 Python
files can be inspected without importing them, downloading weights, running their
entrypoints, or executing model code. Every source call is recorded. Recognized
qualified Tinygrad APIs map to semantic AOI families; local/imported definitions
link to source blocks; all other calls remain explicitly opaque, with optional
unverified method-name hints. Python operators, dynamic dispatch, runtime module
instances, shapes, branches and loop unrolling are not translated into dataflow.
Relative/star imports and transitive files outside the scoped directories remain
unresolved. Static API-family mapping does not imply identical parameter layout,
defaults, training behavior, or numerical equivalence.

## Open in the human viewer

```bash
python -m pip install -e '.[viewer]'
python -m tgaoi.viewer --examples reference_models/tinygrad/catalog.json.gz --output artifacts/tinygrad-viewer.html
```

Open the HTML and select **Tinygrad examples**. Search for a filename or class,
expand its source hierarchy, select a class/function and follow its calls. For
YOLOv8's `Conv_Block`, `Bottleneck`, `C2f`, and `SPPF`, select the linked library
AOI to inspect explicit parameters, decomposition and execution support.
The original RF-DETR package remains available in the same export.

## Reusable executable library blocks

| Library block | Semantics / source relationship |
|---|---|
| YOLOV8_CONV | Conv2D → frozen BatchNorm (epsilon 0.001) → SiLU |
| YOLOV8_BOTTLENECK | Two 3×3 Conv blocks; optional residual when channel counts match |
| YOLOV8_C2F | Split, configurable repeated bottlenecks, concatenation and projection |
| YOLOV8_SPPF | Three repeated max pools plus concatenation and projection |
| BATCHNORM_INFERENCE | Explicit running mean/variance and affine parameters; NCHW only |
| DENSE_MLP | Configurable linear → activation → linear, explicit biases/weights |
| SWIGLU | Bias-free down(silu(gate(x)) × up(x)) |
| RESIDUAL | Explicit x + branch |
| MAX_POOL2D / AVG_POOL2D | Configurable kernel, stride and padding |

Also upgraded existing activation, LayerNorm, Conv2D and unmasked attention
builders to executable graphs. GELU is erf-based. LayerNorm is last-axis affine
population-variance normalization. Attention has explicit scale and no mask or
dropout. High-level blocks inline child definitions with exact bindings and
namespaces, retaining `aoi_instances` metadata. Parameters and running statistics
remain explicit inputs; no source framework objects are hidden in the graph.

The four YOLOv8 blocks are checked against exact class excerpts from the pinned
upstream revision, with randomized weights and BatchNorm state, in inference
mode on small configurations. This does **not** validate full YOLOv8 detection,
weights import, decoding/NMS, training-mode BatchNorm, full RF-DETR, or all possible
configurations. Frozen BatchNorm remains differentiable with respect to supplied
values but does not perform training-statistics updates.

Some semantic-family names still lack a reusable implementation (e.g. GroupNorm,
Embedding, AdamW). They stay visible as gaps. The old RF-DETR C2F/CONVX sketches
are preserved separately; YOLOv8-specific blocks do not silently replace them.

## Refresh against a chosen upstream revision

```bash
python -m tgaoi.source_catalog --checkout /path/to/tinygrad --revision <commit-sha>
```

The generator reads committed Git objects, ignoring dirty/untracked files, hashes
every file, records source lines and pins links to the exact commit. Output is
sorted and gzip has a fixed timestamp. Use `--output catalog.json` for uncompressed
JSON. Review changed source semantics and rerun equivalence tests before updating
library claims. The CLI reports parse errors and exits nonzero if any occur.

## Validation and next expansion

`DEV=PYTHON python -m pytest -q` covers new numerical blocks, round-tripped graphs,
composition bindings, pinned catalog generation, and source-import shadowing.
`tests/viewer_catalog_smoke.cjs` adds optional jsdom interaction checks; it does
not validate browser layout or GPU performance.

Next families: rotary position embeddings, multihead/GQA layout, embeddings and
KV state; YOLOv8 DFL/head and decoding; training BatchNorm; recurrent state-space
blocks; diffusion conditioning; then whole-model equivalence and measured runs. Rotary, masked GQA and explicit DeltaNet recurrence are now available; see [Qwen blocks](QWEN_BLOCKS.md).
A source entry must pass explicit binding/type and numerical equivalence gates
before it becomes a fully executable architecture mapping.

Validation result for this change: **33 passed, 1 skipped**, plus jsdom navigation
smoke checks. Tinygrad 0.14.0 on the Python interpreter backend; PyTorch-dependent
importer test skipped. Real-browser layout and GPU performance are unverified.
Upstream code/excerpts retain the [MIT license](UPSTREAM_LICENSE).

The Qwen expansion adds ten builders; see [Qwen semantics and validation](QWEN_BLOCKS.md).

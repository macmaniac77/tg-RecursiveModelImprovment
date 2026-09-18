# RF-DETR Reference Model — Agent Handoff

This directory is the first serious guinea pig for tg-RecursiveModelImprovment.

The user intends to provide an RF-DETR model already converted to Tinygrad, including at minimum:

- `detect.py`
- `train.py`
- model weights/checkpoint
- enough model structure information to build a model dictionary and architecture map

Your job when those files arrive is **not** to rewrite RF-DETR from scratch. Your job is to wrap the existing working Tinygrad implementation into the generic reference-model contract used by this repository.

## Immediate objective

Turn the supplied RF-DETR Tinygrad implementation into a package that can be:

1. loaded;
2. run for inference;
3. trained;
4. benchmarked;
5. represented as canonical tg-aoi architecture;
6. mapped to reusable AOIs;
7. mutated by the research loop;
8. rebuilt with inherited weights;
9. re-trained;
10. evaluated and promoted/rejected by the cascade.

## First rule

Do not special-case the search engine around RF-DETR.

RF-DETR is the reference organism used to prove the generic interface. Any implementation added here should make it easier for a second arbitrary model package to plug into the same loop.

## Expected supplied files

The user will eventually replace placeholders with real files:

```text
reference_models/rfdetr/
├── detect.py
├── train.py
├── weights.pt
├── model_dictionary.yaml
├── architecture_map.yaml
├── benchmark_adapter.py
└── package.yaml
```

If additional files are required by the converted implementation, keep them under this directory or reference them explicitly from `package.yaml`.

## What to do when detect.py arrives

1. Run it unchanged first.
2. Record required inputs, preprocessing, model construction, checkpoint loading, output format, and device assumptions.
3. Identify the exact object/function that represents model inference.
4. Make the smallest adapter necessary so the generic harness can call inference.
5. Do not change model semantics until the baseline is recorded.

Record:
- input tensor shapes/dtypes;
- preprocessing;
- output tensors;
- score/class/box semantics;
- NMS/postprocessing;
- checkpoint format;
- runtime and device.

## What to do when train.py arrives

1. Run it unchanged first on the smallest practical smoke-test configuration.
2. Record:
   - model construction;
   - optimizer;
   - scheduler;
   - losses;
   - matching/Hungarian logic if present;
   - augmentation;
   - dataset loader;
   - checkpoint save/restore;
   - evaluation hook;
   - distributed/mixed precision assumptions.
3. Identify the minimum callable training interface needed by the generic experiment harness.
4. Preserve a path to run the original training code unchanged as a control.

## Architecture extraction

Build the canonical model progressively.

Do not begin by expanding everything to the lowest UOps if that destroys useful semantics.

Expected hierarchy:

```text
RF_DETR
├── BACKBONE
│   └── ...
├── MULTISCALE_FEATURES
│   └── ...
├── ENCODER
│   ├── ENCODER_BLOCK
│   │   ├── ATTENTION
│   │   ├── NORM
│   │   ├── MLP
│   │   └── RESIDUAL
│   └── ...
├── DECODER
│   ├── DECODER_BLOCK
│   └── ...
├── QUERY_EMBEDDING
├── CLASSIFICATION_HEAD
├── BOX_REGRESSION_HEAD
└── LOSSES / MATCHER / TRAINING-ONLY STRUCTURE
```

Then make each named block expandable into lower AOIs and ultimately canonical primitives.

## Model dictionary

`model_dictionary.yaml` is executable metadata, not prose.

It should define:
- canonical block names;
- source Python paths/symbols;
- parameter ownership;
- aliases;
- expected tensor roles;
- input/output meaning;
- source ↔ canonical names.

Example:

```yaml
blocks:
  encoder_attention:
    source_pattern: "model.encoder.layers.*.self_attn"
    aoi: MULTIHEAD_ATTENTION
    parameters:
      q_weight: "*.q_proj.weight"
      k_weight: "*.k_proj.weight"
      v_weight: "*.v_proj.weight"
```

## Architecture map

`architecture_map.yaml` records the actual instance-level hierarchy of the supplied model.

It should point from source objects to canonical AOIs and parameter IDs.

The map must be machine-readable so an agent can inspect, mutate, and rebuild the model without reparsing source code every loop.

## Weight handling

Do not assume `weights.pt` contains the architecture.

Determine whether it is:
- state_dict;
- checkpoint dict;
- serialized module;
- other.

Establish a parameter map:

```text
source parameter
→ canonical parameter ID
→ owning AOI
→ shape/dtype
→ inheritance rule
```

For each candidate mutation classify weights as:
- exact reuse;
- transformed;
- newly initialized;
- retired.

Record percentages and exact mappings.

## Baseline gate

Do not mutate until all of these pass:

- original `detect.py` runs;
- original/adapter training smoke test runs;
- canonical/Tinygrad model produces equivalent outputs;
- COCO or intended validation path runs;
- baseline metrics are recorded;
- timing/VRAM measurement works.

The baseline must be treated as the parent/champion zero.

## Cascade loop

Initial desired research cascade:

1. **quality first**
   - meet or beat baseline detection quality;
2. **speed second**
   - reduce inference latency while preserving quality;
3. **trainability third**
   - reduce time/FLOPs to target quality while preserving earlier gains;
4. **memory fourth**
   - reduce peak VRAM / parameter footprint without losing prior gains;
5. continue adding constrained objectives as useful.

Prefer Pareto/champion tracking over a single arbitrary weighted score.

## Search loop contract

Each iteration should look like:

```text
champion
  ↓
propose typed mutation
  ↓
validate graph / shape / cost
  ↓
build candidate
  ↓
inherit compatible weights
  ↓
proxy train
  ↓
evaluate
  ↓
promote / reject
  ↓
record lineage
  ↺
```

## Agent behavior

When continuing this work:

- read root `AGENTS.md`;
- read `docs/RUNNING.md`;
- read the README for any subsystem you modify;
- preserve exact provenance;
- do not claim arbitrary-model support until another unrelated model passes the same contract;
- prefer small working steps over broad speculative rewrites;
- keep the original RF-DETR path runnable as a regression control.

## First concrete milestone after real files arrive

```text
real detect.py + train.py + weights.pt
        ↓
run baseline
        ↓
produce model_dictionary.yaml
        ↓
produce architecture_map.yaml
        ↓
canonical graph
        ↓
rebuild/execute through generic interface
        ↓
same predictions within tolerance
        ↓
COCO baseline reproduced
```

Only after that should automated architecture mutation begin.

# Running tg-RecursiveModelImprovment

This document defines the intended end-to-end operating flow.

The system is not meant to start from a bare model name and magically infer everything. It should ingest a model package, reconstruct a canonical architecture graph, attach compatible weights/state, establish a measured baseline, then enter a constrained research cascade.

## 1. What you point it at

Preferred input is a **model manifest**:

```yaml
model:
  name: rfdetr_baseline
  framework: pytorch
  architecture:
    source: python
    module: my_model.py
    class: RFDETR
    config: configs/rfdetr.yaml
  weights:
    path: weights.pt
    format: state_dict
  example_input:
    shape: [1, 3, 640, 640]
    dtype: float32

task:
  type: detection
  dataset: coco
  split: val2017

baseline:
  metrics: [coco_ap, latency_ms, peak_vram_mb]
```

A future CLI should accept either a manifest or enough explicit flags to construct one.

## 2. Why weights.pt alone is usually insufficient

A PyTorch `.pt` / `.pth` file may contain one of several things:

1. a `state_dict` only;
2. a full pickled `nn.Module`;
3. a training checkpoint containing model + optimizer + scheduler + metadata;
4. TorchScript / exported program data;
5. a project-specific dictionary.

A bare `state_dict` contains tensors and parameter names, but not enough information to reliably reconstruct arbitrary control flow and architecture topology.

Therefore ingest follows this order:

```text
checkpoint
   ↓ identify format
architecture source/config/export available?
   ├─ yes → reconstruct executable source model
   └─ no  → attempt registry/signature match
               ├─ unique match → continue
               └─ ambiguous → stop and require architecture metadata
```

Never invent an architecture from tensor shapes and claim it is exact.

## 3. Ingest pipeline

The intended ingest command is conceptually:

```bash
tg-rmi ingest model.yaml
```

This command does not exist yet; it is the target interface.

It should perform:

```text
manifest
   ↓
load source framework model
   ↓
load checkpoint/state
   ↓
run source model on example input
   ↓
capture graph (torch.fx / torch.export / JAXPR / ONNX / Tinygrad)
   ↓
translate to canonical IR
   ↓
canonicalize equivalent operations
   ↓
recognize/compress known AOIs
   ↓
validate types/shapes/state
   ↓
lower back to Tinygrad
   ↓
copy mapped weights
   ↓
run equivalence check
   ↓
save canonical model package
```

The ingest is successful only if the canonical/Tinygrad form reproduces source outputs within an explicit tolerance for supported deterministic paths.

## 4. Canonical model package

After ingest, the system should create a package such as:

```text
artifacts/models/rfdetr_baseline/
├── model.arch.json
├── manifest.yaml
├── parameter_map.json
├── source_provenance.json
├── baseline.json
└── weights/
    └── model.safetensors
```

Recommended long-term behavior is to normalize raw framework checkpoints into a framework-neutral tensor container such as safetensors while preserving the original checkpoint hash and source reference.

The canonical package contains:

- architecture graph;
- AOI hierarchy;
- parameter names/shapes/dtypes;
- source ↔ canonical parameter mapping;
- model/task metadata;
- source checkpoint hash;
- equivalence-test result;
- baseline benchmark result.

## 5. Weight mapping

Weights are not themselves "converted into an architecture."

Architecture conversion and weight mapping are separate steps:

```text
source architecture → canonical graph
source tensors      → canonical parameter IDs
```

For an unchanged operation, weight transfer should be exact.

For a mutation:

- unchanged compatible nodes inherit weights;
- reshaped-but-related nodes may use an explicit transform rule;
- new parameters are initialized according to the mutation definition;
- removed parameters are retired but provenance is retained.

Every candidate records how much of the parent state was reused.

Example:

```json
{
  "parent": "run_0042",
  "mutation": "replace_attention_with_latent_feedback_v2",
  "weights": {
    "exact_reuse": 0.84,
    "transformed": 0.08,
    "newly_initialized": 0.08
  }
}
```

## 6. Establish the baseline before mutation

Before architecture search starts, reproduce the original model under the same benchmark harness.

For RF-DETR-like work, baseline might include:

```text
COCO AP
AP50 / AP75
latency
throughput
peak VRAM
parameter count
training FLOPs estimate
time-to-target-AP
stability / failed runs
```

The canonical model is not allowed into the search pool unless it can reproduce the expected baseline within documented tolerance.

This prevents the research loop from "improving" a broken port.

## 7. The cascade loop

The intended research loop is:

```text
CHAMPION
   ↓
generate candidate mutations
   ↓
static validation
   ↓
cheap proxy train/eval
   ↓
reject most
   ↓
promote survivors
   ↓
fuller train/eval
   ↓
apply current objective gate
   ↓
new champion / Pareto addition
   ↓
record everything
   ↺
```

A future command might look like:

```bash
tg-rmi search experiments/rfdetr_cascade.yaml
```

Again: interface target, not yet implemented.

## 8. Cascade objectives

Do not immediately collapse all goals into one reward scalar.

Example RF-DETR cascade:

### Stage A — quality

Goal:

```text
COCO AP >= parent/baseline
```

Search may modify architecture and training while preserving correctness.

### Stage B — speed

Constraint:

```text
COCO AP >= Stage A champion - tolerance
```

Objective:

```text
minimize inference latency
```

### Stage C — trainability

Constraints:

```text
quality retained
latency retained
```

Objective:

```text
minimize time/FLOPs to target AP
```

### Stage D — memory

Preserve prior gates, then minimize peak VRAM / parameter footprint.

This creates a succession of stronger models without allowing later optimization to silently destroy earlier gains.

## 9. Candidate mutation lifecycle

Each candidate should be represented as:

```text
parent architecture
+ typed graph diff
+ weight inheritance plan
+ training recipe diff
+ data/curriculum diff (if allowed)
+ objective stage
= candidate experiment
```

Possible architecture mutations:

- width/depth;
- block replacement;
- shared/recurrent depth;
- latent feedback;
- norm placement/type;
- residual/gating topology;
- attention topology;
- auxiliary heads;
- AOI composition;
- newly discovered AOI.

The system must know which dimension changed so attribution remains possible.

## 10. Training after mutation

Architecture changes generally require some training.

The loop should choose among:

- zero-shot inherited-weight evaluation;
- short calibration/fine-tuning;
- partial retraining;
- full training from scratch;
- distillation from parent/champion;
- self-training / pseudo-labeling;
- later RL/self-play where the task has a meaningful interactive reward.

Do not compare candidates trained under materially different budgets unless the benchmark explicitly intends to compare training efficiency.

## 11. Experiment promotion on constrained compute

For RTX-4070-class search, use a funnel:

```text
1000 graph proposals
  ↓ static cost/type checks
200 runnable candidates
  ↓ tiny proxy training
40 survivors
  ↓ longer training
8 survivors
  ↓ full validation
1–N Pareto champions
```

Exact numbers are task-dependent.

The point is to spend expensive compute only after cheap evidence exists.

## 12. Human intervention

A human should be able to inject:

- a new AOI;
- a new architecture family;
- a mutation idea;
- a new benchmark;
- a new objective gate;
- a new data source;
- a new training method.

The injected proposal enters the same experiment/provenance system as model-generated proposals.

Human ideas are not privileged from testing.

## 13. Search memory

During the loop, the researcher should query exact historical results:

```text
"What happened when this feedback edge was tried before?"
"Which normalization variants survived at 50M params?"
"Which mutations improved AP but hurt latency?"
```

Use graph/RAG memory for this.

Periodically, experiment history may be distilled into a local research model/LoRA, but the database remains authoritative.

## 14. Expected eventual top-level CLI

Target interface:

```bash
# inspect checkpoint / infer what metadata is available
tg-rmi inspect weights.pt

# import and prove source↔Tinygrad equivalence
tg-rmi ingest model.yaml

# inspect architecture at chosen abstraction depth
tg-rmi view artifacts/models/rfdetr_baseline/model.arch.json

# benchmark canonical baseline
tg-rmi benchmark experiments/rfdetr_baseline.yaml

# run constrained architecture search
tg-rmi search experiments/rfdetr_cascade.yaml

# inspect experiment lineage and current Pareto champions
tg-rmi report runs/rfdetr_search_001
```

Until those commands are implemented, this document defines what the eventual CLI must orchestrate.

## 15. Hard rule

The loop is never:

```text
weights.pt → magic conversion → mutate blindly
```

It is:

```text
architecture + weights + task + example I/O
        ↓
verified canonical representation
        ↓
verified Tinygrad baseline
        ↓
typed mutation + explicit weight inheritance
        ↓
controlled training
        ↓
measured benchmark
        ↓
promotion under current constraints
        ↓
repeat
```

That is the operating model for the repository.

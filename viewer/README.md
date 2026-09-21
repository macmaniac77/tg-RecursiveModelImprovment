# Viewer / dashboard

The viewer is the human instrument panel for tg-RecursiveModelImprovment.

It exists to make architecture translation, mutation, weight ownership, experiment lineage, and benchmark tradeoffs inspectable enough that a human can understand and steer automated research.

The viewer is **not** the source of truth. It reads canonical architecture, mapping, and experiment artifacts produced elsewhere in the repository.

## Core rule

> The UI must visualize the canonical model representation, not invent a second architecture representation of its own.

Do not parse arbitrary model source directly in the browser. The viewer should consume files such as:

- `model.arch.json`
- `model_dictionary.yaml`
- `architecture_map.yaml`
- parameter maps
- experiment lineage/results
- benchmark records

## Staged implementation

The stages below are intentionally ordered. Do not jump to later stages before the earlier contracts are trustworthy.

### Stage 1 — read-only architecture explorer

Goal: prove that a real canonical model can be inspected visually.

Required capabilities:

- load an RF-DETR architecture map/model package;
- show the model as an expandable hierarchy;
- allow AOIs to expand/collapse to different abstraction depths;
- click a block and inspect:
  - canonical AOI name;
  - source class/function/path;
  - inputs/outputs;
  - tensor shapes/dtypes when known;
  - child AOIs;
  - owned parameter IDs/prefixes;
  - provenance/notes;
- render a simple graph/tree without editing;
- run locally with no account/authentication requirements.

Initial reference model:

`reference_models/rfdetr/`

Suggested first UI stack:

- small Python/FastAPI backend;
- React Flow or Cytoscape.js frontend;
- local filesystem/model-package loader.

A simpler static frontend is acceptable if it reaches the milestone faster.

Completion criterion:

A user can open RF-DETR, expand `RF_DETR → DINO_V2_BACKBONE → WINDOWED_DINOV2_BLOCK → VIT_ATTENTION`, and verify source/AOI mappings without reading Python code.

### Stage 2 — paper-style architecture view

Goal: present the same canonical graph at a human explanatory level similar to figures in model papers.

Add:

- automatic grouping of repeated blocks;
- simplified model-level flow:
  - image/input;
  - backbone;
  - projector;
  - encoder;
  - decoder;
  - heads;
  - predictions;
- repeated-block notation such as `×12`;
- semantic block labels instead of low-level ops;
- switch between:
  - paper view;
  - hierarchy/tree view;
  - detailed graph view;
- export/screenshot-friendly layout.

Important:

Paper view is a projection of canonical IR. It must never become a manually maintained diagram that can drift from the actual model.

Completion criterion:

The viewer can generate a clean RF-DETR diagram understandable to someone familiar with ML architecture papers while remaining traceable back to exact source/AOI nodes.

### Stage 3 — parameter ownership and state view

Goal: make architecture ↔ weights relationships visible.

Add:

- block parameter counts;
- parameter name/prefix ownership;
- tensor shapes/dtypes;
- total parameters by subsystem;
- source checkpoint key ↔ canonical parameter ID;
- exact-reuse / transformed / new / retired state;
- weight inheritance summary after a mutation;
- optionally memory footprint estimates.

Completion criterion:

Given a parent and mutated candidate, a user can answer which blocks reused weights, which tensors changed shape, and which parameters were newly initialized.

### Stage 4 — parent/child architecture diff

Goal: make mutations visually inspectable.

Add:

- side-by-side or overlay diff;
- inserted/removed/replaced AOIs;
- changed dimensions/config values;
- graph-edge changes;
- weight inheritance diff;
- mutation rationale/proposer;
- benchmark deltas beside structural changes.

Example target information:

```text
Parent: RF_DETR_0042
Child:  RF_DETR_0042_mut17

Changed:
- decoder depth 2 → 3
- cross-attention AOI v1 → v2

Weights:
- exact reuse: 84%
- transformed: 8%
- new: 8%

Metrics:
- COCO AP: +0.6
- latency: -3%
- VRAM: +5%
```

Completion criterion:

A human can understand what a mutation actually changed without opening a source diff.

### Stage 5 — experiment and lineage dashboard

Goal: visualize automated research, not just one model.

Add:

- mutation lineage/tree;
- champion history;
- candidate queue;
- promotion/rejection reasons;
- training curves;
- experiment status;
- compute usage;
- benchmark tables;
- Pareto fronts;
- search/filter by AOI, mutation type, metric, parent, run ID.

Completion criterion:

A user can follow an automated search run from baseline through branches to current champions and understand why candidates survived or failed.

### Stage 6 — human intervention tools

Goal: let a human steer research without bypassing provenance.

Add controlled actions such as:

- annotate a block;
- nominate an AOI for mutation;
- propose a replacement block;
- pin/freeze a subsystem;
- add a new objective gate;
- queue a human-authored candidate;
- blacklist a mutation family;
- request deeper evaluation of a branch.

All interventions must enter the same experiment/provenance system as agent-generated proposals.

Do not silently edit production candidates in-place.

Completion criterion:

A human can inject architectural ideas into the research queue while preserving exact lineage and reproducibility.

### Stage 7 — architecture editor

Goal: visually compose or modify canonical architectures.

Only build this after the canonical type system, AOI expansion, validation, and mutation engine are mature.

Possible capabilities:

- drag/drop AOIs;
- connect typed ports;
- edit block parameters;
- nest AOIs;
- create reusable compound AOIs;
- validate shape/type compatibility interactively;
- serialize directly to canonical IR.

Do not build a free-form node editor that permits invalid graphs.

Completion criterion:

A graph created or modified in the UI round-trips through canonical IR, validates, lowers to Tinygrad, and can participate in the same equivalence/benchmark pipeline as code-imported models.

### Stage 8 — live research cockpit

Goal: make the viewer the human-facing control surface for recursive architecture research.

Potential capabilities:

- watch search progress live;
- inspect active candidate mutations;
- pause/promote/retire branches;
- compare hypotheses;
- query research memory;
- surface repeated architectural patterns;
- show AOIs being proposed for promotion;
- visualize search-policy behavior;
- inject external ideas/papers/models into the candidate process.

This stage should remain downstream of exact experiment records and canonical architecture artifacts.

## What not to build first

Avoid spending early effort on:

- cloud multi-user SaaS;
- authentication;
- collaborative editing;
- polished design systems;
- custom node-editor frameworks;
- browser-based source-code editing;
- real-time multi-user state;
- manual diagram files disconnected from canonical IR.

The first viewer should be useful, local, read-only, and truthful.

## Initial repository shape

A reasonable first implementation:

```text
viewer/
├── README.md
├── backend/
│   ├── app.py
│   ├── loaders.py
│   └── schemas.py
└── frontend/
    ├── package.json
    └── src/
        ├── components/
        ├── views/
        └── api/
```

If a smaller implementation gets Stage 1 working sooner, prefer it.

## First implementation milestone

Build only enough to:

1. load `reference_models/rfdetr/architecture_map.yaml`;
2. load `reference_models/rfdetr/model_dictionary.yaml`;
3. render RF-DETR as an expandable hierarchy;
4. select a node;
5. show its source mapping, AOI, children, and parameter ownership metadata;
6. provide a simplified paper-style top-level flow.

Once this works against RF-DETR, wire it to generic canonical `model.arch.json` artifacts.

## Long-term purpose

The viewer should become a truth-checking instrument.

It should help answer:

- Did we map the source model correctly?
- What does this architecture actually contain?
- What changed in this candidate?
- What weights were inherited?
- Which mutation caused a metric change?
- Are we improving the model or merely overfitting the benchmark?
- Where is the search repeatedly finding useful structure?
- Where should a human intervene?

## Run the implemented read-only viewer

From the repository root:

```bash
python -m pip install -e '.[viewer]'
python -m tgaoi.viewer --output artifacts/viewer.html
```

Open `artifacts/viewer.html` in a browser. The export embeds the map, dictionary,
and graph catalog; it needs no server, CDN, model import, weights, or GPU.
Regenerate it after changing source artifacts. Treat the exported file as a copy
of the source metadata when sharing it.

Optional canonical graph inspection:

```bash
python -m tgaoi.viewer --graph path/to/model.arch.json --output artifacts/viewer.html
```

Implemented: expandable mapped hierarchy, lazy dictionary decomposition,
node/source details, declared parameter prefixes, capability audit with search,
semantic block overview, canonical node/dependency table, and a measured-loop
review snapshot. Dictionary and graph definitions are displayed separately to
expose disagreement. Unknown counts and missing definitions stay unknown.
The overview is composition, not inferred tensor dataflow or a completed Stage 2
paper diagram. There are no editing, execution, or promotion controls.

The normalized export schema (`schema_version: 1`) contains `model`, `nodes`
(with canonical `id`, display-only `parent`, source mapping and section),
`blocks` (dictionary definitions), `catalog` (graph plus capability audit),
`parameter_materialization`, `notes`, and optional canonical `graphs`.
Parent relationships use the longest existing canonical ID prefix. Symbolic
wildcards are preserved; repeated counts are never guessed. AOI expansion has
cycle guards. Serialized data escapes HTML delimiters and UI values use text nodes.

Review findings and validation limits: [review](../docs/REVIEW_VIEWER_AND_LOOP.md).

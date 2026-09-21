# Viewer staged build plan

This file is the implementation checklist for the viewer. See `README.md` for the architectural contract.

## Stage 1 — read-only architecture explorer

Implemented as an offline HTML export (`python -m tgaoi.viewer`).
Parameter metadata remains declared prefixes; runtime counts/IDs are explicitly pending.

- [x] backend can load `architecture_map.yaml`
- [x] backend can load `model_dictionary.yaml`
- [x] normalized viewer schema exists
- [x] RF-DETR hierarchy renders
- [x] AOI nodes expand/collapse
- [x] selected-node detail panel
- [x] source path/class/function shown
- [x] child AOIs shown
- [x] parameter ownership metadata shown when available
- [x] local run instructions documented
- [x] no editing capability

Exit test:
A user can inspect RF-DETR from root AOI to nested attention block without opening source code.

## Stage 2 — paper-style view

- [ ] repeated blocks grouped
- [ ] semantic subsystem layout
- [ ] `×N` repeated-layer notation
- [ ] paper/hierarchy/detail view switcher
- [ ] stable layout generation from canonical structure
- [ ] export/screenshot-friendly output

Exit test:
The generated RF-DETR overview is understandable as a paper-style architecture figure and remains traceable to exact canonical nodes.

## Stage 3 — parameter/state view

- [ ] source checkpoint key mapping
- [ ] canonical parameter IDs
- [ ] parameter counts per block
- [ ] dtype/shape display
- [ ] parent/child inheritance classification
- [ ] new/transformed/retired tensor display
- [ ] optional memory estimates

Exit test:
A user can inspect where every mapped tensor belongs and understand inheritance after a mutation.

## Stage 4 — architecture diff

- [ ] parent/child graph comparison
- [ ] added/removed/replaced block highlighting
- [ ] config/dimension diffs
- [ ] edge/topology changes
- [ ] mutation metadata
- [ ] weight inheritance diff
- [ ] benchmark deltas

Exit test:
A mutation can be understood visually without reading a code diff.

## Stage 5 — experiment dashboard

- [ ] experiment list/search
- [ ] lineage tree
- [ ] candidate queue
- [ ] champion history
- [ ] training curves
- [ ] benchmark tables
- [ ] Pareto views
- [ ] promotion/rejection reasons
- [ ] compute/runtime summary

Exit test:
A human can follow an automated search from baseline to current champions.

## Stage 6 — human intervention

- [ ] annotations
- [ ] queue human-authored mutation
- [ ] pin/freeze subsystem
- [ ] mutation-family allow/deny controls
- [ ] objective-gate injection
- [ ] deeper-evaluation requests
- [ ] all actions recorded as provenance

Exit test:
Human ideas enter the same experiment lineage as agent proposals.

## Stage 7 — visual architecture editing

Prerequisites:
- stable typed canonical IR
- recursive AOI expansion
- graph validation
- deterministic mutation engine
- Tinygrad lowering

- [ ] typed drag/drop AOIs
- [ ] typed port connection
- [ ] block parameter editing
- [ ] nested AOI creation
- [ ] validation feedback
- [ ] serialize to canonical IR
- [ ] round-trip test
- [ ] execute/benchmark edited model

Exit test:
A visually edited model validates, lowers, runs, and participates in the normal experiment system.

## Stage 8 — live research cockpit

- [ ] live search activity
- [ ] branch/champion monitoring
- [ ] active hypothesis display
- [ ] research-memory queries
- [ ] AOI discovery/promotions
- [ ] search-policy introspection
- [ ] pause/promote/retire controls
- [ ] external idea/paper/model injection flow

Exit test:
The viewer is the human-facing control surface for the automated research loop without becoming the source of truth.

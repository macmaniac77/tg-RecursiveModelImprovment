# Roadmap

This is the chronological build order. The README inside each subsystem defines the **destination contract** for that subsystem.

## v0.1 — alphabet and first translation proof

- [x] canonical Graph / Node / TensorType
- [x] explicit starter AOIs: Softmax, RMSNorm, Linear, Attention, Adam step
- [x] JSON serialization
- [x] Tinygrad execution backend for starter operations
- [x] Rosetta dictionary examples
- [x] PyTorch FX importer scaffold
- [x] JAXPR importer scaffold
- [x] first hard translation path: PyTorch Linear→ReLU→Linear → IR → Tinygrad
- [x] numerical equivalence test committed for that path

## v0.2 — recursive AOI language

- [ ] recursively expand `AOI::...` nodes
- [ ] versioned AOI registry wired into expansion
- [ ] symbolic shape/dimension unification
- [ ] semantic axes
- [ ] explicit Parameter / Buffer / MutableState types
- [ ] graph hashing and stable canonical serialization
- [ ] graph diff
- [ ] tiny TransformerBlock assembled from AOIs
- [ ] full recursive expansion to primitive graph
- [ ] PyTorch TransformerBlock → Tinygrad numerical equivalence

## v0.3 — translation breadth

- [ ] robust PyTorch FX/torch.export importer for a documented supported subset
- [ ] JAXPR canonicalization with real shape/dtype handling
- [ ] ONNX importer
- [ ] direct Tinygrad/UOp inspection/import path
- [ ] framework alias dictionary tied to canonical specs
- [ ] unsupported-op diagnostics and decomposition workflow
- [ ] bidirectional state/parameter mapping where needed for validation
- [ ] cross-framework canonical-equivalence tests

## v0.4 — architecture/model library

- [ ] MLP/CNN/ResNet baselines
- [ ] Transformer/ViT/DiT blocks
- [ ] DETR / RF-DETR-like blocks
- [ ] recurrent-depth / looped Transformer blocks
- [ ] latent-feedback/full-bandwidth blocks
- [ ] multi-head e2e task heads
- [ ] model provenance and source-paper/repository metadata

## v0.5 — experiment + benchmark laboratory

- [ ] fixed-budget training jobs for RTX-4070-class hardware
- [ ] dataset/version/split provenance
- [ ] time-to-target-loss / convergence metrics
- [ ] latency / VRAM / FLOP / parameter measurements
- [ ] task metrics such as COCO AP
- [ ] experiment database and lineage
- [ ] early stopping / successive halving
- [ ] Pareto / successive-constraint evaluation
- [ ] hidden/OOD validation partitions

## v0.6 — mutation + AOI discovery

- [ ] typed mutation operators
- [ ] reversible architecture diffs
- [ ] repeated-subgraph mining
- [ ] candidate AOI scoring by frequency/compression/reuse
- [ ] promotion of validated discovered structures into the registry
- [ ] correlations between architectural motifs and measured outcomes

## v0.7 — automated research search

- [ ] model-assisted proposal generation
- [ ] diversity/novelty pressure
- [ ] proxy experiments and promotion policy
- [ ] replay of historical experiment trees
- [ ] human proposal injection
- [ ] learned search policy separate from executor
- [ ] periodic researcher-model/LoRA distillation from experiment history

## v0.8 — data/training co-evolution

- [ ] serializable training recipes
- [ ] optimizer/loss/curriculum mutations
- [ ] active learning and hard-example mining
- [ ] self-training / pseudo-label loops
- [ ] self-play/simulation where task structure allows it
- [ ] explicit real-world ↔ offline replay cycles
- [ ] attribution separating architecture gains from training/data gains

## Later — recursive model improvement research system

- [ ] architecture, training, data, harness, and search policy as separately mutable layers
- [ ] learned architecture proposer operating over the canonical language
- [ ] research memory available by graph/RAG without bloating context
- [ ] periodic consolidation into local weights/adapters
- [ ] architecture vocabulary capable of declaring new AOIs from discovered structure
- [ ] human-facing dashboard for inspecting, steering, and injecting new ideas
- [ ] larger-compute confirmation only after small-compute evidence justifies it
- [ ] additional execution exporters only when Tinygrad-first flow is mature

## Rule for every checkbox

A feature is not complete because code exists. It is complete when its local subsystem contract is satisfied and tests demonstrate the claimed behavior.

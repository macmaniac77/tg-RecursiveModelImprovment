# Roadmap

## v0.1 — alphabet and dictionary

- canonical Graph / Node / TensorType
- explicit starter AOIs: Softmax, RMSNorm, Linear, Attention, Adam step
- JSON serialization
- Tinygrad execution backend for starter primitive operations
- Rosetta dictionary examples
- PyTorch FX and JAXPR importer scaffolds

## v0.2 — first complete sentence

- recursively expand `AOI::...` nodes
- symbolic shape unification
- parameter/state objects
- working Tinygrad TransformerBlock
- weight-copy equivalence test against a tiny PyTorch model

## v0.3 — translation

- robust PyTorch FX importer for supported op subset
- JAXPR canonicalization
- unsupported-op preservation as opaque nodes
- graph diff and equivalence tooling

## v0.4 — AOI discovery

- repeated-subgraph mining
- candidate AOI scoring by frequency/compression/reuse
- provenance and benchmark history

## v0.5 — 4070 research harness

- fixed-budget training jobs
- early stopping / successive halving
- experiment database
- Pareto/successive-constraint evaluation
- hidden/OOD validation partitions

## v0.6 — architecture search

- typed mutation operators
- model-assisted proposal generation
- replay of historical experiment trees
- periodic researcher-model distillation

## Later

- recurrent-depth / latent-feedback model families
- vision/detection/segmentation/depth/Gaussian heads
- architecture vocabulary that can declare new AOIs from discovered repeated structure
- additional execution exporters only when Tinygrad-first flow is mature

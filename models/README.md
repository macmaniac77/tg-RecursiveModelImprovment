# Model architecture library destination

This directory should hold canonical architecture definitions, not framework source dumps.

Expected families include:

- MLP/CNN baselines
- Transformer language models
- ViT
- DiT
- DETR / RF-DETR-like systems
- recurrent-depth / looped transformers
- latent-feedback / full-bandwidth variants
- MoE / routing systems
- SSM/Mamba-like systems
- multimodal/e2e models with multiple heads
- vision heads for classification, detection, segmentation, depth, optical flow, Gaussian representations, and action prediction

Each model definition should include:

- canonical graph/AOI composition;
- parameterization/config schema;
- source paper/repository provenance;
- known framework implementations;
- benchmark history in this repository;
- known equivalent or related architectures;
- a path to expand to primitives.

Do not copy weights into Git. Store references/checksums and use experiment artifacts or external storage.

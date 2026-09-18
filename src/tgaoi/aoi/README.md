# AOI library destination

AOIs are named reusable function blocks built from lower-level canonical operations or other AOIs.

Examples:

- MATMUL
- SOFTMAX
- RMSNORM
- CONV
- ROPE
- LINEAR
- ATTENTION
- MLP / SwiGLU
- optimizer steps such as Adam/AdamW
- TransformerBlock
- ViTBlock
- DiTBlock
- DETR encoder/decoder blocks
- recurrent-depth blocks
- latent-feedback blocks
- complete model families

Rules:

1. Every AOI has a versioned definition.
2. Every AOI has typed inputs/outputs and state requirements.
3. Every AOI can be recursively expanded until it reaches the primitive substrate.
4. AOIs may contain AOIs indefinitely.
5. AOIs carry provenance, aliases, purpose tags, tests, and benchmark history.
6. Framework spellings belong in the Rosetta dictionary, not in the semantic definition.
7. Discovered repeated subgraphs may be proposed as new AOIs, but should only be promoted after tests and measured reuse justify them.

Long-term objective: architecture reasoning can occur at whatever abstraction level is useful, while exact reducibility is always retained.

# tgaoi package destination

This package is the core architecture language.

It should eventually contain:

- a typed, serializable, framework-neutral graph IR;
- symbolic dimensions and semantic axes;
- explicit parameter, buffer, optimizer-state, and mutable-state objects;
- a registry of versioned reusable AOIs;
- recursive AOI expansion/compression;
- graph matching and canonicalization;
- provenance attached to imported, generated, or discovered structures;
- graph hashing so equivalent architectures can be deduplicated;
- cost metadata (parameters, FLOPs, memory estimates, measured runtime);
- validation that rejects malformed architecture graphs before execution.

The package should **not** become the experiment scheduler, user interface, benchmark database, or model-training service. Those belong in their own subsystems.

Completion criterion: an architecture can be serialized, compared, expanded to primitives, lowered to Tinygrad, reconstructed, and traced back to its source/provenance without framework-specific hidden state.

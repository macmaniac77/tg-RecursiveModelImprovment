# Agent / Contributor Contract

This repository is intended to become a machine-readable and machine-editable research system for neural architecture translation, composition, benchmarking, and search.

Before changing code, preserve these invariants:

1. **A name is compression, not opacity.** Higher-level AOIs must remain expandable toward the primitive substrate.
2. **Canonical IR is framework-neutral.** PyTorch, JAX, ONNX, Tinygrad, etc. are source/target dialects, not the truth representation.
3. **Tinygrad is the first execution target.** Do not add additional exporters merely for symmetry while the Tinygrad path is incomplete.
4. **Never silently fake support.** Unsupported constructs become explicit OPAQUE nodes or hard errors.
5. **Parameters/state are explicit.** Do not hide framework objects inside canonical graphs.
6. **Every translation needs an equivalence test.** Structural translation is not considered complete until numerical or semantic equivalence is demonstrated.
7. **Search must preserve provenance.** Every mutation, training run, benchmark, seed, hardware context, and result must be reconstructable.
8. **Benchmarks are not reality.** Keep search-visible metrics distinct from hidden/OOD/real-world validation where possible.
9. **Prefer composable primitives over framework-specific convenience APIs.**
10. **Do not optimize the research harness into an opaque monolith.** The system must remain inspectable by humans and agents.

## Development order

The intended dependency direction is:

```text
specs/types
   ↓
canonical IR
   ↓
AOI definitions + registry
   ↓
importers / canonicalizers
   ↓
Tinygrad lowering + execution
   ↓
equivalence tests
   ↓
complete architecture families
   ↓
benchmark harness
   ↓
mutation + AOI discovery
   ↓
research/search loop
   ↓
learned researcher / recursive improvement
```

When adding a feature, update the README in the subsystem where the feature belongs and add a test that proves the claimed behavior.

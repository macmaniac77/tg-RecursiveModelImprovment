# Backends destination

Backends execute or emit canonical tg-aoi graphs.

## Near-term

Tinygrad is the reference backend and should mature first.

Tinygrad backend should eventually support:

- the complete chosen primitive alphabet;
- recursive AOI expansion before execution;
- trainable parameters and optimizer state;
- forward and backward execution;
- mixed precision and quantized representations where meaningful;
- device selection and arbitrary Tinygrad-supported hardware;
- measured runtime/VRAM/FLOP instrumentation;
- deterministic test modes;
- export of lowered Tinygrad/UOp information for analysis.

## Later

Only after the Tinygrad path is trustworthy, optional exporters may target PyTorch, JAX, C++, etc. Those are useful for validation and portability, but should not distract from the canonical-IR → Tinygrad path.

A backend is complete for an operation only when equivalence tests exist.

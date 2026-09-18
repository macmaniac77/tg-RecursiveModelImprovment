# Research loop

Long-term loop:

1. Import or define an architecture in canonical IR.
2. Expand or compress reusable AOIs as needed.
3. Propose a mutation.
4. Validate grammar/types.
5. Lower to Tinygrad.
6. Train under a controlled compute budget.
7. Measure task quality, convergence, latency, VRAM, FLOPs, and stability.
8. Store exact architecture + mutation + conditions + result.
9. Promote only candidates that satisfy current constraints.
10. Periodically mine repeated subgraphs as candidate AOIs.
11. Periodically distill the experiment corpus into a research model/adapter; do not rely on model context as the sole memory.

## Memory tiers

- **Symbolic memory:** exact architecture/AOI definitions.
- **Research memory:** searchable experiment database / graph / RAG.
- **Learned intuition:** occasional training or adapter updates from accumulated experiments.

The first two are authoritative. Learned intuition is a proposal mechanism, not the record of truth.

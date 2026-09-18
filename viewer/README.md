# Viewer / dashboard destination

The viewer should make the research process legible enough for a human to intervene intelligently.

Expected views:

- architecture graph at adjustable abstraction depth;
- click-to-expand AOIs down toward primitives;
- side-by-side architecture diff;
- mutation lineage/tree;
- benchmark Pareto fronts;
- training curves;
- compute/VRAM/runtime comparisons;
- AOI provenance and where-used graph;
- experiment search;
- candidate queue and promotion/rejection reasons;
- human annotations/proposals.

The dashboard is not the source of truth. It reads from canonical architecture and experiment stores.

Long-term value: a human should be able to watch automated research, notice conceptual blind spots, inject new architecture ideas, and inspect whether claimed gains survive broader validation.

# Architecture mutation destination

This subsystem proposes typed changes to canonical architecture graphs.

Expected mutation operators include:

- add/remove/replace AOI;
- change width/depth/head count;
- share/unshare weights;
- add recurrent depth;
- add/remove feedback edges;
- alter residual/gating structure;
- swap normalization/activation;
- alter attention topology;
- add/remove auxiliary heads;
- compose known AOIs;
- parameterize discovered compound blocks.

Every mutation must:

- produce a valid typed graph;
- retain a reversible diff from its parent;
- record the operator and rationale/proposer;
- estimate expected cost changes before expensive training;
- be rejectable before execution if constraints are violated.

Later, models may propose mutations, but the mutation engine remains the deterministic authority that applies them.

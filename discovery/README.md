# AOI discovery destination

This subsystem mines the architecture/experiment corpus for reusable structure.

Expected capabilities:

- repeated-subgraph discovery across imported and generated models;
- graph isomorphism/canonical hashing;
- scoring candidate AOIs by frequency, compression gain, cross-family reuse, and measured usefulness;
- proposing names/semantic tags for repeated structures;
- identifying architectural motifs correlated with benchmark improvements;
- detecting equivalent structures expressed differently by source frameworks.

Promotion path:

```text
repeated structure
   ↓
candidate AOI
   ↓
equivalence + type tests
   ↓
cross-model reuse evidence
   ↓
versioned registry entry
```

Do not promote arbitrary compression artifacts merely because they are frequent.

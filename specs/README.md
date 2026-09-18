# Formal specifications destination

This directory should become the human- and machine-readable specification layer for operations and AOIs.

For each important concept, record:

- plain-English meaning;
- LaTeX mathematical definition;
- tensor/type contract;
- shape and axis invariants;
- numerical-stability requirements;
- canonical AOI expansion;
- Tinygrad spelling/lowering;
- PyTorch spelling;
- JAX spelling;
- C/C++ reference form where useful;
- optional Lean definition/proof hooks;
- known aliases and variants;
- equivalence tests.

This is the project's Rosetta Stone.

Lean is optional and should initially be used only where formalization buys something concrete: algebraic equivalence, type/shape invariants, or transformations whose correctness matters. Do not block practical architecture work on formal proof coverage.

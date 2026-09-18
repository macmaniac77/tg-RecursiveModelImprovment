# Rosetta dictionary

Each architecture concept should eventually carry:

- mathematical definition / LaTeX
- exact AOI expansion
- tensor/type contract
- Tinygrad expression or lowering
- PyTorch spelling
- JAX spelling
- C/C++ reference implementation when useful
- Lean specification/proof hooks where worthwhile
- aliases and framework-specific variants
- numerical invariants and equivalence tests
- provenance: where the definition came from and which experiments use it

The JSON files in `dictionary/` are the starter examples. This is intentionally data-first so an agent can search, compare, and extend the dictionary without having to modify Python source for every mapping.

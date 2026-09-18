# Importers destination

Importers translate external model/program representations into canonical tg-aoi IR.

Expected importers:

- PyTorch FX / torch.export
- JAXPR
- ONNX
- direct Tinygrad graph/UOp inspection
- later: TensorFlow/MLIR or other useful dialects

Each importer must:

1. preserve tensor shapes/dtypes/state where available;
2. map recognized operations to canonical operations/AOIs;
3. preserve unsupported constructs as explicit OPAQUE nodes rather than guessing;
4. attach source-framework and source-symbol provenance;
5. provide parameter extraction/state mapping;
6. have numerical equivalence tests for every declared supported pattern.

Long-term goal: importing equivalent architectures from different frameworks should canonicalize to structurally equivalent IR.

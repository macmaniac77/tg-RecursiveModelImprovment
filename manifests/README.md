# Model manifests

A model manifest tells the ingest system what a checkpoint means.

A checkpoint is not assumed to contain enough information to reconstruct the architecture.

Expected manifest fields:

```yaml
model:
  name:
  framework:
  architecture:
    source:
    module:
    class:
    config:
  weights:
    path:
    format:
  example_input:
    shape:
    dtype:

task:
  type:
  dataset:
  split:

baseline:
  metrics:
```

Later manifests may describe multiple inputs/outputs, tokenizer/preprocessor state, dynamic shapes, quantization, distributed topology, and external assets.

The manifest becomes part of canonical model provenance.

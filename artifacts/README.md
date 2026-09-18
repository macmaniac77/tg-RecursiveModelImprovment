# Generated artifacts

This path documents the expected generated artifact layout.

Large model weights and experiment outputs should generally **not** be committed to Git.

Expected local/runtime structure:

```text
artifacts/
├── models/
│   └── <canonical_model>/
│       ├── model.arch.json
│       ├── manifest.yaml
│       ├── parameter_map.json
│       ├── source_provenance.json
│       ├── baseline.json
│       └── weights/
└── runs/
    └── <experiment_id>/
        ├── experiment.yaml
        ├── parent.arch.json
        ├── candidate.arch.json
        ├── mutation.json
        ├── training.json
        ├── metrics.json
        ├── lineage.json
        └── checkpoints/
```

The experiment database should index these artifacts and their hashes.

Do not rely on directory names as identity; use stable graph/checkpoint/run hashes.
